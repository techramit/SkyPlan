#!/usr/bin/env python3
"""
SkyPlan inference runner.

This script executes the SkyPlan environment with LLM-driven agents and emits
the hackathon-required stdout format for each task episode.
"""

import asyncio
import argparse
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


def _bootstrap_agentenv_import_path() -> None:
    """Ensure AgentEnv imports work in both packaged and flattened layouts."""

    current_file = Path(__file__).resolve()
    search_roots = [current_file.parent, *current_file.parents]

    for root in search_roots:
        if (root / "AgentEnv" / "__init__.py").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return

    # Validator fallback: inference.py may be copied to a flat workspace without
    # an AgentEnv/ directory, alongside client.py/models.py/etc.
    flat_root = current_file.parent
    required_files = {"client.py", "models.py", "tasks.py", "workflow.py", "prompts.py"}
    if all((flat_root / filename).exists() for filename in required_files):
        flat_root_str = str(flat_root)
        if flat_root_str not in sys.path:
            sys.path.insert(0, flat_root_str)

        if "AgentEnv" not in sys.modules:
            synthetic_pkg = types.ModuleType("AgentEnv")
            synthetic_pkg.__path__ = [flat_root_str]
            synthetic_pkg.__file__ = str(flat_root / "__init__.py")
            sys.modules["AgentEnv"] = synthetic_pkg


_bootstrap_agentenv_import_path()

import AgentEnv.prompts as agent_prompts
from AgentEnv.client import AgentenvEnv
from AgentEnv.models import SkyPlanAction, SkyPlanObservation
from AgentEnv.tasks import TASKS
from AgentEnv.workflow import (
    get_all_agent_ids,
    get_allowed_actions,
    get_agent_name,
    get_required_documents,
)

BENCHMARK = "skyplan"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for the inference runner."""

    api_base_url: str
    model_name: str
    api_key: str
    image_name: str
    task_selector: str
    temperature: float
    max_tokens: int
    max_action_chars: int
    step_recovery_attempts: int


class ModelRequestError(RuntimeError):
    """Raised when the model request fails and the episode cannot continue."""


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse supported command-line arguments."""

    parser = argparse.ArgumentParser(description="Run SkyPlan inference episodes.")
    parser.add_argument(
        "--task",
        default=None,
        help="Task id to run, or 'all' to run every task. Defaults to SKYPLAN_TASK or all.",
    )
    return parser.parse_args(argv)


def _is_huggingface_router(api_base_url: str) -> bool:
    """Return True when the configured endpoint points at the Hugging Face router."""

    return "huggingface.co" in api_base_url.lower()


def _resolve_inference_credentials() -> tuple[str, str, bool]:
    """Resolve the inference endpoint and API key with provider-aware fallbacks."""

    inference_api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    legacy_api_key = os.getenv("SKYPLAN_LLM_API_KEY")

    if inference_api_key:
        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        return api_base_url, inference_api_key, False

    if legacy_api_key:
        api_base_url = os.getenv(
            "SKYPLAN_LLM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
        return api_base_url, legacy_api_key, True

    raise ValueError(
        "Inference credentials are required. Set HF_TOKEN or API_KEY for the Hugging Face "
        "router, or set SKYPLAN_LLM_API_KEY for the SkyPlan-compatible inference endpoint."
    )


def get_runtime_config(task_override: str | None = None) -> RuntimeConfig:
    """Resolve runtime configuration after environment variables are loaded."""

    api_base_url, api_key, uses_legacy_endpoint = _resolve_inference_credentials()

    if uses_legacy_endpoint:
        model_name = os.getenv("SKYPLAN_LLM_MODEL") or os.getenv("MODEL_NAME") or "meta/llama-3.1-405b-instruct"
    else:
        default_model_name = (
            "Qwen/Qwen2.5-72B-Instruct"
            if _is_huggingface_router(api_base_url)
            else "meta/llama-3.1-405b-instruct"
        )
        model_name = os.getenv("MODEL_NAME", default_model_name)

    return RuntimeConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        image_name=os.getenv("IMAGE_NAME", "skyplan-env"),
        task_selector=(task_override or os.getenv("SKYPLAN_TASK", "all")).strip(),
        temperature=float(os.getenv("SKYPLAN_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("SKYPLAN_MAX_TOKENS", "2000")),
        max_action_chars=int(os.getenv("SKYPLAN_MAX_ACTION_CHARS", "16000")),
        step_recovery_attempts=int(os.getenv("SKYPLAN_STEP_RECOVERY_ATTEMPTS", "1")),
    )


def log_start(task: str, env: str, model: str) -> None:
    """Emit the start line required by the judges."""

    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    """Emit a step line in the exact required format."""

    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    """Emit the episode end line in the exact required format."""

    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def log_error(message: str) -> None:
    """Send non-protocol errors to stderr so stdout stays judge-compliant."""

    print(message, file=sys.stderr, flush=True)


def build_user_prompt(
    step: int,
    agent_id: str,
    observation: SkyPlanObservation,
    task_description: str,
) -> str:
    """Build a compact user prompt for the acting agent."""

    context_parts = [
        f"Step: {step} of {observation.total_steps}",
        f"Your role: {get_agent_name(agent_id)}",
        f"Task: {task_description}",
    ]

    selected_documents = _select_document_context(agent_id, observation)
    if selected_documents:
        context_parts.append("\nExisting planning artifacts:")
        for doc_type, document in selected_documents:
            preview = document.content.strip()
            if len(preview) > 240:
                preview = preview[:240] + "\n... [truncated]"
            context_parts.append(f"\n- {doc_type} ({document.status}):\n{preview}")
        omitted_documents = len(observation.documents) - len(selected_documents)
        if omitted_documents > 0:
            context_parts.append(f"\nAdditional artifacts omitted for brevity: {omitted_documents}")

    unresolved_feedback = [
        item
        for item in observation.feedback
        if not item.resolved and item.to_agent in {"", agent_id}
    ]
    if unresolved_feedback:
        context_parts.append("\nOutstanding feedback to consider:")
        for feedback in unresolved_feedback[-3:]:
            target = feedback.to_agent or "team"
            context_parts.append(
                f"\n- from={feedback.from_agent} to={target} doc={feedback.document_type or 'general'}: "
                f"{feedback.comment}"
            )
        omitted_feedback = len(unresolved_feedback) - 3
        if omitted_feedback > 0:
            context_parts.append(f"\nAdditional relevant feedback omitted for brevity: {omitted_feedback}")

    if observation.document_status_summary:
        context_parts.append(
            "\nDocument status summary: "
            + ", ".join(
                f"{status}={count}"
                for status, count in observation.document_status_summary.items()
            )
        )

    if observation.documents_awaiting_review:
        context_parts.append(
            "\nDocuments awaiting review: " + ", ".join(observation.documents_awaiting_review)
        )

    if observation.last_action_result:
        context_parts.append(
            f"\nLast action result: {observation.last_action_result.get_summary()}"
        )

    if observation.current_state:
        context_parts.append(
            f"\nCurrent phase: {observation.current_state.get('phase', 'unknown')}"
        )

    context_parts.append(
        "\nRespond as JSON with keys action_type, reasoning, and content. "
        "Your content should advance the workflow and directly address any unresolved feedback relevant to your role."
    )
    return "\n".join(context_parts)


def _select_document_context(
    agent_id: str,
    observation: SkyPlanObservation,
) -> list[tuple[str, Any]]:
    """Select the highest-signal document subset for the acting agent."""

    selected_doc_types: list[str] = []
    for doc_type in get_required_documents(agent_id):
        if doc_type in observation.documents and doc_type not in selected_doc_types:
            selected_doc_types.append(doc_type)

    for doc_type in list(observation.documents.keys())[-2:]:
        if doc_type not in selected_doc_types:
            selected_doc_types.append(doc_type)

    return [
        (doc_type, observation.documents[doc_type])
        for doc_type in selected_doc_types
        if doc_type in observation.documents
    ]


def parse_agent_response(response: str, agent_id: str) -> dict[str, str]:
    """Parse the model response into a valid action payload."""

    allowed_actions = get_allowed_actions(agent_id)
    default_action = allowed_actions[0] if allowed_actions else "UNKNOWN"

    try:
        data = json.loads(response)
        action_type = str(data.get("action_type", default_action))
        if action_type not in allowed_actions:
            action_type = default_action

        return {
            "action_type": action_type,
            "reasoning": str(data.get("reasoning", "")).strip() or "Providing the next workflow deliverable.",
            "content": str(data.get("content", "")).strip(),
        }
    except json.JSONDecodeError:
        return {
            "action_type": default_action,
            "reasoning": "Providing the next workflow deliverable based on the available context.",
            "content": response.strip(),
        }


def _sanitize_text(value: str, *, max_chars: int) -> str:
    """Normalize model text before sending it over the WebSocket."""

    cleaned = "".join(
        character
        for character in value
        if character in {"\n", "\r", "\t"} or ord(character) >= 32
    )
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def sanitize_action_payload(
    action_data: dict[str, str],
    agent_id: str,
    runtime_config: RuntimeConfig,
) -> dict[str, str]:
    """Clamp and normalize model output into a transport-safe action payload."""

    allowed_actions = get_allowed_actions(agent_id)
    default_action = allowed_actions[0] if allowed_actions else "UNKNOWN"
    action_type = action_data.get("action_type", default_action)
    if action_type not in allowed_actions:
        action_type = default_action

    reasoning = _sanitize_text(
        action_data.get("reasoning", "") or "Providing the next workflow deliverable.",
        max_chars=2000,
    )
    content = _sanitize_text(
        action_data.get("content", ""),
        max_chars=max(runtime_config.max_action_chars, 1),
    )

    return {
        "action_type": action_type,
        "reasoning": reasoning or "Providing the next workflow deliverable.",
        "content": content,
    }


def _is_retryable_step_error(exc: Exception) -> bool:
    """Identify transient transport failures that are worth replaying once."""

    message = str(exc).lower()
    retryable_markers = (
        "no close frame received or sent",
        "connection closed",
        "connection reset",
        "broken pipe",
        "websocket",
        "keepalive ping timeout",
    )
    return any(marker in message for marker in retryable_markers)


def _task_reset_kwargs(task_id: str, task_config: Any) -> dict[str, Any]:
    """Build reset kwargs from the task configuration."""

    return {
        "task_description": task_config.description,
        "task_keywords": task_config.required_keywords,
        "task_difficulty": task_config.difficulty,
        "required_sections": task_config.required_sections,
        "task_id": task_id,
    }


async def _reset_task_environment(
    env: AgentenvEnv,
    task_id: str,
    task_config: Any,
) -> SkyPlanObservation:
    """Reset an environment instance for a specific task."""

    reset_result = await env.reset(**_task_reset_kwargs(task_id, task_config))
    return reset_result.observation


async def _recover_episode_environment(
    current_env: AgentenvEnv,
    runtime_config: RuntimeConfig,
    task_id: str,
    task_config: Any,
    completed_actions: list[SkyPlanAction],
) -> tuple[AgentenvEnv, SkyPlanObservation]:
    """Recreate the environment and replay successful actions after a transport failure."""

    try:
        await current_env.close()
    except Exception as exc:
        log_error(f"[step-recover] close error before reconnect for task {task_id}: {exc}")

    recovered_env = await AgentenvEnv.from_docker_image(runtime_config.image_name)
    observation = await _reset_task_environment(recovered_env, task_id, task_config)

    for replay_action in completed_actions:
        replay_result = await recovered_env.step(replay_action)
        observation = replay_result.observation
        if observation.errors:
            raise RuntimeError(
                f"Replay failed after {replay_action.agent_id}:{replay_action.action_type}: "
                f"{observation.errors[-1]}"
            )

    return recovered_env, observation


def get_model_message(
    client: OpenAI,
    step: int,
    agent_id: str,
    observation: SkyPlanObservation,
    task_description: str,
    runtime_config: RuntimeConfig,
) -> dict[str, str]:
    """Get the current agent's action proposal from the model."""

    system_prompt = agent_prompts.get_agent_prompt(agent_id)
    user_prompt = build_user_prompt(step, agent_id, observation, task_description)

    try:
        completion = client.chat.completions.create(
            model=runtime_config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=runtime_config.temperature,
            max_tokens=runtime_config.max_tokens,
            response_format={"type": "json_object"},
        )
        response_text = completion.choices[0].message.content or ""
        return parse_agent_response(response_text, agent_id)
    except Exception as exc:
        raise ModelRequestError(f"{agent_id}: {exc}") from exc


def format_action_string(action: dict[str, str]) -> str:
    """Format an action for the step log line."""

    return f"{action.get('agent_id', 'unknown')}:{action.get('action_type', 'UNKNOWN')}"


async def run_episode(
    client: OpenAI,
    env: AgentenvEnv,
    task_id: str,
    task_config: Any,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    """Run a single task episode end to end."""

    rewards: list[float] = []
    steps_taken = 0
    observation: SkyPlanObservation | None = None
    encountered_error = False
    completed_actions: list[SkyPlanAction] = []

    try:
        observation = await _reset_task_environment(env, task_id, task_config)
    except Exception as exc:
        log_error(f"[reset-error] task={task_id}: {exc}")
        return {"success": False, "steps": 0, "rewards": rewards, "env": env}

    for step_number, agent_id in enumerate(get_all_agent_ids(), start=1):
        if observation.done:
            break

        try:
            action_data = get_model_message(
                client=client,
                step=step_number,
                agent_id=agent_id,
                observation=observation,
                task_description=task_config.description,
                runtime_config=runtime_config,
            )
        except ModelRequestError as exc:
            encountered_error = True
            steps_taken = step_number
            rewards.append(0.0)
            log_error(f"[model-error] {exc}")
            log_step(
                step=step_number,
                action=f"{agent_id}:MODEL_REQUEST_FAILED",
                reward=0.0,
                done=False,
                error=str(exc),
            )
            break

        action_data = sanitize_action_payload(action_data, agent_id, runtime_config)
        action_data["agent_id"] = agent_id

        try:
            action = SkyPlanAction(
                agent_id=agent_id,
                action_type=action_data["action_type"],
                reasoning=action_data["reasoning"],
                content=action_data["content"],
            )
        except Exception as exc:
            log_error(f"[action-error] task={task_id} step={step_number}: {exc}")
            rewards.append(0.0)
            steps_taken = step_number
            log_step(
                step=step_number,
                action=format_action_string(action_data),
                reward=0.0,
                done=False,
                error=str(exc),
            )
            break

        step_error: str | None = None
        done = False
        reward = 0.0

        attempts_remaining = runtime_config.step_recovery_attempts
        while True:
            try:
                result = await env.step(action)
                observation = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                if observation.errors:
                    step_error = observation.errors[-1]
                break
            except Exception as exc:
                if attempts_remaining > 0 and _is_retryable_step_error(exc):
                    attempts_remaining -= 1
                    log_error(
                        f"[step-recover] task={task_id} step={step_number}: "
                        f"restarting env after transport error: {exc}"
                    )
                    try:
                        env, observation = await _recover_episode_environment(
                            current_env=env,
                            runtime_config=runtime_config,
                            task_id=task_id,
                            task_config=task_config,
                            completed_actions=completed_actions,
                        )
                    except Exception as recovery_exc:
                        step_error = str(recovery_exc)
                        encountered_error = True
                        log_error(
                            f"[step-error] task={task_id} step={step_number}: "
                            f"recovery failed: {recovery_exc}"
                        )
                        break
                    continue

                step_error = str(exc)
                encountered_error = True
                log_error(f"[step-error] task={task_id} step={step_number}: {exc}")
                break

        rewards.append(reward)
        steps_taken = step_number
        log_step(
            step=step_number,
            action=format_action_string(action_data),
            reward=reward,
            done=done,
            error=step_error,
        )

        if step_error:
            encountered_error = True
            break
        completed_actions.append(action)
        if done:
            break

    success = bool(observation and observation.done) and not encountered_error
    return {"success": success, "steps": steps_taken, "rewards": rewards, "env": env}


def resolve_task_ids(task_selector: str) -> list[str]:
    """Resolve the task selector into concrete task ids."""

    if not task_selector or task_selector.lower() == "all":
        return list(TASKS.keys())
    if task_selector not in TASKS:
        raise ValueError(f"Unknown task: {task_selector}")
    return [task_selector]


async def main(argv: list[str] | None = None) -> None:
    """Execute the configured task set against the local environment image."""

    load_env_file()
    args = parse_cli_args(argv)

    try:
        runtime_config = get_runtime_config(args.task)
        task_ids = resolve_task_ids(runtime_config.task_selector)
    except ValueError as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc

    client = OpenAI(
        base_url=runtime_config.api_base_url,
        api_key=runtime_config.api_key,
    )
    for task_id in task_ids:
        task_config = TASKS[task_id]
        env: AgentenvEnv | None = None
        result: dict[str, Any] = {"success": False, "steps": 0, "rewards": [], "env": None}

        log_start(task=task_id, env=BENCHMARK, model=runtime_config.model_name)
        try:
            env = await AgentenvEnv.from_docker_image(runtime_config.image_name)
            result = await run_episode(client, env, task_id, task_config, runtime_config)
        except Exception as exc:
            log_error(f"Failed to execute task {task_id}: {exc}")
        finally:
            env_to_close = result.get("env") or env
            if env_to_close is not None:
                try:
                    await env_to_close.close()
                except Exception as exc:
                    log_error(f"env.close() error for task {task_id}: {exc}")
            log_end(
                success=result["success"],
                steps=result["steps"],
                rewards=result["rewards"],
            )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
