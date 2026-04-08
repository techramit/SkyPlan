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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

import AgentEnv.prompts as agent_prompts
from AgentEnv import SkyPlanAction, SkyPlanObservation, TASKS, get_all_agent_ids, get_allowed_actions, get_agent_name
from AgentEnv.client import AgentenvEnv
from AgentEnv.workflow import get_required_documents

BENCHMARK = "skyplan"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for the inference runner."""

    api_base_url: str
    model_name: str
    hf_token: str
    image_name: str
    task_selector: str
    temperature: float
    max_tokens: int


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


def get_runtime_config(task_override: str | None = None) -> RuntimeConfig:
    """Resolve runtime configuration after environment variables are loaded."""

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
        or os.getenv("SKYPLAN_LLM_API_KEY")
    )
    if not hf_token:
        raise ValueError(
            "HF_TOKEN or API_KEY environment variable is required. "
            "Set it in your shell or local .env file."
        )

    return RuntimeConfig(
        api_base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        hf_token=hf_token,
        image_name=os.getenv("IMAGE_NAME", "skyplan-env"),
        task_selector=(task_override or os.getenv("SKYPLAN_TASK", "all")).strip(),
        temperature=float(os.getenv("SKYPLAN_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("SKYPLAN_MAX_TOKENS", "2000")),
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
        log_error(f"[model-error] {agent_id}: {exc}")
        return {
            "action_type": get_allowed_actions(agent_id)[0],
            "reasoning": "Falling back to a safe default action because the model request failed.",
            "content": f"# {get_agent_name(agent_id)} Output\nProvide the next planning artifact for: {task_description}",
        }


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

    try:
        reset_result = await env.reset(
            task_description=task_config.description,
            task_keywords=task_config.required_keywords,
            task_difficulty=task_config.difficulty,
            required_sections=task_config.required_sections,
            task_id=task_id,
        )
        observation = reset_result.observation
    except Exception as exc:
        log_error(f"[reset-error] task={task_id}: {exc}")
        return {"success": False, "steps": 0, "rewards": rewards}

    for step_number, agent_id in enumerate(get_all_agent_ids(), start=1):
        if observation.done:
            break

        action_data = get_model_message(
            client=client,
            step=step_number,
            agent_id=agent_id,
            observation=observation,
            task_description=task_config.description,
            runtime_config=runtime_config,
        )
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

        try:
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            if observation.errors:
                step_error = observation.errors[-1]
        except Exception as exc:
            step_error = str(exc)
            encountered_error = True
            log_error(f"[step-error] task={task_id} step={step_number}: {exc}")

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
        if done:
            break

    success = bool(observation and observation.done) and not encountered_error
    return {"success": success, "steps": steps_taken, "rewards": rewards}


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
        api_key=runtime_config.hf_token,
    )
    for task_id in task_ids:
        task_config = TASKS[task_id]
        env: AgentenvEnv | None = None
        result: dict[str, Any] = {"success": False, "steps": 0, "rewards": []}

        log_start(task=task_id, env=BENCHMARK, model=runtime_config.model_name)
        try:
            env = await AgentenvEnv.from_docker_image(runtime_config.image_name)
            result = await run_episode(client, env, task_id, task_config, runtime_config)
        except Exception as exc:
            log_error(f"Failed to execute task {task_id}: {exc}")
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception as exc:
                    log_error(f"env.close() error for task {task_id}: {exc}")
            log_end(
                success=result["success"],
                steps=result["steps"],
                rewards=result["rewards"],
            )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
