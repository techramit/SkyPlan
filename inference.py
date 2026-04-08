#!/usr/bin/env python3
"""
SkyPlan inference runner.

This script executes the SkyPlan environment with LLM-driven agents and emits
the hackathon-required stdout format for each task episode.
"""

import asyncio
import json
import os
import sys
from typing import Any

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AgentEnv"))

import prompts as agent_prompts

from AgentEnv import SkyPlanAction, SkyPlanObservation, TASKS, get_all_agent_ids, get_allowed_actions, get_agent_name
from AgentEnv.client import AgentenvEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "AgentEnv-env:latest")

TASK_SELECTOR = os.getenv("SKYPLAN_TASK", "all").strip()
TEMPERATURE = float(os.getenv("SKYPLAN_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("SKYPLAN_MAX_TOKENS", "2000"))
SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "skyplan"


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

    if observation.documents:
        context_parts.append("\nExisting planning artifacts:")
        for doc_type, document in observation.documents.items():
            preview = document.content.strip()
            if len(preview) > 500:
                preview = preview[:500] + "\n... [truncated]"
            context_parts.append(f"\n- {doc_type} ({document.status}):\n{preview}")

    unresolved_feedback = [item for item in observation.feedback if not item.resolved]
    if unresolved_feedback:
        context_parts.append("\nOutstanding feedback to consider:")
        for feedback in unresolved_feedback[-5:]:
            target = feedback.to_agent or "team"
            context_parts.append(
                f"\n- from={feedback.from_agent} to={target} doc={feedback.document_type or 'general'}: "
                f"{feedback.comment}"
            )

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
) -> dict[str, str]:
    """Get the current agent's action proposal from the model."""

    system_prompt = agent_prompts.get_agent_prompt(agent_id)
    user_prompt = build_user_prompt(step, agent_id, observation, task_description)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
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
) -> dict[str, Any]:
    """Run a single task episode end to end."""

    rewards: list[float] = []
    steps_taken = 0
    observation: SkyPlanObservation | None = None

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
            break
        if done:
            break

    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success = average_reward >= SUCCESS_SCORE_THRESHOLD
    return {"success": success, "steps": steps_taken, "rewards": rewards}


def resolve_task_ids() -> list[str]:
    """Resolve the task selector into concrete task ids."""

    if not TASK_SELECTOR or TASK_SELECTOR.lower() == "all":
        return list(TASKS.keys())
    if TASK_SELECTOR not in TASKS:
        raise ValueError(f"Unknown task: {TASK_SELECTOR}")
    return [TASK_SELECTOR]


async def main() -> None:
    """Execute the configured task set against the local environment image."""

    if not HF_TOKEN:
        log_error("HF_TOKEN or API_KEY environment variable is required")
        raise SystemExit(1)

    try:
        task_ids = resolve_task_ids()
    except ValueError as exc:
        log_error(str(exc))
        raise SystemExit(1) from exc

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        env = await AgentenvEnv.from_docker_image(IMAGE_NAME)
    except Exception as exc:
        log_error(f"Failed to connect to environment: {exc}")
        raise SystemExit(1) from exc

    try:
        for task_id in task_ids:
            task_config = TASKS[task_id]
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            result = await run_episode(client, env, task_id, task_config)
            log_end(
                success=result["success"],
                steps=result["steps"],
                rewards=result["rewards"],
            )
    finally:
        try:
            await env.close()
        except Exception as exc:
            log_error(f"env.close() error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
