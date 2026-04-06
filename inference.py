#!/usr/env python3
"""
SkyPlan Inference Script

This script runs the SkyPlan environment with LLM-powered agents.
It manages the complete workflow from start to finish, emitting standardized
output for hackathon evaluation.

Environment Variables:
    API_BASE_URL: The API endpoint for the LLM
    MODEL_NAME: The model identifier to use for inference
    HF_TOKEN: Your Hugging Face / API key
    IMAGE_NAME: The name of the local image to use (if using docker)
    SKYPLAN_TASK: Task to run (easy_user_authentication, medium_chat_app, hard_saas_platform)
    SKYPLAN_MAX_STEPS: Maximum steps per episode (default: 6)
    SKYPLAN_TEMPERATURE: LLM temperature (default: 0.0)
    SKYPLAN_MAX_TOKENS: Max tokens per LLM call (default: 2000)
"""

import asyncio
import json
import os
import sys
from typing import Optional

from openai import OpenAI

# Add AgentEnv to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AgentEnv"))

# Import prompts directly to avoid openenv dependency
import prompts as agent_prompts

from AgentEnv import (
    AgentId,
    SkyPlanAction,
    SkyPlanObservation,
    get_all_agent_ids,
    get_allowed_actions,
    get_agent_name,
    get_next_agent,
    get_workflow_entry,
    TASKS,
    get_task_summary,
)
from AgentEnv.client import AgentenvEnv
from AgentEnv.reward import RewardCalculator


# ============================================================================
# Configuration
# ============================================================================

# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME", "AgentEnv-env:latest")

# Task configuration
TASK_NAME = os.getenv("SKYPLAN_TASK", "easy_user_authentication")
MAX_STEPS = int(os.getenv("SKYPLAN_MAX_STEPS", "6"))
TEMPERATURE = float(os.getenv("SKYPLAN_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("SKYPLAN_MAX_TOKENS", "2000"))

# Success threshold
SUCCESS_SCORE_THRESHOLD = 0.5

# Benchmark name
BENCHMARK = "skyplan"

# ============================================================================
# Logging Functions
# ============================================================================


def log_start(task: str, env: str, model: str) -> None:
    """Log the start of an episode.

    Args:
        task: Task name
        env: Environment/benchmark name
        model: Model name
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Log a step in the episode.

    Args:
        step: Step number
        action: Action string representation
        reward: Reward value
        done: Whether episode is done
        error: Error message or None
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """Log the end of an episode.

    Args:
        success: Whether the episode was successful
        steps: Number of steps taken
        score: Final score
        rewards: List of all rewards
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# Agent System Prompts
# ============================================================================

# Agent prompts are imported directly from prompts module to avoid openenv dependency
# Use agent_prompts.get_agent_prompt(agent_id) to get the system prompt for each agent


def build_user_prompt(
    step: int,
    agent_id: str,
    observation: SkyPlanObservation,
    task_description: str,
) -> str:
    """Build the user prompt for an agent.

    Args:
        step: Current step number
        agent_id: The agent ID
        observation: Current observation from environment
        task_description: The task description

    Returns:
        User prompt for the agent
    """
    agent_name = get_agent_name(agent_id)

    # Build context from observation
    context_parts = [
        f"Step: {step} of {observation.total_steps}",
        f"Your role: {agent_name}",
        f"Task: {task_description}",
    ]

    # Add information about previous work
    if observation.documents:
        context_parts.append("\nPrevious work produced:")
        for doc_type, doc in observation.documents.items():
            context_parts.append(f"\n- {doc_type}: {doc.content[:200]}..." if len(doc.content) > 200 else f"\n- {doc_type}: {doc.content}")

    # Add feedback if any
    if observation.feedback:
        context_parts.append("\nFeedback from previous agents:")
        for feedback in observation.feedback[-3:]:  # Last 3 feedback entries
            context_parts.append(f"\n- {feedback.from_agent}: {feedback.comment}")

    # Add last action result if available
    if observation.last_action_result:
        context_parts.append(
            f"\nLast action result: {observation.last_action_result.get_summary()}"
        )

    # Add current state
    if observation.current_state:
        context_parts.append(f"\nCurrent phase: {observation.current_state.get('phase', 'unknown')}")

    context_parts.append("\n\nYour task: Complete your assigned work and produce the required document.")

    return "\n".join(context_parts)


def parse_agent_response(response: str, agent_id: str) -> dict:
    """Parse the agent's LLM response into action components.

    Args:
        response: The LLM response text
        agent_id: The agent ID

    Returns:
        Dictionary with action_type, reasoning, and content
    """
    try:
        # Try to parse as JSON
        data = json.loads(response)

        # Validate required fields
        if "action_type" not in data:
            raise ValueError("Missing action_type in response")

        # Set defaults for optional fields
        data.setdefault("reasoning", "")
        data.setdefault("content", "")

        # Validate action_type is allowed
        allowed_actions = get_allowed_actions(agent_id)
        if data["action_type"] not in allowed_actions:
            # Try to find closest match
            for action in allowed_actions:
                if action.lower() in data["action_type"].lower():
                    data["action_type"] = action
                    break
            else:
                raise ValueError(f"Invalid action_type: {data['action_type']}")

        return data
    except json.JSONDecodeError:
        # Fallback: treat entire response as content
        # Try to infer action_type from content
        content = response.strip()
        action_type = get_allowed_actions(agent_id)[0] if get_allowed_actions(agent_id) else "UNKNOWN"

        return {
            "action_type": action_type,
            "reasoning": "Generated from free-form response",
            "content": content,
        }


def get_model_message(
    client: OpenAI,
    step: int,
    agent_id: str,
    observation: SkyPlanObservation,
    task_description: str,
) -> dict:
    """Get the model's response for the current step.

    Args:
        client: OpenAI client
        step: Current step number
        agent_id: The agent ID
        observation: Current observation
        task_description: Task description

    Returns:
        Dictionary with action_type, reasoning, and content
    """
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
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback response
        return {
            "action_type": get_allowed_actions(agent_id)[0],
            "reasoning": "Model request failed, using default",
            "content": "Default content due to error",
        }


def format_action_string(action: dict) -> str:
    """Format an action dictionary as a string for logging.

    Args:
        action: Action dictionary

    Returns:
        Formatted action string
    """
    return f"{action.get('agent_id', 'unknown')}:{action.get('action_type', 'UNKNOWN')}"


# ============================================================================
# Main Inference Loop
# ============================================================================


async def run_episode(
    client: OpenAI,
    env,
    task_id: str,
    task_config: dict,
) -> dict:
    """Run a single episode for a task.

    Args:
        client: OpenAI client
        env: Environment instance
        task_id: Task ID
        task_config: Task configuration

    Returns:
        Dictionary with success, steps, score, rewards
    """
    task_description = task_config.get("description", "")
    task_keywords = task_config.get("required_keywords", [])
    task_difficulty = task_config.get("difficulty", "medium")
    required_sections = task_config.get("required_sections", [])

    # Initialize tracking
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error = None
    documents: dict = {}  # Accumulate documents for final scoring

    # Reset environment with task configuration
    try:
        result = await env.reset(
            task_description=task_description,
            task_keywords=task_keywords,
            task_difficulty=task_difficulty,
            required_sections=required_sections,
        )
        observation = result.observation
    except Exception as e:
        print(f"[DEBUG] Environment reset failed: {e}", flush=True)
        return {
            "success": False,
            "steps": 0,
            "score": 0.0,
            "rewards": [],
        }

    # Get workflow order
    workflow_agents = get_all_agent_ids()

    # Run through workflow
    for step, agent_id in enumerate(workflow_agents, 1):
        # Check if episode is done
        if observation.done:
            break

        # Get model response
        action_data = get_model_message(
            client,
            step,
            agent_id,
            observation,
            task_description,
        )

        # Create action
        action = SkyPlanAction(
            agent_id=agent_id,
            action_type=action_data["action_type"],
            reasoning=action_data["reasoning"],
            content=action_data["content"],
        )

        # Step environment
        try:
            result = await env.step(action)
            observation = result.observation
            reward = result.reward or 0.0
            done = result.done
            last_error = None

            # Accumulate documents from observation
            documents = observation.documents
        except Exception as e:
            print(f"[DEBUG] Environment step failed: {e}", flush=True)
            reward = 0.0
            done = False
            last_error = str(e)

        # Track
        rewards.append(reward)
        steps_taken = step

        # Log step
        action_str = format_action_string(action_data)
        log_step(
            step=step,
            action=action_str,
            reward=reward,
            done=done,
            error=last_error,
        )

        # Check if done
        if done:
            break

    # Calculate final score
    # Normalize score to [0, 1]
    if rewards:
        score = sum(rewards) / len(rewards)
    else:
        score = 0.0
    score = max(0.0, min(score, 1.0))

    # Determine success
    success = score >= SUCCESS_SCORE_THRESHOLD

    return {
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


async def main() -> None:
    """Main inference function."""
    # Validate required environment variables
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN or API_KEY environment variable is required", flush=True)
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to environment
    try:
        env = await AgentenvEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to connect to environment: {e}", flush=True)
        sys.exit(1)

    # Get task configuration
    task_config = TASKS.get(TASK_NAME)
    if not task_config:
        print(f"[ERROR] Unknown task: {TASK_NAME}", flush=True)
        sys.exit(1)

    # Log start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Run episode
    result = await run_episode(client, env, TASK_NAME, task_config)

    # Log end
    log_end(
        success=result["success"],
        steps=result["steps"],
        score=result["score"],
        rewards=result["rewards"],
    )

    # Close environment
    try:
        await env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
