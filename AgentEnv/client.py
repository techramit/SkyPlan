# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agentenv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SkyPlanAction, SkyPlanObservation
from .workflow import get_first_agent


class AgentenvEnv(
    EnvClient[SkyPlanAction, SkyPlanObservation, State]
):
    """
    Client for the SkyPlan Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AgentenvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_agent)
        ...
        ...     action = SkyPlanAction(
        ...         agent_id="maya",
        ...         action_type="SEARCH_MARKET",
        ...         reasoning="I need to research the market first",
        ...         content="Market research content here..."
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.result)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AgentenvEnv.from_docker_image("AgentEnv-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SkyPlanAction) -> Dict:
        """
        Convert SkyPlanAction to JSON payload for step message.

        Args:
            action: SkyPlanAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "agent_id": action.agent_id,
            "action_type": action.action_type,
            "reasoning": action.reasoning,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SkyPlanObservation]:
        """
        Parse server response into StepResult[SkyPlanObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SkyPlanObservation
        """
        obs_data = payload.get("observation", {})
        observation = SkyPlanObservation(
            task_description=obs_data.get("task_description", ""),
            result=obs_data.get("result", ""),
            reasoning=obs_data.get("reasoning", ""),
            current_agent=obs_data.get("current_agent", get_first_agent()),
            step_number=obs_data.get("step_number", 1),
            total_steps=obs_data.get("total_steps", 10),
            documents=obs_data.get("documents", {}),
            feedback=obs_data.get("feedback", []),
            last_action_result=obs_data.get("last_action_result"),
            current_state=obs_data.get("current_state", {}),
            errors=obs_data.get("errors", []),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
