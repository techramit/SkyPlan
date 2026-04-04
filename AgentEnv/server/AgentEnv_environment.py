# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SkyPlan Environment Implementation.

A multi-agent planning environment where specialized agents collaborate
to transform an idea into structured planning documents.
"""

from datetime import datetime
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        AgentId,
        Document,
        DocumentType,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
    )
except ImportError:
    from models import (
        AgentId,
        Document,
        DocumentType,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
    )


class SkyPlanEnvironment(Environment):
    """
    Multi-agent planning environment for SkyPlan.

    Six specialized agents collaborate to produce planning documents:
    - Maya (Research Analyst) → Research Summary
    - Elon (Product Manager) → PRD, feature list
    - Jordan (Architect) → TRD, system architecture
    - Robert (Execution Planner) → Roadmap, sprint backlog
    - Taylor (Validator) → Validation report
    - Sam (CEO) → Final strategic approval

    Workflow: Maya → Elon → Jordan → Robert → Taylor → Sam
    """

    # Enable concurrent WebSocket sessions.
    # Each session has its own isolated state, allowing multiple
    # projects to run simultaneously.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the SkyPlan environment with isolated state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = AgentId.MAYA.value
        self._step_number = 1
        self._total_steps = 10
        self._documents: dict[str, Document] = {}
        self._feedback: list[Feedback] = []
        self._last_action_result: LastAction | None = None
        self._task_description = ""
        self._done = False

    def reset(self) -> SkyPlanObservation:
        """
        Reset the environment for a new episode.

        Returns:
            SkyPlanObservation with initial state, ready for Maya to start
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = AgentId.MAYA.value
        self._step_number = 1
        self._total_steps = 10
        self._documents = {}
        self._feedback = []
        self._last_action_result = None
        self._task_description = "Create a comprehensive planning document for the given idea."
        self._done = False

        return SkyPlanObservation(
            task_description=self._task_description,
            result="Environment reset. Ready to begin planning.",
            reasoning="Starting new planning episode with Maya (Research Analyst).",
            current_agent=self._current_agent,
            step_number=self._step_number,
            total_steps=self._total_steps,
            documents=self._documents,
            feedback=self._feedback,
            last_action_result=self._last_action_result,
            current_state={"status": "ready", "phase": "research"},
            errors=[],
            step_count=self._state.step_count,
            done=False,
            reward=0.0,
        )

    def step(self, action: SkyPlanAction) -> SkyPlanObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: SkyPlanAction containing agent_id, action_type, reasoning, and content

        Returns:
            SkyPlanObservation with updated state
        """
        self._state.step_count += 1

        # Validate action matches current agent
        if action.agent_id != self._current_agent:
            return self._create_error_observation(
                f"Expected agent {self._current_agent}, got {action.agent_id}"
            )

        # Process action and update state
        result, reward = self._process_action(action)

        # Move to next agent
        self._current_agent = AgentId.get_next_agent(self._current_agent)
        self._step_number += 1

        # Check if episode is done
        if self._step_number > self._total_steps:
            self._done = True

        return SkyPlanObservation(
            task_description=self._task_description,
            result=result,
            reasoning=f"Action processed by {action.agent_id}. Moving to {self._current_agent}.",
            current_agent=self._current_agent,
            step_number=self._step_number,
            total_steps=self._total_steps,
            documents=self._documents,
            feedback=self._feedback,
            last_action_result=self._last_action_result,
            current_state={"status": "in_progress", "phase": self._get_current_phase()},
            errors=[],
            step_count=self._state.step_count,
            done=self._done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _process_action(self, action: SkyPlanAction) -> tuple[str, float]:
        """Process an action and return (result_message, reward)."""
        # Create LastAction record
        self._last_action_result = LastAction.create(
            agent_id=action.agent_id,
            action_type=action.action_type,
            result="success",
            message=f"{action.action_type} completed",
        )

        # Simple reward based on content length (placeholder for quality scoring)
        reward = min(len(action.content) / 1000.0, 1.0)

        return f"{action.action_type} completed successfully", reward

    def _create_error_observation(self, error_message: str) -> SkyPlanObservation:
        """Create an observation with an error state."""
        return SkyPlanObservation(
            task_description=self._task_description,
            result="Action failed",
            reasoning=error_message,
            current_agent=self._current_agent,
            step_number=self._step_number,
            total_steps=self._total_steps,
            documents=self._documents,
            feedback=self._feedback,
            last_action_result=self._last_action_result,
            current_state={"status": "error"},
            errors=[error_message],
            step_count=self._state.step_count,
            done=False,
            reward=0.0,
        )

    def _get_current_phase(self) -> str:
        """Get the current planning phase based on step number."""
        phases = ["research", "product", "architecture", "planning", "validation", "strategy"]
        phase_index = min((self._step_number - 1) // 2, len(phases) - 1)
        return phases[phase_index]
