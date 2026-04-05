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
        ACTION_TO_DOCUMENT,
        Document,
        DocumentType,
        Feedback,
        LastAction,
        RewardConfig,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
    )
    from ..workflow import (
        get_first_agent,
        get_handoff_message,
        get_next_agent,
    )
except ImportError:
    from models import (
        ACTION_TO_DOCUMENT,
        Document,
        DocumentType,
        Feedback,
        LastAction,
        RewardConfig,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
    )
    from workflow import (
        get_first_agent,
        get_handoff_message,
        get_next_agent,
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

    def __init__(self, total_steps: int = WorkflowConfig.DEFAULT_TOTAL_STEPS):
        """Initialize the SkyPlan environment with isolated state.

        Args:
            total_steps: Total number of steps for the planning workflow
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = get_first_agent()
        self._step_number = 1
        self._total_steps = total_steps
        self._documents: dict[str, Document] = {}
        self._feedback: list[Feedback] = []
        self._last_action_result: LastAction | None = None
        self._task_description = ""
        self._done = False

    def reset(self) -> SkyPlanObservation:
        """
        Reset the environment for a new episode.

        Returns:
            SkyPlanObservation with initial state, ready for the first agent to start
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = get_first_agent()
        self._step_number = 1
        self._documents = {}
        self._feedback = []
        self._last_action_result = None
        self._task_description = "Create a comprehensive planning document for the given idea."
        self._done = False

        return self._build_observation(
            result="Environment reset. Ready to begin planning.",
            reasoning="Starting new planning episode.",
            status="ready",
        )

    def step(self, action: SkyPlanAction) -> SkyPlanObservation:  # type: ignore[override]
        """
        Execute a step in the environment - "The Hand-Off".

        This is where the supervisor does their job:
        1. The Inspection: Check if the agent turned in valid work
        2. The Filing: Save the document to the Shared Folder
        3. The Next Person: Tell the next agent it's their turn
        4. The Performance Review: Give a reward based on quality

        Args:
            action: SkyPlanAction containing agent_id, action_type, reasoning, and content

        Returns:
            SkyPlanObservation with updated state
        """
        self._state.step_count += 1

        # 1. The Inspection: Validate action matches current agent
        if not self._is_valid_agent(action):
            return self._create_error_observation(
                f"Expected agent {self._current_agent}, got {action.agent_id}"
            )

        # 2. The Inspection: Validate document quality
        validation_result = self._validate_action(action)
        if not validation_result["is_valid"]:
            self._last_action_result = LastAction.create(
                agent_id=action.agent_id,
                action_type=action.action_type,
                result="failure",
                message=validation_result["error"],
            )
            return self._create_error_observation(validation_result["error"])

        # 3. The Filing: Save document to Shared Folder
        self._file_document(action)

        # 4. The Performance Review: Calculate reward based on quality
        reward = self._calculate_reward(action)

        # 5. The Next Person: Move to next agent
        next_agent = get_next_agent(self._current_agent)
        if next_agent:
            self._current_agent = next_agent
            self._step_number += 1
        else:
            # No next agent - this is the end of the workflow
            self._done = True

        # 6. Check if episode is done
        if self._step_number > self._total_steps:
            self._done = True

        # 7. Create LastAction record
        self._last_action_result = LastAction.create(
            agent_id=action.agent_id,
            action_type=action.action_type,
            result="success",
            message=f"{action.action_type} completed successfully",
        )

        # Get handoff message for the next agent
        handoff_msg = get_handoff_message(action.agent_id)
        reasoning_msg = f"Action processed by {action.agent_id}. {handoff_msg}"

        return self._build_observation(
            result=f"{action.action_type} completed successfully",
            reasoning=reasoning_msg,
            status="in_progress",
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

    # ==========================================================================
    # Private Methods - Internal Implementation
    # ==========================================================================

    def _is_valid_agent(self, action: SkyPlanAction) -> bool:
        """Check if the action is from the expected agent.

        Args:
            action: The action to validate

        Returns:
            True if agent matches current_agent, False otherwise
        """
        return action.agent_id == self._current_agent

    def _validate_action(self, action: SkyPlanAction) -> dict:
        """
        The Inspection: Validate if the agent turned in valid work.

        Args:
            action: The action to validate

        Returns:
            Dict with 'is_valid' (bool) and 'error' (str if invalid)
        """
        # Check if content meets minimum length
        if not action.content or len(action.content.strip()) < ValidationConfig.MIN_CONTENT_LENGTH:
            return {
                "is_valid": False,
                "error": f"Content too short or empty for action {action.action_type}",
            }

        # Check if reasoning meets minimum length
        if not action.reasoning or len(action.reasoning.strip()) < ValidationConfig.MIN_REASONING_LENGTH:
            return {
                "is_valid": False,
                "error": "Reasoning too short or empty",
            }

        return {"is_valid": True, "error": ""}

    def _file_document(self, action: SkyPlanAction) -> None:
        """
        The Filing: Save the document to the Shared Folder.

        Maps action_type to document_type and stores the document.

        Args:
            action: The action containing the document content
        """
        doc_type = ACTION_TO_DOCUMENT.get(action.action_type)
        if not doc_type:
            return  # Action doesn't produce a document

        timestamp = datetime.utcnow().isoformat() + "Z"

        if doc_type in self._documents:
            # Update existing document
            self._documents[doc_type].content = action.content
            self._documents[doc_type].author = action.agent_id
            self._documents[doc_type].updated_at = timestamp
        else:
            # Create new document
            self._documents[doc_type] = Document(
                type=doc_type,
                content=action.content,
                author=action.agent_id,
                created_at=timestamp,
                updated_at=timestamp,
                status="draft",
            )

    def _calculate_reward(self, action: SkyPlanAction) -> float:
        """
        The Performance Review: Calculate reward based on quality.

        Args:
            action: The action to score

        Returns:
            Reward in range 0.0 to 1.0
        """
        # Base reward for completing action
        reward = RewardConfig.BASE_REWARD

        # Content length bonus
        content_length = len(action.content)
        length_bonus = min(
            content_length / RewardConfig.CONTENT_LENGTH_TARGET,
            RewardConfig.CONTENT_LENGTH_WEIGHT,
        )
        reward += length_bonus

        # Reasoning quality bonus
        reasoning_length = len(action.reasoning)
        reasoning_bonus = min(
            reasoning_length / RewardConfig.REASONING_TARGET,
            RewardConfig.REASONING_WEIGHT,
        )
        reward += reasoning_bonus

        # Document structure bonus
        structure_bonus = self._calculate_structure_bonus(action.content)
        reward += structure_bonus

        # Cap at maximum reward
        return min(reward, RewardConfig.MAX_REWARD)

    def _calculate_structure_bonus(self, content: str) -> float:
        """Calculate structure bonus based on document content.

        Args:
            content: The document content to analyze

        Returns:
            Structure bonus in range 0.0 to STRUCTURE_WEIGHT
        """
        bonus = 0.0
        weight_per_element = RewardConfig.STRUCTURE_WEIGHT / 4

        # Has headers
        if "##" in content:
            bonus += weight_per_element

        # Has lists
        if "-" in content or "*" in content:
            bonus += weight_per_element

        # Has multiple paragraphs
        if content.count("\n") > 5:
            bonus += weight_per_element

        # Has key structural keywords
        keywords = ["overview", "summary", "goal", "objective"]
        if any(keyword in content.lower() for keyword in keywords):
            bonus += weight_per_element

        return bonus

    def _build_observation(
        self,
        result: str,
        reasoning: str,
        status: str,
        reward: float = 0.0,
    ) -> SkyPlanObservation:
        """Build a SkyPlanObservation with current state.

        Args:
            result: Result message
            reasoning: System reasoning
            status: Current status
            reward: Reward value

        Returns:
            Complete SkyPlanObservation
        """
        return SkyPlanObservation(
            task_description=self._task_description,
            result=result,
            reasoning=reasoning,
            current_agent=self._current_agent,
            step_number=self._step_number,
            total_steps=self._total_steps,
            documents=self._documents,
            feedback=self._feedback,
            last_action_result=self._last_action_result,
            current_state={"status": status, "phase": self._get_current_phase()},
            errors=[],
            step_count=self._state.step_count,
            done=self._done,
            reward=reward,
        )

    def _create_error_observation(self, error_message: str) -> SkyPlanObservation:
        """Create an observation with an error state.

        Args:
            error_message: The error message to include

        Returns:
            SkyPlanObservation with error state
        """
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
        """Get the current planning phase based on step number.

        Returns:
            Current phase name
        """
        phase_index = min(
            (self._step_number - 1) // 2,
            len(WorkflowConfig.PHASES) - 1,
        )
        return WorkflowConfig.PHASES[phase_index]
