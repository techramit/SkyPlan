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
        if action.agent_id != self._current_agent:
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
        next_agent = AgentId.get_next_agent(self._current_agent)
        self._current_agent = next_agent
        self._step_number += 1

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

        return SkyPlanObservation(
            task_description=self._task_description,
            result=f"{action.action_type} completed successfully",
            reasoning=f"Action processed by {action.agent_id}. Moving to {next_agent}.",
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

    def _validate_action(self, action: SkyPlanAction) -> dict:
        """
        The Inspection: Validate if the agent turned in valid work.

        Args:
            action: The action to validate

        Returns:
            Dict with 'is_valid' (bool) and 'error' (str if invalid)
        """
        # Check if content is empty
        if not action.content or len(action.content.strip()) < 10:
            return {
                "is_valid": False,
                "error": f"Content too short or empty for action {action.action_type}",
            }

        # Check if reasoning is provided
        if not action.reasoning or len(action.reasoning.strip()) < 5:
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
        # Map action types to document types
        action_to_doc = {
            "SEARCH_MARKET": DocumentType.RESEARCH.value,
            "ANALYZE_COMPETITORS": DocumentType.RESEARCH.value,
            "VALIDATE_PROBLEM": DocumentType.RESEARCH.value,
            "SUMMARIZE_INSIGHTS": DocumentType.RESEARCH.value,
            "IDENTIFY_OPPORTUNITIES": DocumentType.RESEARCH.value,
            "WRITE_PRD": DocumentType.PRD.value,
            "DEFINE_FEATURES": DocumentType.PRD.value,
            "IDENTIFY_USER_PERSONA": DocumentType.PRD.value,
            "PRIORITIZE_FEATURES": DocumentType.PRD.value,
            "DEFINE_SUCCESS_METRICS": DocumentType.PRD.value,
            "DESIGN_ARCHITECTURE": DocumentType.ARCHITECTURE.value,
            "SELECT_TECH_STACK": DocumentType.TRD.value,
            "DEFINE_APIS": DocumentType.TRD.value,
            "DESIGN_DATA_MODEL": DocumentType.TRD.value,
            "WRITE_TRD": DocumentType.TRD.value,
            "CREATE_ROADMAP": DocumentType.ROADMAP.value,
            "BREAK_INTO_TASKS": DocumentType.TASKS.value,
            "PLAN_SPRINTS": DocumentType.TASKS.value,
            "ESTIMATE_TIMELINES": DocumentType.ROADMAP.value,
            "DEFINE_DEPENDENCIES": DocumentType.TASKS.value,
            "REVIEW_DOCUMENTS": DocumentType.VALIDATION.value,
            "CHECK_CONSISTENCY": DocumentType.VALIDATION.value,
            "VALIDATE_CLAIMS": DocumentType.VALIDATION.value,
            "IDENTIFY_RISKS": DocumentType.VALIDATION.value,
            "SCORE_PLAN": DocumentType.VALIDATION.value,
            "SET_DIRECTION": DocumentType.STRATEGY.value,
            "REVIEW_PLAN": DocumentType.STRATEGY.value,
            "APPROVE_STRATEGY": DocumentType.STRATEGY.value,
            "PRIORITIZE_OBJECTIVES": DocumentType.STRATEGY.value,
            "REQUEST_REVISION": DocumentType.STRATEGY.value,
        }

        doc_type = action_to_doc.get(action.action_type)
        if not doc_type:
            return  # Action doesn't produce a document

        # Create or update document
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
        reward = 0.1

        # Content length bonus (up to 0.3)
        content_length = len(action.content)
        length_bonus = min(content_length / 1000.0, 0.3)
        reward += length_bonus

        # Reasoning quality bonus (up to 0.2)
        reasoning_length = len(action.reasoning)
        reasoning_bonus = min(reasoning_length / 200.0, 0.2)
        reward += reasoning_bonus

        # Document structure bonus (up to 0.4)
        structure_bonus = 0.0
        if "##" in action.content:  # Has headers
            structure_bonus += 0.1
        if "-" in action.content or "*" in action.content:  # Has lists
            structure_bonus += 0.1
        if action.content.count("\n") > 5:  # Has multiple paragraphs
            structure_bonus += 0.1
        if any(keyword in action.content.lower() for keyword in ["overview", "summary", "goal", "objective"]):
            structure_bonus += 0.1
        reward += structure_bonus

        # Cap at 1.0
        return min(reward, 1.0)

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
