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
        DocumentStatus,
        DocumentStatusConfig,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
    )
    from ..reward import RewardCalculator, RewardConfig
    from ..workflow import (
        get_first_agent,
        get_handoff_message,
        get_next_agent,
        get_required_documents,
    )
except ImportError:
    from models import (
        ACTION_TO_DOCUMENT,
        Document,
        DocumentType,
        DocumentStatus,
        DocumentStatusConfig,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
    )
    from reward import RewardCalculator, RewardConfig
    from workflow import (
        get_first_agent,
        get_handoff_message,
        get_next_agent,
        get_required_documents,
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

    def __init__(
        self,
        total_steps: int = WorkflowConfig.DEFAULT_TOTAL_STEPS,
        use_llm_reward: bool = True,
        llm_api_key: str | None = None,
    ):
        """Initialize the SkyPlan environment with isolated state.

        Args:
            total_steps: Total number of steps for the planning workflow
            use_llm_reward: Whether to use LLM for quality assessment
            llm_api_key: API key for LLM reward service
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

        # Initialize reward calculator
        self._reward_calculator = RewardCalculator(
            use_llm=use_llm_reward,
            api_key=llm_api_key,
        )

        # Task configuration (can be set via reset)
        self._task_keywords: list[str] = []
        self._task_difficulty: str = "medium"
        self._required_sections: list[str] = []

    def reset(
        self,
        task_description: str = "Create a comprehensive planning document for the given idea.",
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
        required_sections: list[str] | None = None,
    ) -> SkyPlanObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_description: The task description/goal
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level (easy, medium, hard)
            required_sections: Required sections for documents

        Returns:
            SkyPlanObservation with initial state, ready for the first agent to start
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = get_first_agent()
        self._step_number = 1
        self._documents = {}
        self._feedback = []
        self._last_action_result = None
        self._task_description = task_description
        self._done = False

        # Set task configuration
        self._task_keywords = task_keywords or []
        self._task_difficulty = task_difficulty
        self._required_sections = required_sections or []

        # Reset reward calculator
        self._reward_calculator.reset()

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

        # 3.5. Generate and process feedback
        self._generate_and_process_feedback(action)

        # 4. The Performance Review: Calculate reward using the new reward system
        # Track feedback values for reward calculation
        feedback_generated_this_step = self._feedback[-(self._last_action_result.feedback_generated_count or 0):] if self._last_action_result else []
        feedback_resolved_this_step = [fb for fb in self._feedback if fb.resolved and fb.resolution_timestamp and fb.resolution_timestamp.startswith(datetime.utcnow().isoformat()[:10])]
        new_approvals_this_step = [(doc_type, doc.author) for doc_type, doc in self._documents.items() if doc.status == "approved" and doc.updated_at and doc.updated_at.startswith(datetime.utcnow().isoformat()[:10])]

        step_reward = self._reward_calculator.calculate_step_reward(
            action=action,
            documents=self._documents,
            task_keywords=self._task_keywords,
            task_difficulty=self._task_difficulty,
            required_sections=self._required_sections,
            feedback_generated=feedback_generated_this_step,
            feedback_resolved=feedback_resolved_this_step,
            new_approvals=new_approvals_this_step,
        )
        reward = step_reward.total

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

        # 7.5. Process feedback resolutions after action completion
        self._process_feedback_resolutions(action.agent_id, action)

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

    def get_episode_reward(self) -> dict:
        """
        Get the final episode reward including completion bonus.

        This should be called after the episode is done.

        Returns:
            Dictionary with final_score, breakdown, and details
        """
        episode_reward = self._reward_calculator.calculate_episode_reward(
            documents=self._documents,
        )

        return {
            "final_score": episode_reward.final_score,
            "total_raw": episode_reward.total_raw,
            "completion_bonus": episode_reward.completion_bonus,
            "breakdown": episode_reward.breakdown,
            "step_count": len(episode_reward.step_rewards),
        }

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
        The Filing: Save the document to the Shared Folder with dynamic status updates.

        This method is called after every action to:
        1. Map action_type to document_type
        2. Determine the appropriate document status based on action type
        3. Create new document or update existing document with content and status
        4. Apply status transitions automatically based on DocumentStatusConfig

        The status transition is dynamic:
        - For content-creation actions (WRITE_PRD, DESIGN_ARCHITECTURE, etc.), status = "draft"
        - For status-changes actions (APPROVE_DOCUMENT, MARK_DOCUMENT_REVIEW, etc.),
          status = DocumentStatusConfig.STATUS_TRANSITIONS[action.action_type]
        - Status is extracted from the DocumentStatusConfig configuration

        Args:
            action: The action containing the document content and action_type

        Returns:
            None

        Examples:
            - action.action_type="WRITE_PRD" → doc_type="PRD", status="draft"
            - action.action_type="APPROVE_DOCUMENT" → doc_type="VALIDATION", status="approved"
            - action.action_type="FINAL_APPROVAL" → doc_type="STRATEGY", status="approved"
        """
        doc_type = ACTION_TO_DOCUMENT.get(action.action_type)
        if not doc_type:
            return  # Action doesn't produce a document

        try:
            timestamp = datetime.utcnow().isoformat() + "Z"

            # DYNAMIC STATUS DETERMINATION: Get status from configuration
            # This is the key integration: automatically map actions to document statuses
            action_status = DocumentStatusConfig.STATUS_TRANSITIONS.get(action.action_type)

            # DEBUG: Log for testing status transitions
            if action_status:
                print(f"[STATUS TRANSITION] {action.action_type} → {action_status}")

            # Apply status change to the document
            if doc_type in self._documents:
                # Update existing document
                if len(action.content.strip()) > 5:  # Has meaningful content
                    self._documents[doc_type].content = action.content
                self._documents[doc_type].author = action.agent_id
                self._documents[doc_type].updated_at = timestamp

                # Apply dynamic status transition if available
                if action_status:
                    self._documents[doc_type].status = action_status
                    print(f"[UPDATED] {doc_type} status → {action_status}")
            else:
                # Create new document - determine initial status
                # For new documents, default to DRAFT unless action specifies otherwise
                if action_status:
                    # Action explicitly sets status (e.g., MARK_DOCUMENT_REVIEW)
                    status = action_status
                else:
                    # Default content-creation actions to DRAFT
                    status = DocumentStatus.DRAFT

                self._documents[doc_type] = Document(
                    type=doc_type,
                    content=action.content,
                    author=action.agent_id,
                    created_at=timestamp,
                    updated_at=timestamp,
                    status=status,
                )
                print(f"[CREATED] {doc_type} with status: {status}")
                print(
                    f"[CONFIG] DocumentStatusConfig.STATUS_TRANSITIONS = {DocumentStatusConfig.STATUS_TRANSITIONS}"
                )
        except Exception as e:
            # Log error but don't fail the step
            print(f"Error filing document: {e}")

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
            status: Current workflow status
            reward: Reward value

        Returns:
            Complete SkyPlanObservation with document status summary
        """
        # Calculate document status summary
        status_counts = {"draft": 0, "in_review": 0, "approved": 0, "rejected": 0}
        awaiting_review = []

        for doc_type, doc in self._documents.items():
            # Count by status
            if doc.status in status_counts:
                status_counts[doc.status] += 1

            # Track documents awaiting review/approval
            if doc.status == "draft":
                awaiting_review.append(f"{doc_type} (needs review)")
            elif doc.status == "in_review":
                awaiting_review.append(f"{doc_type} (in review)")

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
            current_state={
                "status": status,
                "phase": self._get_current_phase(),
                "document_summary": status_counts,
            },
            errors=[],
            step_count=self._state.step_count,
            done=self._done,
            reward=reward,
            document_status_summary=status_counts,
            documents_awaiting_review=awaiting_review,
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

    # ============================================================================
    # Feedback Generation and Processing Methods
    # ============================================================================

    def _generate_collaborative_feedback(self, action: SkyPlanAction) -> Feedback | None:
        """Generate collaborative feedback based on document quality.

        Args:
            action: The action to evaluate

        Returns:
            Feedback object or None if no feedback needed
        """
        # Inline configuration since FeedbackConfig caused issues
        ENABLE_COLLABORATIVE_FEEDBACK = True
        COLLABORATIVE_FEEDBACK_THRESHOLD = 0.6

        if not ENABLE_COLLABORATIVE_FEEDBACK:
            return None

        doc_type = ACTION_TO_DOCUMENT.get(action.action_type)
        if not doc_type:
            return None

        # Calculate quality score
        quality_score = self._calculate_document_quality(action.content, doc_type)

        if quality_score < COLLABORATIVE_FEEDBACK_THRESHOLD:
            return Feedback.create(
                from_agent=action.agent_id,
                to_agent="",  # General feedback
                document_type=doc_type,
                feedback_type="suggestion",
                comment=f"Document quality score ({quality_score:.2f}) suggests improvements needed. Consider expanding content with specific examples and technical details."
            )
        return None

    def _generate_validation_feedback(self, agent_id: str) -> list[Feedback]:
        """Generate comprehensive validation feedback from Taylor.

        Args:
            agent_id: Agent being validated (should be "taylor")

        Returns:
            List of Feedback objects
        """
        # Inline configuration
        ENABLE_VALIDATOR_FEEDBACK = True
        TAYLOR_APPROVAL_THRESHOLD = 0.7

        if not ENABLE_VALIDATOR_FEEDBACK or agent_id != "taylor":
            return []

        feedback_list = []
        required_docs = get_required_documents(agent_id)

        for doc_type in required_docs:
            doc = self._documents.get(doc_type)
            if not doc or doc.author == "taylor":  # Don't self-review
                continue

            # Calculate quality score
            quality_score = self._calculate_document_quality(doc.content, doc_type)

            # Identify specific issues
            issues = self._identify_document_issues(doc, quality_score)

            if issues:
                # Choose feedback type based on severity
                if quality_score < 0.4:
                    feedback_type = "concern"
                elif quality_score < 0.6:
                    feedback_type = "critique"
                else:
                    feedback_type = "suggestion"

                feedback = Feedback.create(
                    from_agent="taylor",
                    to_agent=doc.author,
                    document_type=doc_type,
                    feedback_type=feedback_type,
                    comment=f"Validation feedback: {issues}"
                )
                feedback_list.append(feedback)

                # Update document status based on quality
                if quality_score >= TAYLOR_APPROVAL_THRESHOLD:
                    doc.status = "approved"
                elif quality_score >= 0.5:
                    doc.status = "in_review"
                else:
                    doc.status = "rejected"

        return feedback_list

    def _generate_strategic_feedback(self, agent_id: str) -> list[Feedback]:
        """Generate strategic feedback from CEO Sam.

        Args:
            agent_id: Should be "sam"

        Returns:
            List of Feedback objects
        """
        # Inline configuration
        ENABLE_STRATEGIC_FEEDBACK = True

        if not ENABLE_STRATEGIC_FEEDBACK or agent_id != "sam":
            return []

        feedback_list = []
        required_docs = get_required_documents(agent_id)

        # Check for missing documents
        missing_docs = [d for d in required_docs if d not in self._documents]
        if missing_docs:
            feedback_list.append(Feedback.create(
                from_agent="sam",
                to_agent="",
                document_type="STRATEGY",
                feedback_type="concern",
                comment=f"Missing required documents: {', '.join(missing_docs)}"
            ))

        # Check document consistency
        consistency_score = self._check_document_consistency()
        if consistency_score < 0.7:
            feedback_list.append(Feedback.create(
                from_agent="sam",
                to_agent="taylor",
                document_type="STRATEGY",
                feedback_type="critique",
                comment=f"Document consistency low ({consistency_score:.2f}). Taylor, please review for alignment and identify discrepancies."
            ))

        # Check approval status
        approved_count = sum(1 for doc in self._documents.values() if doc.status == "approved")
        if approved_count < len(required_docs):
            feedback_list.append(Feedback.create(
                from_agent="sam",
                to_agent="",
                document_type="STRATEGY",
                feedback_type="request_revision",
                comment=f"Only {approved_count}/{len(required_docs)} documents approved. All documents must reach 'approved' status before final strategic approval."
            ))

        return feedback_list

    def _identify_document_issues(self, doc: Document, quality_score: float) -> str:
        """Identify specific issues in a document for feedback."""
        issues = []

        if len(doc.content) < 200:
            issues.append("content too short")
        if "\n\n" not in doc.content:  # No paragraphs
            issues.append("lack of structured paragraphs")
        if "#" not in doc.content:  # No headers
            issues.append("missing section headers")

        return "; ".join(issues)

    def _check_document_consistency(self) -> float:
        """Check consistency across all planning documents."""
        # Simple consistency check - could be enhanced
        return 0.8  # Placeholder for now

    def _process_feedback_resolutions(self, agent_id: str, action: SkyPlanAction) -> None:
        """Check if current action addresses previous feedback and mark as resolved."""
        # Inline configuration
        ENABLE_FEEDBACK_RESOLUTION = True

        if not ENABLE_FEEDBACK_RESOLUTION:
            return

        # Get unresolved feedback targeting this agent
        unresolved_feedback = [
            fb for fb in self._feedback
            if not fb.resolved and (fb.to_agent == agent_id or fb.to_agent == "")
        ]

        resolved_count = 0
        primary_feedback_addressed = False

        for feedback in unresolved_feedback:
            if self._is_feedback_addressed(feedback, action):
                feedback.resolved = True
                feedback.resolution_timestamp = datetime.utcnow().isoformat() + "Z"
                feedback.addressed_by = agent_id
                resolved_count += 1

                if feedback.from_agent in ["taylor", "sam"]:
                    primary_feedback_addressed = True

        # Update last action result
        if self._last_action_result:
            self._last_action_result.resolved_feedback_count = resolved_count
            self._last_action_result.primary_feedback_addressed = primary_feedback_addressed

    def _is_feedback_addressed(self, feedback: Feedback, action: SkyPlanAction) -> bool:
        """Determine if an action addresses a specific feedback item."""
        # Check content includes relevant keywords
        feedback_keywords = set(feedback.comment.lower().split())
        action_keywords = set(action.content.lower().split()) | set(action.reasoning.lower().split())

        keyword_overlap = len(feedback_keywords.intersection(action_keywords))
        total_keywords = len(feedback_keywords)

        # Consider addressed if >30% keywords overlap or action mentions feedback
        return (keyword_overlap / max(total_keywords, 1) > 0.3 or
                "feedback" in action.reasoning.lower() or
                "addressing" in action.reasoning.lower())

    def _calculate_document_quality(self, content: str, doc_type: str) -> float:
        """Calculate document quality score (0-1)."""
        # Simple quality calculation - could be enhanced
        # Check length, structure, keywords
        min_length = 50
        if len(content) < min_length:
            return 0.3
        if len(content) < min_length * 2:
            return 0.6
        if "\n" in content and "#" in content:
            return 0.8
        return 1.0

    def _generate_and_process_feedback(self, action: SkyPlanAction) -> None:
        """Centralized feedback generation for an action."""
        # Generate collaborative feedback
        collaborative_fb = self._generate_collaborative_feedback(action)
        if collaborative_fb:
            self._feedback.append(collaborative_fb)
            if self._last_action_result:
                self._last_action_result.feedback_generated_count = 1

        # Generate validator feedback (Taylor-specific)
        if action.agent_id == "taylor":
            validation_feedback = self._generate_validation_feedback(action.agent_id)
            self._feedback.extend(validation_feedback)
            if self._last_action_result:
                self._last_action_result.feedback_generated_count = len(validation_feedback)

        # Generate strategic feedback (Sam-specific)
        if action.agent_id == "sam":
            strategic_feedback = self._generate_strategic_feedback(action.agent_id)
            self._feedback.extend(strategic_feedback)
            if self._last_action_result:
                self._last_action_result.feedback_generated_count = len(strategic_feedback)
