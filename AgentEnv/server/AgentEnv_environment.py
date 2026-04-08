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

import re
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..content_utils import (
        count_paragraph_blocks,
        has_markdown_headers,
        has_markdown_lists,
        keyword_coverage_ratio,
    )
    from ..models import (
        ACTION_TO_DOCUMENT,
        Document,
        DocumentStatus,
        DocumentStatusConfig,
        DocumentType,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
        utc_timestamp,
    )
    from ..reward import RewardCalculator
    from ..workflow import (
        get_all_agent_ids,
        get_first_agent,
        get_handoff_message,
        get_produced_documents,
        get_required_documents,
    )
except ImportError:
    from content_utils import (
        count_paragraph_blocks,
        has_markdown_headers,
        has_markdown_lists,
        keyword_coverage_ratio,
    )
    from models import (
        ACTION_TO_DOCUMENT,
        Document,
        DocumentStatus,
        DocumentStatusConfig,
        DocumentType,
        Feedback,
        LastAction,
        SkyPlanAction,
        SkyPlanObservation,
        ValidationConfig,
        WorkflowConfig,
        utc_timestamp,
    )
    from reward import RewardCalculator
    from workflow import (
        get_all_agent_ids,
        get_first_agent,
        get_handoff_message,
        get_produced_documents,
        get_required_documents,
    )


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "they",
    "them",
    "must",
    "need",
    "more",
    "also",
    "have",
    "has",
    "had",
    "can",
    "will",
    "should",
    "would",
    "could",
    "please",
    "document",
    "documents",
    "feedback",
    "validation",
    "strategic",
    "review",
    "revision",
    "details",
    "detail",
    "team",
}
ROLE_AUTHORING_AGENTS = {"maya", "elon", "jordan", "robert"}
VALID_AGENT_IDS = ROLE_AUTHORING_AGENTS | {"taylor", "sam"}
BLOCKING_FEEDBACK_TYPES = {"concern", "request_revision"}
COMPANION_DOCUMENTS = {
    "jordan": ("TRD", "ARCHITECTURE"),
    "robert": ("ROADMAP", "TASKS"),
}


class SkyPlanEnvironment(Environment):
    """
    Multi-agent planning environment for SkyPlan.

    Six specialized agents collaborate to produce planning documents:
    - Maya (Research Analyst) -> Research Summary
    - Elon (Product Manager) -> PRD, feature list
    - Jordan (Architect) -> TRD, system architecture
    - Robert (Execution Planner) -> Roadmap, sprint backlog
    - Taylor (Validator) -> Validation report
    - Sam (CEO) -> Final strategic approval

    Workflow: Maya -> Elon -> Jordan -> Robert -> Taylor -> Sam
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        total_steps: int = WorkflowConfig.MAX_TOTAL_STEPS,
        use_llm_reward: bool = True,
        llm_api_key: str | None = None,
    ):
        """Initialize the SkyPlan environment with isolated state."""

        self._workflow_order = get_all_agent_ids()
        self._pending_agents = self._workflow_order.copy()
        self._max_total_steps = max(total_steps, WorkflowConfig.DEFAULT_TOTAL_STEPS)
        self._revision_cycles_used = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_agent = self._pending_agents[0] if self._pending_agents else get_first_agent()
        self._step_number = 1
        self._documents: dict[str, Document] = {}
        self._feedback: list[Feedback] = []
        self._last_action_result: LastAction | None = None
        self._task_description = ""
        self._task_id: str | None = None
        self._done = False

        self._task_keywords: list[str] = []
        self._task_difficulty: str = "medium"
        self._required_sections: list[str] = []

        self._feedback_generated_this_step: list[Feedback] = []
        self._feedback_resolved_this_step: list[Feedback] = []
        self._new_approvals_this_step: list[tuple[str, str]] = []
        self._primary_feedback_addressed_this_step = False

        self._reward_calculator = RewardCalculator(
            use_llm=use_llm_reward,
            api_key=llm_api_key,
        )

    def reset(
        self,
        task_description: str = "Create a comprehensive planning document for the given idea.",
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
        required_sections: list[str] | None = None,
        task_id: str | None = None,
    ) -> SkyPlanObservation:
        """Reset the environment for a new episode."""

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._pending_agents = self._workflow_order.copy()
        self._revision_cycles_used = 0
        self._current_agent = self._pending_agents[0] if self._pending_agents else get_first_agent()
        self._step_number = 1
        self._documents = {}
        self._feedback = []
        self._last_action_result = None
        self._task_description = task_description
        self._task_id = task_id
        self._done = False

        self._task_keywords = task_keywords or []
        self._task_difficulty = task_difficulty
        self._required_sections = required_sections or []
        self._reset_step_tracking()

        self._reward_calculator.reset()

        return self._build_observation(
            result="Environment reset. Ready to begin planning.",
            reasoning="Starting new planning episode.",
            status="ready",
        )

    def step(self, action: SkyPlanAction) -> SkyPlanObservation:  # type: ignore[override]
        """Execute a step in the environment."""

        self._state.step_count += 1
        self._reset_step_tracking()

        if not self._is_valid_agent(action):
            self._last_action_result = LastAction.create(
                agent_id=action.agent_id,
                action_type=action.action_type,
                result="failure",
                message=f"Expected agent {self._current_agent}, got {action.agent_id}",
            )
            return self._create_error_observation(
                f"Expected agent {self._current_agent}, got {action.agent_id}"
            )

        validation_result = self._validate_action(action)
        if not validation_result["is_valid"]:
            self._last_action_result = LastAction.create(
                agent_id=action.agent_id,
                action_type=action.action_type,
                result="failure",
                message=validation_result["error"],
            )
            return self._create_error_observation(validation_result["error"])

        self._file_document(action)
        self._process_feedback_resolutions(action.agent_id, action)
        self._generate_and_process_feedback(action)

        reward = self._reward_calculator.calculate_step_reward(
            action=action,
            documents=self._documents,
            task_keywords=self._task_keywords,
            task_difficulty=self._task_difficulty,
            required_sections=self._required_sections,
            feedback_generated=self._feedback_generated_this_step,
            feedback_resolved=self._feedback_resolved_this_step,
            new_approvals=self._new_approvals_this_step,
        ).total

        workflow_status, workflow_note = self._advance_workflow(action.agent_id)

        message = f"{action.action_type} completed successfully"
        if self._feedback_generated_this_step or self._feedback_resolved_this_step:
            message += (
                f" ({len(self._feedback_generated_this_step)} feedback generated, "
                f"{len(self._feedback_resolved_this_step)} resolved)"
            )
        if workflow_status == "revision_requested":
            message += " Revision loop scheduled."

        self._last_action_result = LastAction.create(
            agent_id=action.agent_id,
            action_type=action.action_type,
            result="success",
            message=message,
        )
        self._last_action_result.feedback_generated_count = len(self._feedback_generated_this_step)
        self._last_action_result.resolved_feedback_count = len(self._feedback_resolved_this_step)
        self._last_action_result.primary_feedback_addressed = (
            self._primary_feedback_addressed_this_step
        )

        reasoning_msg = f"Action processed by {action.agent_id}. {workflow_note}"

        return self._build_observation(
            result=f"{action.action_type} completed successfully",
            reasoning=reasoning_msg,
            status=workflow_status,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""

        return self._state

    def get_episode_reward(self) -> dict:
        """Get the final episode reward including completion bonus."""

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

    def _advance_workflow(self, completed_agent: str) -> tuple[str, str]:
        """Advance the dynamic workflow queue and schedule revision loops when needed."""

        if self._pending_agents and self._pending_agents[0] == completed_agent:
            self._pending_agents.pop(0)

        revision_queue = self._build_revision_queue(completed_agent)
        if revision_queue:
            if self._revision_cycles_used >= WorkflowConfig.MAX_REVISION_CYCLES:
                self._pending_agents = []
                self._done = True
                self._step_number = self._state.step_count
                return (
                    "blocked",
                    "Revision loop requested, but the configured revision budget is exhausted.",
                )

            self._revision_cycles_used += 1
            self._pending_agents = revision_queue

        projected_total_steps = self._state.step_count + len(self._pending_agents)
        if projected_total_steps > self._max_total_steps:
            self._pending_agents = []
            self._done = True
            self._step_number = self._state.step_count
            return (
                "blocked",
                "Workflow terminated because the episode exceeded the configured step budget.",
            )

        if self._pending_agents:
            self._done = False
            self._current_agent = self._pending_agents[0]
            self._step_number = self._state.step_count + 1
            if revision_queue:
                queue_label = " -> ".join(revision_queue)
                return (
                    "revision_requested",
                    f"Revision loop scheduled through {queue_label}.",
                )
            return (
                "in_progress",
                get_handoff_message(completed_agent),
            )

        self._done = True
        self._step_number = self._state.step_count
        return (
            "completed",
            "Workflow complete. All required planning steps have finished.",
        )

    def _build_revision_queue(self, completed_agent: str) -> list[str]:
        """Build the next workflow slice when review or strategy requires revision."""

        if completed_agent not in {"taylor", "sam"}:
            return []

        target_agents = self._get_nonapproved_agent_targets(completed_agent)
        target_agents.update(self._get_revision_targets_from_feedback(self._feedback_generated_this_step))

        if completed_agent == "sam":
            target_agents.discard("sam")
            if not target_agents:
                strategy_doc = self._documents.get(DocumentType.STRATEGY)
                if strategy_doc and strategy_doc.status == DocumentStatus.REJECTED:
                    target_agents.add("taylor")

        return self._workflow_slice_from(target_agents)

    def _get_nonapproved_agent_targets(self, completed_agent: str) -> set[str]:
        """Identify agent owners for documents that still need revision or approval."""

        if completed_agent == "taylor":
            relevant_docs = get_required_documents("taylor")
        else:
            relevant_docs = (
                DocumentStatusConfig.get_required_approvals(self._task_id)
                or get_required_documents("sam")
            )
            relevant_docs = list(relevant_docs) + [DocumentType.VALIDATION]

        targets: set[str] = set()
        for doc_type in relevant_docs:
            document = self._documents.get(doc_type)
            if document is None:
                owner = self._get_document_owner(doc_type)
                if owner:
                    targets.add(owner)
                continue

            if document.status != DocumentStatus.APPROVED:
                owner = document.author or self._get_document_owner(doc_type)
                if owner in VALID_AGENT_IDS:
                    targets.add(owner)

        return targets

    def _get_revision_targets_from_feedback(self, feedback_items: list[Feedback]) -> set[str]:
        """Derive workflow targets from actionable feedback generated in the current step."""

        targets: set[str] = set()
        for feedback in feedback_items:
            if feedback.feedback_type == "approval":
                continue

            if feedback.to_agent in VALID_AGENT_IDS:
                targets.add(feedback.to_agent)
                continue

            if feedback.document_type:
                owner = self._get_document_owner(feedback.document_type)
                if owner in VALID_AGENT_IDS:
                    targets.add(owner)

        return targets

    def _get_document_owner(self, doc_type: str) -> str:
        """Return the workflow agent responsible for authoring a document type."""

        for agent_id in self._workflow_order:
            if doc_type in get_produced_documents(agent_id):
                return agent_id
        return ""

    def _workflow_slice_from(self, target_agents: set[str]) -> list[str]:
        """Return the ordered workflow slice starting at the earliest target agent."""

        ordered_targets = [agent for agent in self._workflow_order if agent in target_agents]
        if not ordered_targets:
            return []

        first_target = ordered_targets[0]
        start_index = self._workflow_order.index(first_target)
        return self._workflow_order[start_index:].copy()

    def _get_projected_total_steps(self) -> int:
        """Estimate the total episode length from completed plus pending work."""

        if self._done:
            return max(self._state.step_count, WorkflowConfig.DEFAULT_TOTAL_STEPS)

        return max(
            self._state.step_count + len(self._pending_agents),
            WorkflowConfig.DEFAULT_TOTAL_STEPS,
        )

    def _reset_step_tracking(self) -> None:
        """Reset per-step workflow metrics used by rewards and observations."""

        self._feedback_generated_this_step = []
        self._feedback_resolved_this_step = []
        self._new_approvals_this_step = []
        self._primary_feedback_addressed_this_step = False

    def _is_valid_agent(self, action: SkyPlanAction) -> bool:
        """Check if the action is from the expected agent."""

        return action.agent_id == self._current_agent

    def _validate_action(self, action: SkyPlanAction) -> dict:
        """Validate action quality before it enters the workflow."""

        if not action.content or len(action.content.strip()) < ValidationConfig.MIN_CONTENT_LENGTH:
            return {
                "is_valid": False,
                "error": f"Content too short or empty for action {action.action_type}",
            }

        if not action.reasoning or len(action.reasoning.strip()) < ValidationConfig.MIN_REASONING_LENGTH:
            return {
                "is_valid": False,
                "error": "Reasoning too short or empty",
            }

        return {"is_valid": True, "error": ""}

    def _file_document(self, action: SkyPlanAction) -> None:
        """
        Save the current action output and apply the first workflow status.

        Authoring agents always create or refresh drafts. Taylor and Sam create
        review artifacts that begin in review until their validation or strategy
        pass decides whether the project is approved or sent back for revision.
        """

        doc_type = ACTION_TO_DOCUMENT.get(action.action_type)
        if not doc_type:
            return

        timestamp = utc_timestamp()
        action_status = DocumentStatusConfig.STATUS_TRANSITIONS.get(action.action_type)
        document = self._documents.get(doc_type)

        if document is None:
            document = Document(
                type=doc_type,
                content=action.content,
                author=action.agent_id,
                created_at=timestamp,
                updated_at=timestamp,
                status=DocumentStatus.DRAFT,
            )
            self._documents[doc_type] = document
        else:
            if action.content.strip():
                document.content = action.content
            document.author = action.agent_id
            document.updated_at = timestamp

        if action_status:
            self._set_document_status(
                doc_type,
                action_status,
                timestamp=timestamp,
                approved_by=action.agent_id
                if action_status == DocumentStatus.APPROVED.value
                else None,
            )
        elif action.agent_id in ROLE_AUTHORING_AGENTS:
            self._set_document_status(doc_type, DocumentStatus.DRAFT, timestamp=timestamp)
        elif action.agent_id == "taylor" and doc_type == DocumentType.VALIDATION:
            self._set_document_status(doc_type, DocumentStatus.IN_REVIEW, timestamp=timestamp)
        elif action.agent_id == "sam" and doc_type == DocumentType.STRATEGY:
            initial_status = (
                DocumentStatus.REJECTED
                if action.action_type == "REQUEST_REVISION"
                else DocumentStatus.IN_REVIEW
            )
            self._set_document_status(doc_type, initial_status, timestamp=timestamp)

        self._sync_companion_documents(
            action=action,
            primary_doc_type=doc_type,
            timestamp=timestamp,
        )

        if action.agent_id in ROLE_AUTHORING_AGENTS:
            affected_doc_types = {doc_type, *COMPANION_DOCUMENTS.get(action.agent_id, ())}
            self._invalidate_downstream_documents(
                author_agent=action.agent_id,
                timestamp=timestamp,
                excluded_doc_types=affected_doc_types,
            )

    def _set_document_status(
        self,
        doc_type: str,
        status: DocumentStatus | str,
        *,
        timestamp: str | None = None,
        approved_by: str | None = None,
    ) -> None:
        """Apply a document status transition and record new approvals."""

        document = self._documents.get(doc_type)
        if document is None:
            return

        status_value = status.value if isinstance(status, DocumentStatus) else status
        previous_status = document.status
        document.status = status_value
        document.updated_at = timestamp or utc_timestamp()

        if (
            previous_status != DocumentStatus.APPROVED
            and status_value == DocumentStatus.APPROVED
            and approved_by
        ):
            approval = (doc_type, approved_by)
            if approval not in self._new_approvals_this_step:
                self._new_approvals_this_step.append(approval)

    def _sync_companion_documents(
        self,
        action: SkyPlanAction,
        primary_doc_type: str,
        timestamp: str,
    ) -> None:
        """Keep paired architecture/planning documents in sync for single-turn agents."""

        companion_types = COMPANION_DOCUMENTS.get(action.agent_id)
        if not companion_types:
            return

        for companion_doc_type in companion_types:
            if companion_doc_type == primary_doc_type:
                continue

            companion_document = self._documents.get(companion_doc_type)
            companion_content = (
                f"# {companion_doc_type}\n"
                f"Derived from {primary_doc_type} during {action.action_type}.\n\n"
                f"{action.content}"
            )

            if companion_document is None:
                self._documents[companion_doc_type] = Document(
                    type=companion_doc_type,
                    content=companion_content,
                    author=action.agent_id,
                    created_at=timestamp,
                    updated_at=timestamp,
                    status=DocumentStatus.DRAFT,
                )
            else:
                companion_document.content = companion_content
                companion_document.author = action.agent_id
                companion_document.updated_at = timestamp
                companion_document.status = DocumentStatus.DRAFT

    def _invalidate_downstream_documents(
        self,
        author_agent: str,
        timestamp: str,
        excluded_doc_types: set[str],
    ) -> None:
        """Mark downstream artifacts as drafts after an upstream document changes."""

        if author_agent not in self._workflow_order:
            return

        start_index = self._workflow_order.index(author_agent) + 1
        for downstream_agent in self._workflow_order[start_index:]:
            for downstream_doc_type in get_produced_documents(downstream_agent):
                if downstream_doc_type in excluded_doc_types:
                    continue

                downstream_doc = self._documents.get(downstream_doc_type)
                if downstream_doc is None:
                    continue

                downstream_doc.status = DocumentStatus.DRAFT
                downstream_doc.updated_at = timestamp

    def _build_observation(
        self,
        result: str,
        reasoning: str,
        status: str,
        reward: float = 0.0,
    ) -> SkyPlanObservation:
        """Build a SkyPlanObservation with current state."""

        status_counts = {"draft": 0, "in_review": 0, "approved": 0, "rejected": 0}
        awaiting_review: list[str] = []

        for doc_type, doc in self._documents.items():
            if doc.status in status_counts:
                status_counts[doc.status] += 1
            if doc.status == DocumentStatus.DRAFT:
                awaiting_review.append(f"{doc_type} (needs review)")
            elif doc.status == DocumentStatus.IN_REVIEW:
                awaiting_review.append(f"{doc_type} (in review)")

        return SkyPlanObservation(
            task_description=self._task_description,
            result=result,
            reasoning=reasoning,
            current_agent=self._current_agent,
            step_number=self._step_number,
            total_steps=self._get_projected_total_steps(),
            documents=self._documents,
            feedback=self._feedback,
            last_action_result=self._last_action_result,
            current_state={
                "status": status,
                "phase": self._get_current_phase(),
                "task_id": self._task_id or "",
                "document_summary": status_counts,
                "unresolved_feedback_count": len(self.get_unresolved_feedback()),
            },
            document_status_summary=status_counts,
            documents_awaiting_review=awaiting_review,
            errors=[],
            step_count=self._state.step_count,
            done=self._done,
            reward=reward,
        )

    def _create_error_observation(self, error_message: str) -> SkyPlanObservation:
        """Create an observation with an error state."""

        observation = self._build_observation(
            result="Action failed",
            reasoning=error_message,
            status="error",
            reward=0.0,
        )
        observation.errors = [error_message]
        return observation

    def _get_current_phase(self) -> str:
        """Get the current planning phase from the agent currently holding the turn."""

        if self._current_agent in WorkflowConfig.AGENT_PHASES:
            return WorkflowConfig.AGENT_PHASES[self._current_agent]

        if self._last_action_result and self._last_action_result.agent_id in WorkflowConfig.AGENT_PHASES:
            return WorkflowConfig.AGENT_PHASES[self._last_action_result.agent_id]

        return WorkflowConfig.PHASES[0]

    def _generate_collaborative_feedback(self, action: SkyPlanAction) -> Feedback | None:
        """Generate collaborative feedback when a new document is weak."""

        doc_type = ACTION_TO_DOCUMENT.get(action.action_type)
        if not doc_type:
            return None

        quality_score = self._calculate_document_quality(action.content, doc_type)
        if quality_score >= 0.6:
            return None

        return Feedback.create(
            from_agent=action.agent_id,
            to_agent="",
            document_type=doc_type,
            feedback_type="suggestion",
            comment=(
                f"Document quality score ({quality_score:.2f}) suggests improvements needed. "
                "Consider expanding content with clearer structure, task-specific details, and "
                "handoff-ready guidance for downstream agents."
            ),
        )

    def _generate_validation_feedback(
        self,
        agent_id: str,
        action: SkyPlanAction | None = None,
    ) -> list[Feedback]:
        """Generate validation feedback from Taylor and transition document statuses."""

        if agent_id != "taylor":
            return []

        feedback_list: list[Feedback] = []
        blocking_findings = False

        for doc_type in get_required_documents(agent_id):
            document = self._documents.get(doc_type)
            if document is None:
                feedback_list.append(
                    Feedback.create(
                        from_agent="taylor",
                        to_agent="",
                        document_type=doc_type,
                        feedback_type="concern",
                        comment=f"Missing required {doc_type} document for validation.",
                    )
                )
                blocking_findings = True
                continue

            if document.author == "taylor":
                continue

            self._set_document_status(doc_type, DocumentStatus.IN_REVIEW)
            quality_score = self._calculate_document_quality(document.content, doc_type)
            issues = self._identify_document_issues(document, quality_score)

            if issues:
                if quality_score < 0.45:
                    feedback_type = "concern"
                    next_status = DocumentStatus.REJECTED
                    blocking_findings = True
                elif quality_score < 0.65:
                    feedback_type = "critique"
                    next_status = DocumentStatus.IN_REVIEW
                else:
                    feedback_type = "suggestion"
                    next_status = DocumentStatus.IN_REVIEW

                feedback_list.append(
                    Feedback.create(
                        from_agent="taylor",
                        to_agent=document.author,
                        document_type=doc_type,
                        feedback_type=feedback_type,
                        comment=f"Validation feedback: {issues}",
                    )
                )
                self._set_document_status(doc_type, next_status)
                continue

            feedback_list.append(
                Feedback.create(
                    from_agent="taylor",
                    to_agent=document.author,
                    document_type=doc_type,
                    feedback_type="approval",
                    comment=f"{doc_type} meets the current validation bar and is approved for strategy review.",
                )
            )
            self._set_document_status(
                doc_type,
                DocumentStatus.APPROVED,
                approved_by="taylor",
            )

        if DocumentType.VALIDATION in self._documents:
            validation_status = (
                DocumentStatus.IN_REVIEW if blocking_findings else DocumentStatus.APPROVED
            )
            self._set_document_status(
                DocumentType.VALIDATION,
                validation_status,
                approved_by="taylor" if validation_status == DocumentStatus.APPROVED else None,
            )

        return feedback_list

    def _generate_strategic_feedback(
        self,
        agent_id: str,
        action: SkyPlanAction | None = None,
    ) -> list[Feedback]:
        """Generate strategic feedback from CEO Sam and finalize approval state."""

        if agent_id != "sam":
            return []

        feedback_list: list[Feedback] = []
        required_docs = get_required_documents(agent_id)

        missing_docs = [doc_type for doc_type in required_docs if doc_type not in self._documents]
        if missing_docs:
            feedback_list.append(
                Feedback.create(
                    from_agent="sam",
                    to_agent="",
                    document_type="STRATEGY",
                    feedback_type="concern",
                    comment=f"Missing required documents: {', '.join(missing_docs)}",
                )
            )

        consistency_score = self._check_document_consistency()
        if consistency_score < 0.7:
            feedback_list.append(
                Feedback.create(
                    from_agent="sam",
                    to_agent="taylor",
                    document_type="STRATEGY",
                    feedback_type="critique",
                    comment=(
                        f"Document consistency low ({consistency_score:.2f}). "
                        "Taylor, please reconcile cross-document assumptions before approval."
                    ),
                )
            )

        required_approvals = (
            DocumentStatusConfig.get_required_approvals(self._task_id) or required_docs
        )
        required_approvals = [
            doc_type for doc_type in required_approvals if doc_type != DocumentType.STRATEGY
        ]
        unapproved_docs = [
            doc_type
            for doc_type in required_approvals
            if doc_type not in self._documents
            or self._documents[doc_type].status != DocumentStatus.APPROVED
        ]
        if unapproved_docs or (action and action.action_type == "REQUEST_REVISION"):
            feedback_list.append(
                Feedback.create(
                    from_agent="sam",
                    to_agent="",
                    document_type="STRATEGY",
                    feedback_type="request_revision",
                    comment=(
                        "Final strategic approval is blocked until these documents are approved: "
                        f"{', '.join(unapproved_docs or required_approvals)}"
                    ),
                )
            )

        if feedback_list:
            strategy_status = (
                DocumentStatus.REJECTED
                if any(item.feedback_type in BLOCKING_FEEDBACK_TYPES for item in feedback_list)
                else DocumentStatus.IN_REVIEW
            )
            if DocumentType.STRATEGY in self._documents:
                self._set_document_status(DocumentType.STRATEGY, strategy_status)
            return feedback_list

        feedback_list.append(
            Feedback.create(
                from_agent="sam",
                to_agent="",
                document_type="STRATEGY",
                feedback_type="approval",
                comment="Strategy approved. The planning package is aligned, complete, and ready for execution.",
            )
        )

        for doc_type in set(required_docs + [DocumentType.VALIDATION, DocumentType.STRATEGY]):
            if doc_type in self._documents:
                self._set_document_status(
                    doc_type,
                    DocumentStatus.APPROVED,
                    approved_by="sam",
                )

        return feedback_list

    def _identify_document_issues(self, doc: Document, quality_score: float) -> str:
        """Identify specific issues in a document for targeted feedback."""

        issues: list[str] = []
        content_lower = doc.content.lower()
        content_length = len(doc.content.strip())

        if content_length < 200:
            issues.append("content too short")
        if not has_markdown_headers(doc.content):
            issues.append("missing section headers")
        if count_paragraph_blocks(doc.content) < 2:
            issues.append("lack of structured paragraphs")
        if not has_markdown_lists(doc.content):
            issues.append("missing actionable lists")
        if self._task_keywords and keyword_coverage_ratio(doc.content, self._task_keywords) == 0.0:
            issues.append("missing task-specific requirements")
        if self._required_sections and not any(
            section.lower() in content_lower for section in self._required_sections
        ):
            issues.append("missing required task sections")
        if quality_score < 0.45 and "risk" not in content_lower and doc.type in {
            DocumentType.TRD,
            DocumentType.ARCHITECTURE,
            DocumentType.ROADMAP,
            DocumentType.TASKS,
        }:
            issues.append("insufficient risk or dependency coverage")

        return "; ".join(issues)

    def _check_document_consistency(self) -> float:
        """Check consistency across all planning documents using deterministic heuristics."""

        documents = [doc for doc in self._documents.values() if doc.content.strip()]
        if not documents:
            return 0.0
        if len(documents) == 1:
            return 0.6

        keyword_score = 1.0
        if self._task_keywords:
            keyword_hits = sum(
                1
                for doc in documents
                if any(keyword.lower() in doc.content.lower() for keyword in self._task_keywords)
            )
            keyword_score = keyword_hits / len(documents)

        structure_score = sum(has_markdown_headers(doc.content) for doc in documents) / len(documents)
        status_score = sum(doc.status != DocumentStatus.REJECTED for doc in documents) / len(documents)

        keyword_sets = [self._extract_keywords(doc.content) for doc in documents]
        overlap_scores: list[float] = []
        for left, right in zip(keyword_sets, keyword_sets[1:]):
            if not left or not right:
                overlap_scores.append(0.0)
                continue
            overlap_scores.append(len(left & right) / len(left | right))
        overlap_score = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

        consistency = (
            0.35 * keyword_score
            + 0.20 * structure_score
            + 0.20 * status_score
            + 0.25 * overlap_score
        )
        return max(0.0, min(consistency, 1.0))

    def _process_feedback_resolutions(self, agent_id: str, action: SkyPlanAction) -> None:
        """Mark targeted prior feedback as resolved when the action addresses it."""

        unresolved_feedback = [
            feedback
            for feedback in self._feedback
            if not feedback.resolved and feedback.to_agent in {"", agent_id}
        ]

        for feedback in unresolved_feedback:
            if not self._is_feedback_addressed(feedback, action):
                continue

            feedback.resolved = True
            feedback.resolution_timestamp = utc_timestamp()
            feedback.addressed_by = agent_id
            self._feedback_resolved_this_step.append(feedback)

            if feedback.from_agent in {"taylor", "sam"}:
                self._primary_feedback_addressed_this_step = True

    def _is_feedback_addressed(self, feedback: Feedback, action: SkyPlanAction) -> bool:
        """Determine if an action substantively addresses a specific feedback item."""

        action_doc_type = ACTION_TO_DOCUMENT.get(action.action_type, "")
        if feedback.document_type and action_doc_type and feedback.document_type != action_doc_type:
            return False

        action_text = f"{action.reasoning}\n{action.content}".lower()
        feedback_text = feedback.comment.lower()
        action_keywords = self._extract_keywords(action_text)
        feedback_keywords = self._extract_keywords(feedback_text)
        keyword_overlap = len(action_keywords & feedback_keywords)
        keyword_ratio = keyword_overlap / max(len(feedback_keywords), 1)

        acknowledged = any(
            phrase in action.reasoning.lower()
            for phrase in (
                "addressing feedback",
                "addressing taylor",
                "addressing sam",
                "review comments",
                "requested revision",
                "feedback",
                "revision",
            )
        )
        structural_fix = (
            ("header" in feedback_text and "#" in action.content)
            or ("paragraph" in feedback_text and "\n\n" in action.content)
            or ("list" in feedback_text and any(marker in action.content for marker in ("- ", "* ", "1. ")))
            or ("short" in feedback_text and len(action.content.strip()) >= 200)
            or ("expand" in feedback_text and len(action.content.strip()) >= 200)
            or ("detail" in feedback_text and len(action.content.strip()) >= 200)
        )

        return acknowledged or structural_fix or keyword_ratio >= 0.35

    def _calculate_document_quality(self, content: str, doc_type: str) -> float:
        """Calculate a deterministic quality score in the range [0.0, 1.0]."""

        difficulty_targets = {
            "easy": 180,
            "medium": 320,
            "hard": 480,
        }
        target_length = difficulty_targets.get(self._task_difficulty, 320)
        content_lower = content.lower()

        length_score = min(len(content.strip()) / target_length, 1.0)
        header_score = 1.0 if has_markdown_headers(content) else 0.0
        paragraph_score = 1.0 if count_paragraph_blocks(content) >= 2 else 0.0
        list_score = 1.0 if has_markdown_lists(content) else 0.0

        keyword_score = 1.0
        if self._task_keywords:
            keyword_score = keyword_coverage_ratio(content, self._task_keywords)

        doc_hint_bonus = 1.0 if doc_type.lower() in content_lower else 0.0

        quality = (
            0.35 * length_score
            + 0.20 * header_score
            + 0.15 * paragraph_score
            + 0.15 * list_score
            + 0.10 * keyword_score
            + 0.05 * doc_hint_bonus
        )
        return max(0.0, min(quality, 1.0))

    def _generate_and_process_feedback(self, action: SkyPlanAction) -> None:
        """Generate all feedback for an action and persist it once."""

        generated_feedback: list[Feedback] = []

        collaborative_feedback = self._generate_collaborative_feedback(action)
        if collaborative_feedback:
            generated_feedback.append(collaborative_feedback)

        if action.agent_id == "taylor":
            generated_feedback.extend(self._generate_validation_feedback(action.agent_id, action))

        if action.agent_id == "sam":
            generated_feedback.extend(self._generate_strategic_feedback(action.agent_id, action))

        self._feedback.extend(generated_feedback)
        self._feedback_generated_this_step = generated_feedback

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract normalized keywords for lightweight consistency matching."""

        return {
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in TOKEN_STOPWORDS
        }

    def get_unresolved_feedback(self) -> list[Feedback]:
        """Expose unresolved feedback for observation summaries and tests."""

        return [feedback for feedback in self._feedback if not feedback.resolved]
