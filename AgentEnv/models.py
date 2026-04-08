# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SkyPlan Environment.

The SkyPlan environment is a multi-agent planning system where specialized agents
collaborate to transform an idea into structured planning documents.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Configuration Constants
# ============================================================================


class ValidationConfig:
    """Configuration for action validation thresholds.

    Attributes:
        MIN_CONTENT_LENGTH: Minimum content length for an action
        MIN_REASONING_LENGTH: Minimum reasoning length for an action
    """

    MIN_CONTENT_LENGTH: int = 10
    MIN_REASONING_LENGTH: int = 5


class WorkflowConfig:
    """Configuration for workflow and phases.

    Attributes:
        PHASES: List of workflow phases
        DEFAULT_TOTAL_STEPS: Default total steps in a workflow
    """

    PHASES: list[str] = ["research", "product", "architecture", "planning", "validation", "strategy"]
    DEFAULT_TOTAL_STEPS: int = 10


class DocumentStatusConfig:
    """Configuration for document status transition rules.
    
    Attributes:
    STATUS_TRANSITIONS: Map of action to status change
    APPROVAL_REQUIREMENTS: Which documents need approval per task
    """
    
    # Map actions to status changes
    STATUS_TRANSITIONS: dict[str, str] = {
        # Taylor actions that change document status
        "MARK_DOCUMENT_REVIEW": "in_review",
        "APPROVE_DOCUMENT": "approved",
        "REJECT_DOCUMENT": "rejected",
        # Sam actions that change document status
        "APPROVE_ALL_DOCUMENTS": "approved",
        "FINAL_APPROVAL": "approved",
    }
    
    # Which documents require approval before final submission
    APPROVAL_REQUIREMENTS: dict[str, list[str]] = {
        "easy_user_authentication": ["PRD", "TRD"],
        "medium_chat_app": ["PRD", "ARCHITECTURE", "TASKS"],
        "hard_saas_platform": ["PRD", "ARCHITECTURE", "ROADMAP", "VALIDATION", "STRATEGY"],
    }


# ============================================================================
# Action to Document Mapping
# ============================================================================


ACTION_TO_DOCUMENT: dict[str, str] = {
    # Maya (Research Analyst) - RESEARCH actions
    "SEARCH_MARKET": "RESEARCH",
    "ANALYZE_COMPETITORS": "RESEARCH",
    "VALIDATE_PROBLEM": "RESEARCH",
    "SUMMARIZE_INSIGHTS": "RESEARCH",
    "IDENTIFY_OPPORTUNITIES": "RESEARCH",
    # Elon (Product Manager) - PRODUCT actions
    "WRITE_PRD": "PRD",
    "DEFINE_FEATURES": "PRD",
    "IDENTIFY_USER_PERSONA": "PRD",
    "PRIORITIZE_FEATURES": "PRD",
    "DEFINE_SUCCESS_METRICS": "PRD",
    # Jordan (Architect) - ARCHITECTURE actions
    "DESIGN_ARCHITECTURE": "ARCHITECTURE",
    "SELECT_TECH_STACK": "TRD",
    "DEFINE_APIS": "TRD",
    "DESIGN_DATA_MODEL": "TRD",
    "WRITE_TRD": "TRD",
    # Robert (Execution Planner) - PLANNING actions
    "CREATE_ROADMAP": "ROADMAP",
    "BREAK_INTO_TASKS": "TASKS",
    "PLAN_SPRINTS": "TASKS",
    "ESTIMATE_TIMELINES": "ROADMAP",
    "DEFINE_DEPENDENCIES": "TASKS",
    # Taylor (Validator) - VALIDATION actions
    "REVIEW_DOCUMENTS": "VALIDATION",
    "CHECK_CONSISTENCY": "VALIDATION",
    "VALIDATE_CLAIMS": "VALIDATION",
    "IDENTIFY_RISKS": "VALIDATION",
    "SCORE_PLAN": "VALIDATION",
    "MARK_DOCUMENT_REVIEW": "VALIDATION",
    "APPROVE_DOCUMENT": "VALIDATION",
    "REJECT_DOCUMENT": "VALIDATION",
    # Sam (CEO) - STRATEGY actions
    "SET_DIRECTION": "STRATEGY",
    "REVIEW_PLAN": "STRATEGY",
    "APPROVE_STRATEGY": "STRATEGY",
    "PRIORITIZE_OBJECTIVES": "STRATEGY",
    "REQUEST_REVISION": "STRATEGY",
    "APPROVE_ALL_DOCUMENTS": "STRATEGY",
    "FINAL_APPROVAL": "STRATEGY",
}


# ============================================================================
# Document Types
# ============================================================================


class DocumentType(str, Enum):
    """Enumeration of all document types in the SkyPlan system.

    Each document type represents a specific planning artifact:
    - RESEARCH: Market research and competitive analysis
    - PRD: Product Requirements Document
    - TRD: Technical Requirements Document
    - ARCHITECTURE: System architecture and design
    - ROADMAP: Product roadmap and milestones
    - TASKS: Task breakdown and sprint planning
    - VALIDATION: Validation report and quality assessment
    - STRATEGY: Strategic direction and approval
    """

    RESEARCH = "RESEARCH"
    PRD = "PRD"
    TRD = "TRD"
    ARCHITECTURE = "ARCHITECTURE"
    ROADMAP = "ROADMAP"
    TASKS = "TASKS"
    VALIDATION = "VALIDATION"
    STRATEGY = "STRATEGY"

    # Display names mapping
    _DISPLAY_NAMES: dict[str, str] = {
        RESEARCH: "Research Summary",
        PRD: "Product Requirements Document",
        TRD: "Technical Requirements Document",
        ARCHITECTURE: "System Architecture",
        ROADMAP: "Product Roadmap",
        TASKS: "Task Breakdown",
        VALIDATION: "Validation Report",
        STRATEGY: "Strategic Direction",
    }

    # Filename mapping
    _FILENAMES: dict[str, str] = {
        RESEARCH: "RESEARCH.md",
        PRD: "PRD.md",
        TRD: "TRD.md",
        ARCHITECTURE: "ARCHITECTURE.md",
        ROADMAP: "ROADMAP.md",
        TASKS: "TASKS.md",
        VALIDATION: "VALIDATION.md",
        STRATEGY: "STRATEGY.md",
    }

    @classmethod
    def get_display_name(cls, doc_type: str) -> str:
        """Get human-readable display name for a document type.

        Args:
            doc_type: The document type

        Returns:
            Human-readable display name
        """
        return cls._DISPLAY_NAMES.get(doc_type, doc_type)

    @classmethod
    def get_filename(cls, doc_type: str) -> str:
        """Get the filename for a document type.

        Args:
            doc_type: The document type

        Returns:
            Filename for the document
        """
        return cls._FILENAMES.get(doc_type, f"{doc_type}.md")


# ============================================================================
# Document Status
# ============================================================================


class DocumentStatus(str, Enum):
    """Enumeration of document status values."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================================
# Feedback Types
# ============================================================================


class FeedbackType(str, Enum):
    """Enumeration of feedback types in the SkyPlan system.

    Each feedback type represents a different kind of peer review:
    - SUGGESTION: Ideas for improvement
    - CRITIQUE: Critical feedback on issues
    - QUESTION: Clarification requests
    - APPROVAL: Positive feedback or approval
    - CONCERN: Risk or issue identification
    - REQUEST_REVISION: Request for changes
    """

    SUGGESTION = "suggestion"
    CRITIQUE = "critique"
    QUESTION = "question"
    APPROVAL = "approval"
    CONCERN = "concern"
    REQUEST_REVISION = "request_revision"

    # Display names mapping
    _DISPLAY_NAMES: dict[str, str] = {
        SUGGESTION: "Suggestion",
        CRITIQUE: "Critique",
        QUESTION: "Question",
        APPROVAL: "Approval",
        CONCERN: "Concern",
        REQUEST_REVISION: "Request for Revision",
    }

    @classmethod
    def get_display_name(cls, feedback_type: str) -> str:
        """Get human-readable display name for a feedback type.

        Args:
            feedback_type: The feedback type

        Returns:
            Human-readable display name
        """
        return cls._DISPLAY_NAMES.get(feedback_type, feedback_type)


# ============================================================================
# Action Result Types
# ============================================================================


class ActionResult(str, Enum):
    """Enumeration of action result statuses in the SkyPlan system.

    Each result status represents the outcome of a previous action:
    - SUCCESS: The action completed successfully
    - FAILURE: The action failed due to an error or issue
    - PARTIAL: The action completed but with some issues or incomplete work
    - REJECTED: The action was rejected (e.g., document validation failed)
    - PENDING: The action is still being processed
    """

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    REJECTED = "rejected"
    PENDING = "pending"

    # Display names mapping
    _DISPLAY_NAMES: dict[str, str] = {
        SUCCESS: "Success",
        FAILURE: "Failure",
        PARTIAL: "Partial",
        REJECTED: "Rejected",
        PENDING: "Pending",
    }

    # Successful outcomes
    _SUCCESSFUL_OUTCOMES: set[str] = {SUCCESS, PARTIAL}

    # Failure outcomes
    _FAILURE_OUTCOMES: set[str] = {FAILURE, REJECTED}

    @classmethod
    def get_display_name(cls, result: str) -> str:
        """Get human-readable display name for an action result.

        Args:
            result: The result status

        Returns:
            Human-readable display name
        """
        return cls._DISPLAY_NAMES.get(result, result)

    @classmethod
    def is_successful(cls, result: str) -> bool:
        """Check if the result indicates a successful outcome.

        Args:
            result: The result status

        Returns:
            True if successful
        """
        return result in cls._SUCCESSFUL_OUTCOMES

    @classmethod
    def is_failure(cls, result: str) -> bool:
        """Check if the result indicates a failed outcome.

        Args:
            result: The result status

        Returns:
            True if failed
        """
        return result in cls._FAILURE_OUTCOMES


# ============================================================================
# Document Model
# ============================================================================


class Document(BaseModel):
    """Represents a planning document in the SkyPlan system.

    Each document contains:
    - type: The type of document (e.g., PRD, TRD)
    - content: The actual document content
    - author: The agent who created/last modified the document
    - created_at: Timestamp when the document was created
    - updated_at: Timestamp when the document was last updated
    - status: Current status of the document (draft, in_review, approved, rejected)
    """

    type: str = Field(..., description="The type of document (e.g., PRD, TRD)")
    content: str = Field(default="", description="The actual document content")
    author: str = Field(..., description="The agent who created/last modified the document")
    created_at: str = Field(default="", description="Timestamp when the document was created")
    updated_at: str = Field(default="", description="Timestamp when the document was last updated")
    status: Literal["draft", "in_review", "approved", "rejected"] = Field(
        default="draft",
        description="Current status of the document",
    )

    @classmethod
    def create(
        cls,
        doc_type: str,
        content: str,
        author: str,
    ) -> "Document":
        """Factory method to create a document with auto-generated timestamps.

        Args:
            doc_type: The type of document
            content: The document content
            author: The agent who created the document

        Returns:
            A new Document instance
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        return cls(
            type=doc_type,
            content=content,
            author=author,
            created_at=timestamp,
            updated_at=timestamp,
            status=DocumentStatus.DRAFT,
        )

    def update_content(self, content: str, author: str) -> None:
        """Update the document content and metadata.

        Args:
            content: New content
            author: Agent who made the update
        """
        self.content = content
        self.author = author
        self.updated_at = datetime.utcnow().isoformat() + "Z"


# ============================================================================
# Feedback Model
# ============================================================================


class Feedback(BaseModel):
    """Represents peer review feedback in the SkyPlan system.

    Each feedback entry contains:
    - from_agent: Who gave the feedback
    - to_agent: Who the feedback is for (optional, can be general)
    - document_type: Which document this feedback is about (optional)
    - feedback_type: The type of feedback (suggestion, critique, etc.)
    - comment: The actual feedback text
    - timestamp: When the feedback was given
    - resolved: Whether the feedback has been addressed
    """

    from_agent: str = Field(..., description="The agent who provided this feedback")
    to_agent: str = Field(
        default="",
        description="The agent this feedback is for (empty if general)",
    )
    document_type: str = Field(
        default="",
        description="The document type this feedback is about (empty if general)",
    )
    feedback_type: Literal["suggestion", "critique", "question", "approval", "concern", "request_revision"] = Field(
        default="suggestion",
        description="The type of feedback",
    )
    comment: str = Field(..., description="The actual feedback text or critique")
    timestamp: str = Field(default="", description="Timestamp when the feedback was given")
    resolved: bool = Field(
        default=False,
        description="Whether this feedback has been addressed/resolved",
    )

    @classmethod
    def create(
        cls,
        from_agent: str,
        comment: str,
        feedback_type: str = "suggestion",
        to_agent: str = "",
        document_type: str = "",
    ) -> "Feedback":
        """Factory method to create a feedback entry with auto-generated timestamp.

        Args:
            from_agent: The agent providing the feedback
            comment: The feedback text
            feedback_type: Type of feedback (suggestion, critique, etc.)
            to_agent: Target agent (optional)
            document_type: Related document (optional)

        Returns:
            A new Feedback instance
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            document_type=document_type,
            feedback_type=feedback_type,
            comment=comment,
            timestamp=timestamp,
            resolved=False,
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of this feedback.

        Returns:
            Formatted summary string
        """
        from_name = AgentId.get_display_name(self.from_agent)
        type_name = FeedbackType.get_display_name(self.feedback_type)

        parts = [f"[{type_name}] {from_name}:"]
        if self.document_type:
            parts.append(f"({self.document_type})")
        parts.append(self.comment)
        return " ".join(parts)


# ============================================================================
# Last Action Model
# ============================================================================


class LastAction(BaseModel):
    """Represents the result of the previous action in the SkyPlan system.

    This model provides information about the last action taken, allowing
    the current agent to determine if the previous work was successful
    or if it needs to be revisited.

    Contains:
    - agent_id: Who took the last action
    - action_type: What action was taken
    - result: The outcome status (success, failure, partial, rejected, pending)
    - message: A descriptive message about the result
    - timestamp: When the action was completed
    """

    agent_id: str = Field(..., description="The agent who took the last action")
    action_type: str = Field(..., description="The type of action that was taken")
    result: Literal["success", "failure", "partial", "rejected", "pending"] = Field(
        default="pending",
        description="The outcome status of the last action",
    )
    message: str = Field(
        default="",
        description="A descriptive message about the result (e.g., 'PRD created successfully' or 'Document rejected due to missing sections')",
    )
    timestamp: str = Field(default="", description="Timestamp when the action was completed")

    @classmethod
    def create(
        cls,
        agent_id: str,
        action_type: str,
        result: str,
        message: str = "",
    ) -> "LastAction":
        """Factory method to create a LastAction entry with auto-generated timestamp.

        Args:
            agent_id: The agent who took the action
            action_type: The type of action taken
            result: The outcome status
            message: Descriptive message about the result

        Returns:
            A new LastAction instance
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        return cls(
            agent_id=agent_id,
            action_type=action_type,
            result=result,
            message=message,
            timestamp=timestamp,
        )

    def is_successful(self) -> bool:
        """Check if the last action was successful.

        Returns:
            True if successful
        """
        return ActionResult.is_successful(self.result)

    def is_failure(self) -> bool:
        """Check if the last action failed.

        Returns:
            True if failed
        """
        return ActionResult.is_failure(self.result)

    def get_summary(self) -> str:
        """Get a human-readable summary of the last action.

        Returns:
            Formatted summary string
        """
        agent_name = AgentId.get_display_name(self.agent_id)
        result_name = ActionResult.get_display_name(self.result)
        return f"{agent_name} performed {self.action_type}: {result_name} - {self.message}"


# ============================================================================
# Agent Definitions
# ============================================================================


class AgentId(str, Enum):
    """Enumeration of all available agents in the SkyPlan system.

    Each agent has a specialized role in the planning workflow:
    - MAYA: Research Analyst - conducts market research and competitive analysis
    - ELON: Product Manager - defines product requirements and features
    - JORDAN: Architect - designs system architecture and technical specifications
    - ROBERT: Execution Planner - creates roadmaps and sprint backlogs
    - TAYLOR: Validator - validates all planning artifacts for quality and completeness
    - SAM: CEO - provides final strategic approval and direction

    Workflow Order: Maya → Elon → Jordan → Robert → Taylor → Sam
    """

    MAYA = "maya"
    ELON = "elon"
    JORDAN = "jordan"
    ROBERT = "robert"
    TAYLOR = "taylor"
    SAM = "sam"

    # Define the workflow order for agent progression
    WORKFLOW_ORDER = [MAYA, ELON, JORDAN, ROBERT, TAYLOR, SAM]

    # Display names mapping
    _DISPLAY_NAMES: dict[str, str] = {
        MAYA: "Maya (Research Analyst)",
        ELON: "Elon (Product Manager)",
        JORDAN: "Jordan (Architect)",
        ROBERT: "Robert (Execution Planner)",
        TAYLOR: "Taylor (Validator)",
        SAM: "Sam (CEO)",
    }

    @classmethod
    def get_display_name(cls, agent_id: str) -> str:
        """Get human-readable display name for an agent ID.

        This method delegates to the workflow module for data-driven configuration.

        Args:
            agent_id: The agent ID

        Returns:
            Human-readable display name
        """
        try:
            from .workflow import get_agent_name
            return get_agent_name(agent_id)
        except ImportError:
            # Fallback to hardcoded values if workflow module is not available
            return cls._DISPLAY_NAMES.get(agent_id, agent_id)

    @classmethod
    def get_next_agent(cls, current_agent: str) -> str:
        """Get the next agent in the workflow order.

        This method delegates to the workflow module for data-driven configuration.

        Args:
            current_agent: The current agent ID

        Returns:
            The next agent ID in the workflow, or empty string if at the end
        """
        try:
            from .workflow import get_next_agent
            return get_next_agent(current_agent) or ""
        except ImportError:
            # Fallback to hardcoded values if workflow module is not available
            try:
                current_index = cls.WORKFLOW_ORDER.index(cls(current_agent))
                next_index = (current_index + 1) % len(cls.WORKFLOW_ORDER)
                return cls.WORKFLOW_ORDER[next_index].value
            except (ValueError, AttributeError):
                return cls.MAYA.value

    @classmethod
    def get_workflow_position(cls, agent_id: str) -> int:
        """Get the position of an agent in the workflow (0-indexed).

        This method delegates to the workflow module for data-driven configuration.

        Args:
            agent_id: The agent ID

        Returns:
            The position in the workflow, or -1 if not found
        """
        try:
            from .workflow import get_workflow_position
            return get_workflow_position(agent_id)
        except ImportError:
            # Fallback to hardcoded values if workflow module is not available
            try:
                return cls.WORKFLOW_ORDER.index(cls(agent_id))
            except (ValueError, AttributeError):
                return -1


# ============================================================================
# Action Type Definitions
# ============================================================================


class ActionType(str, Enum):
    """Enumeration of all possible action types in the SkyPlan system.

    Actions are categorized by the agent that can perform them and the type of work:
    - RESEARCH: Market analysis, competitive research, problem validation
    - PRODUCT: PRD writing, feature definition, user persona identification
    - ARCHITECTURE: System design, tech stack selection, API definition
    - PLANNING: Roadmap creation, task breakdown, sprint planning
    - VALIDATION: Document review, consistency checks, risk identification
    - STRATEGY: Direction setting, plan review, approval, prioritization
    """

    # Maya (Research Analyst) - RESEARCH actions
    SEARCH_MARKET = "SEARCH_MARKET"
    ANALYZE_COMPETITORS = "ANALYZE_COMPETITORS"
    VALIDATE_PROBLEM = "VALIDATE_PROBLEM"
    SUMMARIZE_INSIGHTS = "SUMMARIZE_INSIGHTS"
    IDENTIFY_OPPORTUNITIES = "IDENTIFY_OPPORTUNITIES"

    # Elon (Product Manager) - PRODUCT actions
    WRITE_PRD = "WRITE_PRD"
    DEFINE_FEATURES = "DEFINE_FEATURES"
    IDENTIFY_USER_PERSONA = "IDENTIFY_USER_PERSONA"
    PRIORITIZE_FEATURES = "PRIORITIZE_FEATURES"
    DEFINE_SUCCESS_METRICS = "DEFINE_SUCCESS_METRICS"

    # Jordan (Architect) - ARCHITECTURE actions
    DESIGN_ARCHITECTURE = "DESIGN_ARCHITECTURE"
    SELECT_TECH_STACK = "SELECT_TECH_STACK"
    DEFINE_APIS = "DEFINE_APIS"
    DESIGN_DATA_MODEL = "DESIGN_DATA_MODEL"
    WRITE_TRD = "WRITE_TRD"

    # Robert (Execution Planner) - PLANNING actions
    CREATE_ROADMAP = "CREATE_ROADMAP"
    BREAK_INTO_TASKS = "BREAK_INTO_TASKS"
    PLAN_SPRINTS = "PLAN_SPRINTS"
    ESTIMATE_TIMELINES = "ESTIMATE_TIMELINES"
    DEFINE_DEPENDENCIES = "DEFINE_DEPENDENCIES"

    # Taylor (Validator) - VALIDATION actions
    REVIEW_DOCUMENTS = "REVIEW_DOCUMENTS"
    CHECK_CONSISTENCY = "CHECK_CONSISTENCY"
    VALIDATE_CLAIMS = "VALIDATE_CLAIMS"
    IDENTIFY_RISKS = "IDENTIFY_RISKS"
    SCORE_PLAN = "SCORE_PLAN"

    # Sam (CEO) - STRATEGY actions
    SET_DIRECTION = "SET_DIRECTION"
    REVIEW_PLAN = "REVIEW_PLAN"
    APPROVE_STRATEGY = "APPROVE_STRATEGY"
    PRIORITIZE_OBJECTIVES = "PRIORITIZE_OBJECTIVES"
    REQUEST_REVISION = "REQUEST_REVISION"

    # Action to category mapping
    _ACTION_CATEGORIES: dict[str, str] = {
        # RESEARCH category
        SEARCH_MARKET: "RESEARCH",
        ANALYZE_COMPETITORS: "RESEARCH",
        VALIDATE_PROBLEM: "RESEARCH",
        SUMMARIZE_INSIGHTS: "RESEARCH",
        IDENTIFY_OPPORTUNITIES: "RESEARCH",
        # PRODUCT category
        WRITE_PRD: "PRODUCT",
        DEFINE_FEATURES: "PRODUCT",
        IDENTIFY_USER_PERSONA: "PRODUCT",
        PRIORITIZE_FEATURES: "PRODUCT",
        DEFINE_SUCCESS_METRICS: "PRODUCT",
        # ARCHITECTURE category
        DESIGN_ARCHITECTURE: "ARCHITECTURE",
        SELECT_TECH_STACK: "ARCHITECTURE",
        DEFINE_APIS: "ARCHITECTURE",
        DESIGN_DATA_MODEL: "ARCHITECTURE",
        WRITE_TRD: "ARCHITECTURE",
        # PLANNING category
        CREATE_ROADMAP: "PLANNING",
        BREAK_INTO_TASKS: "PLANNING",
        PLAN_SPRINTS: "PLANNING",
        ESTIMATE_TIMELINES: "PLANNING",
        DEFINE_DEPENDENCIES: "PLANNING",
        # VALIDATION category
        REVIEW_DOCUMENTS: "VALIDATION",
        CHECK_CONSISTENCY: "VALIDATION",
        VALIDATE_CLAIMS: "VALIDATION",
        IDENTIFY_RISKS: "VALIDATION",
        SCORE_PLAN: "VALIDATION",
        # STRATEGY category
        SET_DIRECTION: "STRATEGY",
        REVIEW_PLAN: "STRATEGY",
        APPROVE_STRATEGY: "STRATEGY",
        PRIORITIZE_OBJECTIVES: "STRATEGY",
        REQUEST_REVISION: "STRATEGY",
    }

    @classmethod
    def get_category(cls, action_type: str) -> str:
        """Get the category of an action type.

        Args:
            action_type: The action type

        Returns:
            The category name
        """
        return cls._ACTION_CATEGORIES.get(action_type, "UNKNOWN")

    @classmethod
    def get_allowed_actions_for_agent(cls, agent_id: str) -> list[str]:
        """Get the list of actions that a specific agent can perform.

        This method delegates to the workflow module for data-driven configuration.

        Args:
            agent_id: The agent ID

        Returns:
            List of action types allowed for this agent
        """
        try:
            from .workflow import get_allowed_actions
            return get_allowed_actions(agent_id)
        except ImportError:
            # Fallback to hardcoded values if workflow module is not available
            agent_actions = {
                "maya": [
                    cls.SEARCH_MARKET.value,
                    cls.ANALYZE_COMPETITORS.value,
                    cls.VALIDATE_PROBLEM.value,
                    cls.SUMMARIZE_INSIGHTS.value,
                    cls.IDENTIFY_OPPORTUNITIES.value,
                ],
                "elon": [
                    cls.WRITE_PRD.value,
                    cls.DEFINE_FEATURES.value,
                    cls.IDENTIFY_USER_PERSONA.value,
                    cls.PRIORITIZE_FEATURES.value,
                    cls.DEFINE_SUCCESS_METRICS.value,
                ],
                "jordan": [
                    cls.DESIGN_ARCHITECTURE.value,
                    cls.SELECT_TECH_STACK.value,
                    cls.DEFINE_APIS.value,
                    cls.DESIGN_DATA_MODEL.value,
                    cls.WRITE_TRD.value,
                ],
                "robert": [
                    cls.CREATE_ROADMAP.value,
                    cls.BREAK_INTO_TASKS.value,
                    cls.PLAN_SPRINTS.value,
                    cls.ESTIMATE_TIMELINES.value,
                    cls.DEFINE_DEPENDENCIES.value,
                ],
                "taylor": [
                    cls.REVIEW_DOCUMENTS.value,
                    cls.CHECK_CONSISTENCY.value,
                    cls.VALIDATE_CLAIMS.value,
                    cls.IDENTIFY_RISKS.value,
                    cls.SCORE_PLAN.value,
                ],
                "sam": [
                    cls.SET_DIRECTION.value,
                    cls.REVIEW_PLAN.value,
                    cls.APPROVE_STRATEGY.value,
                    cls.PRIORITIZE_OBJECTIVES.value,
                    cls.REQUEST_REVISION.value,
                ],
            }
            return agent_actions.get(agent_id, [])


# ============================================================================
# Action and Observation Models
# ============================================================================


class SkyPlanAction(Action):
    """Action for the SkyPlan environment.

    Each action must specify:
    - agent_id: Who is taking the action (required)
    - action_type: What type of action is being performed (categorized by work type)
    - reasoning: Why the agent decided to take this action at this time (thought process)
    - content: The actual product of their effort (the work output)

    The action_type is categorized into:
    - RESEARCH: Market analysis, competitive research, problem validation
    - PRODUCT: PRD writing, feature definition, user persona identification
    - ARCHITECTURE: System design, tech stack selection, API definition
    - PLANNING: Roadmap creation, task breakdown, sprint planning
    - VALIDATION: Document review, consistency checks, risk identification
    - STRATEGY: Direction setting, plan review, approval, prioritization
    """

    agent_id: Literal["maya", "elon", "jordan", "robert", "taylor", "sam"] = Field(
        ...,
        description="The ID of the agent taking this action. Must be one of: maya, elon, jordan, robert, taylor, sam",
    )
    action_type: str = Field(
        ...,
        description="The type of action being performed. Must be one of the valid ActionType values for the specified agent.",
    )
    reasoning: str = Field(
        ...,
        description="The agent's thought process explaining why this specific action was chosen at this time. This helps with debugging and system improvement.",
    )
    content: str = Field(
        default="",
        description="The actual product of the agent's effort. For WRITE_PRD, this contains the PRD text. For DESIGN_ARCHITECTURE, this contains the architecture specification.",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type_for_agent(cls, v: str, info) -> str:
        """Validate that the action_type is allowed for the specified agent_id.

        Args:
            v: The action type value
            info: Validation info containing the agent_id

        Returns:
            The validated action type

        Raises:
            ValueError: If action_type is not allowed for the agent
        """
        agent_id = info.data.get("agent_id")
        if agent_id:
            allowed_actions = ActionType.get_allowed_actions_for_agent(agent_id)
            if v not in allowed_actions:
                raise ValueError(
                    f"Action '{v}' is not allowed for agent '{agent_id}'. "
                    f"Allowed actions for {agent_id}: {allowed_actions}"
                )
        return v


class SkyPlanObservation(Observation):
    """Observation from the SkyPlan environment.

    Provides feedback to agents about:
    - The task description (the work order/goal)
    - The result of their action
    - The system's reasoning for the result (why it succeeded/failed)
    - Who's next to take action (current_agent)
    - Progress tracking (step_number)
    - Documents: The shared folder containing all work produced so far
    - Feedback: Peer reviews and critiques from previous agents
    - Last action result: Success/failure check for the previous action
    - Current state of the planning documents
    - Any errors or validation issues
    """

    task_description: str = Field(
        default="",
        description="The work order/goal that the team is trying to achieve. A simple reminder of the specific objective for the current task.",
    )
    result: str = Field(default="", description="Result message from the action")
    reasoning: str = Field(
        default="",
        description="The system's reasoning explaining why this result was produced. Helps agents understand the decision-making process.",
    )
    current_agent: Literal["maya", "elon", "jordan", "robert", "taylor", "sam"] = Field(
        default="maya",
        description="The agent whose turn it is to take the next action. Like a scoreboard showing who is 'holding the pen'.",
    )
    step_number: int = Field(
        default=1,
        description="The current step number in the overall project workflow. Tracks progress (e.g., 'Step 4 of 10').",
    )
    total_steps: int = Field(
        default=WorkflowConfig.DEFAULT_TOTAL_STEPS,
        description="The total number of steps in the current project workflow. Used with step_number to show progress.",
    )
    documents: dict[str, Document] = Field(
        default_factory=dict,
        description="The shared folder containing all work produced by agents so far. Maps document types to Document objects (e.g., 'PRD' → Document with PRD content).",
    )
    feedback: list[Feedback] = Field(
        default_factory=list,
        description="Peer reviews and critiques from previous agents. Lists any comments or critiques (e.g., 'The architecture is too complex').",
    )
    last_action_result: LastAction | None = Field(
        default=None,
        description="Success/failure check for the previous action. Tells the agent if the person who just went before them succeeded or if something went wrong (like a system error or a rejected document).",
    )
    current_state: dict = Field(
        default_factory=dict,
        description="Current state of all planning documents",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of any errors encountered during action execution",
    )
    step_count: int = Field(default=0, description="Current step number in the episode")

    def get_feedback_for_agent(self, agent_id: str) -> list[Feedback]:
        """Get all feedback addressed to a specific agent.

        Args:
            agent_id: The agent ID to filter feedback for

        Returns:
            List of feedback entries for the specified agent
        """
        return [f for f in self.feedback if f.to_agent == agent_id or f.to_agent == ""]

    def get_feedback_for_document(self, document_type: str) -> list[Feedback]:
        """Get all feedback related to a specific document.

        Args:
            document_type: The document type to filter feedback for

        Returns:
            List of feedback entries for the specified document
        """
        return [f for f in self.feedback if f.document_type == document_type]

    def get_unresolved_feedback(self) -> list[Feedback]:
        """Get all unresolved feedback.

        Returns:
            List of unresolved feedback entries
        """
        return [f for f in self.feedback if not f.resolved]

    def was_previous_action_successful(self) -> bool:
        """Check if the previous action was successful.

        Returns:
            True if the previous action succeeded, False otherwise or if no previous action exists
        """
        if self.last_action_result is None:
            return False
        return self.last_action_result.is_successful()

    def did_previous_action_fail(self) -> bool:
        """Check if the previous action failed.

        Returns:
            True if the previous action failed, False otherwise or if no previous action exists
        """
        if self.last_action_result is None:
            return False
        return self.last_action_result.is_failure()

    def get_previous_action_summary(self) -> str:
        """Get a summary of the previous action.

        Returns:
            Human-readable summary of the previous action, or empty string if no previous action exists
        """
        if self.last_action_result is None:
            return ""
        return self.last_action_result.get_summary()
