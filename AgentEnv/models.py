# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agentenv Environment.

The AgentEnv environment is a multi-agent planning system where specialized agents
collaborate to transform an idea into structured planning documents.
"""

from enum import Enum
from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


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

    @classmethod
    def get_display_name(cls, doc_type: str) -> str:
        """Get human-readable display name for a document type."""
        names = {
            cls.RESEARCH.value: "Research Summary",
            cls.PRD.value: "Product Requirements Document",
            cls.TRD.value: "Technical Requirements Document",
            cls.ARCHITECTURE.value: "System Architecture",
            cls.ROADMAP.value: "Product Roadmap",
            cls.TASKS.value: "Task Breakdown",
            cls.VALIDATION.value: "Validation Report",
            cls.STRATEGY.value: "Strategic Direction",
        }
        return names.get(doc_type, doc_type)

    @classmethod
    def get_filename(cls, doc_type: str) -> str:
        """Get the filename for a document type."""
        filenames = {
            cls.RESEARCH.value: "RESEARCH.md",
            cls.PRD.value: "PRD.md",
            cls.TRD.value: "TRD.md",
            cls.ARCHITECTURE.value: "ARCHITECTURE.md",
            cls.ROADMAP.value: "ROADMAP.md",
            cls.TASKS.value: "TASKS.md",
            cls.VALIDATION.value: "VALIDATION.md",
            cls.STRATEGY.value: "STRATEGY.md",
        }
        return filenames.get(doc_type, f"{doc_type}.md")


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

    @classmethod
    def get_display_name(cls, feedback_type: str) -> str:
        """Get human-readable display name for a feedback type."""
        names = {
            cls.SUGGESTION.value: "Suggestion",
            cls.CRITIQUE.value: "Critique",
            cls.QUESTION.value: "Question",
            cls.APPROVAL.value: "Approval",
            cls.CONCERN.value: "Concern",
            cls.REQUEST_REVISION.value: "Request for Revision",
        }
        return names.get(feedback_type, feedback_type)


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

    @classmethod
    def get_display_name(cls, result: str) -> str:
        """Get human-readable display name for an action result."""
        names = {
            cls.SUCCESS.value: "Success",
            cls.FAILURE.value: "Failure",
            cls.PARTIAL.value: "Partial",
            cls.REJECTED.value: "Rejected",
            cls.PENDING.value: "Pending",
        }
        return names.get(result, result)

    @classmethod
    def is_successful(cls, result: str) -> bool:
        """Check if the result indicates a successful outcome."""
        return result in [cls.SUCCESS.value, cls.PARTIAL.value]

    @classmethod
    def is_failure(cls, result: str) -> bool:
        """Check if the result indicates a failed outcome."""
        return result in [cls.FAILURE.value, cls.REJECTED.value]


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
        from datetime import datetime

        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            document_type=document_type,
            feedback_type=feedback_type,
            comment=comment,
            timestamp=datetime.utcnow().isoformat() + "Z",
            resolved=False,
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of this feedback."""
        from_name = AgentId.get_display_name(self.from_agent)
        type_name = FeedbackType.get_display_name(self.feedback_type)

        parts = [f"[{type_name}] {from_name}:"]
        if self.document_type:
            parts.append(f"({self.document_type})")
        parts.append(self.comment)
        return " ".join(parts)


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
        from datetime import datetime

        return cls(
            agent_id=agent_id,
            action_type=action_type,
            result=result,
            message=message,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def is_successful(self) -> bool:
        """Check if the last action was successful."""
        return ActionResult.is_successful(self.result)

    def is_failure(self) -> bool:
        """Check if the last action failed."""
        return ActionResult.is_failure(self.result)

    def get_summary(self) -> str:
        """Get a human-readable summary of the last action."""
        agent_name = AgentId.get_display_name(self.agent_id)
        result_name = ActionResult.get_display_name(self.result)
        return f"{agent_name} performed {self.action_type}: {result_name} - {self.message}"


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

    @classmethod
    def get_display_name(cls, agent_id: str) -> str:
        """Get human-readable display name for an agent ID."""
        names = {
            cls.MAYA.value: "Maya (Research Analyst)",
            cls.ELON.value: "Elon (Product Manager)",
            cls.JORDAN.value: "Jordan (Architect)",
            cls.ROBERT.value: "Robert (Execution Planner)",
            cls.TAYLOR.value: "Taylor (Validator)",
            cls.SAM.value: "Sam (CEO)",
        }
        return names.get(agent_id, agent_id)

    @classmethod
    def get_next_agent(cls, current_agent: str) -> str:
        """Get the next agent in the workflow order.

        Args:
            current_agent: The current agent ID

        Returns:
            The next agent ID in the workflow, or the first agent if at the end
        """
        try:
            current_index = cls.WORKFLOW_ORDER.index(cls(current_agent))
            next_index = (current_index + 1) % len(cls.WORKFLOW_ORDER)
            return cls.WORKFLOW_ORDER[next_index].value
        except (ValueError, AttributeError):
            return cls.MAYA.value

    @classmethod
    def get_workflow_position(cls, agent_id: str) -> int:
        """Get the position of an agent in the workflow (0-indexed).

        Args:
            agent_id: The agent ID

        Returns:
            The position in the workflow, or -1 if not found
        """
        try:
            return cls.WORKFLOW_ORDER.index(cls(agent_id))
        except (ValueError, AttributeError):
            return -1


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

    @classmethod
    def get_category(cls, action_type: str) -> str:
        """Get the category of an action type."""
        categories = {
            # RESEARCH category
            cls.SEARCH_MARKET.value: "RESEARCH",
            cls.ANALYZE_COMPETITORS.value: "RESEARCH",
            cls.VALIDATE_PROBLEM.value: "RESEARCH",
            cls.SUMMARIZE_INSIGHTS.value: "RESEARCH",
            cls.IDENTIFY_OPPORTUNITIES.value: "RESEARCH",
            # PRODUCT category
            cls.WRITE_PRD.value: "PRODUCT",
            cls.DEFINE_FEATURES.value: "PRODUCT",
            cls.IDENTIFY_USER_PERSONA.value: "PRODUCT",
            cls.PRIORITIZE_FEATURES.value: "PRODUCT",
            cls.DEFINE_SUCCESS_METRICS.value: "PRODUCT",
            # ARCHITECTURE category
            cls.DESIGN_ARCHITECTURE.value: "ARCHITECTURE",
            cls.SELECT_TECH_STACK.value: "ARCHITECTURE",
            cls.DEFINE_APIS.value: "ARCHITECTURE",
            cls.DESIGN_DATA_MODEL.value: "ARCHITECTURE",
            cls.WRITE_TRD.value: "ARCHITECTURE",
            # PLANNING category
            cls.CREATE_ROADMAP.value: "PLANNING",
            cls.BREAK_INTO_TASKS.value: "PLANNING",
            cls.PLAN_SPRINTS.value: "PLANNING",
            cls.ESTIMATE_TIMELINES.value: "PLANNING",
            cls.DEFINE_DEPENDENCIES.value: "PLANNING",
            # VALIDATION category
            cls.REVIEW_DOCUMENTS.value: "VALIDATION",
            cls.CHECK_CONSISTENCY.value: "VALIDATION",
            cls.VALIDATE_CLAIMS.value: "VALIDATION",
            cls.IDENTIFY_RISKS.value: "VALIDATION",
            cls.SCORE_PLAN.value: "VALIDATION",
            # STRATEGY category
            cls.SET_DIRECTION.value: "STRATEGY",
            cls.REVIEW_PLAN.value: "STRATEGY",
            cls.APPROVE_STRATEGY.value: "STRATEGY",
            cls.PRIORITIZE_OBJECTIVES.value: "STRATEGY",
            cls.REQUEST_REVISION.value: "STRATEGY",
        }
        return categories.get(action_type, "UNKNOWN")

    @classmethod
    def get_allowed_actions_for_agent(cls, agent_id: str) -> list[str]:
        """Get the list of actions that a specific agent can perform."""
        agent_actions = {
            AgentId.MAYA.value: [
                cls.SEARCH_MARKET.value,
                cls.ANALYZE_COMPETITORS.value,
                cls.VALIDATE_PROBLEM.value,
                cls.SUMMARIZE_INSIGHTS.value,
                cls.IDENTIFY_OPPORTUNITIES.value,
            ],
            AgentId.ELON.value: [
                cls.WRITE_PRD.value,
                cls.DEFINE_FEATURES.value,
                cls.IDENTIFY_USER_PERSONA.value,
                cls.PRIORITIZE_FEATURES.value,
                cls.DEFINE_SUCCESS_METRICS.value,
            ],
            AgentId.JORDAN.value: [
                cls.DESIGN_ARCHITECTURE.value,
                cls.SELECT_TECH_STACK.value,
                cls.DEFINE_APIS.value,
                cls.DESIGN_DATA_MODEL.value,
                cls.WRITE_TRD.value,
            ],
            AgentId.ROBERT.value: [
                cls.CREATE_ROADMAP.value,
                cls.BREAK_INTO_TASKS.value,
                cls.PLAN_SPRINTS.value,
                cls.ESTIMATE_TIMELINES.value,
                cls.DEFINE_DEPENDENCIES.value,
            ],
            AgentId.TAYLOR.value: [
                cls.REVIEW_DOCUMENTS.value,
                cls.CHECK_CONSISTENCY.value,
                cls.VALIDATE_CLAIMS.value,
                cls.IDENTIFY_RISKS.value,
                cls.SCORE_PLAN.value,
            ],
            AgentId.SAM.value: [
                cls.SET_DIRECTION.value,
                cls.REVIEW_PLAN.value,
                cls.APPROVE_STRATEGY.value,
                cls.PRIORITIZE_OBJECTIVES.value,
                cls.REQUEST_REVISION.value,
            ],
        }
        return agent_actions.get(agent_id, [])


class AgentenvAction(Action):
    """Action for the Agentenv environment.

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
        """Validate that the action_type is allowed for the specified agent_id."""
        agent_id = info.data.get("agent_id")
        if agent_id:
            allowed_actions = ActionType.get_allowed_actions_for_agent(agent_id)
            if v not in allowed_actions:
                raise ValueError(
                    f"Action '{v}' is not allowed for agent '{agent_id}'. "
                    f"Allowed actions for {agent_id}: {allowed_actions}"
                )
        return v


class AgentenvObservation(Observation):
    """Observation from the Agentenv environment.

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
        default=10,
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
