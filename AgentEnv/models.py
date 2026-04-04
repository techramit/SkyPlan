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
from pydantic import Field, field_validator


class AgentId(str, Enum):
    """Enumeration of all available agents in the SkyPlan system.

    Each agent has a specialized role in the planning workflow:
    - MAYA: Research Analyst - conducts market research and competitive analysis
    - ELON: Product Manager - defines product requirements and features
    - JORDAN: Architect - designs system architecture and technical specifications
    - ROBERT: Execution Planner - creates roadmaps and sprint backlogs
    - TAYLOR: Validator - validates all planning artifacts for quality and completeness
    - SAM: CEO - provides final strategic approval and direction
    """

    MAYA = "maya"
    ELON = "elon"
    JORDAN = "jordan"
    ROBERT = "robert"
    TAYLOR = "taylor"
    SAM = "sam"

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
    - The result of their action
    - The system's reasoning for the result (why it succeeded/failed)
    - Current state of the planning documents
    - Any errors or validation issues
    """

    result: str = Field(default="", description="Result message from the action")
    reasoning: str = Field(
        default="",
        description="The system's reasoning explaining why this result was produced. Helps agents understand the decision-making process.",
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
