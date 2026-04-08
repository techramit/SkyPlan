# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SkyPlan Workflow Configuration.

Defines the agent workflow (assembly line) for the multi-agent planning system.
The WORKFLOW list orchestrates the sequential hand-off between agents.

Each agent in the workflow:
- Has a specific role and responsibilities
- Produces specific document types
- Requires documents from previous agents
- Has a set of allowed actions
- Provides a handoff message to the next agent
"""

from typing import Literal

# ============================================================================
# WORKFLOW Definition
# ============================================================================

WORKFLOW = [
    {
        "agent_id": "maya",
        "name": "Maya",
        "role": "Research Analyst",
        "position": 1,
        "description": "Conducts market research and validates the idea",
        "responsibilities": [
            "Search market for similar products",
            "Analyze competitors",
            "Validate the problem space",
            "Summarize key insights",
            "Identify opportunities"
        ],
        "produces": ["RESEARCH"],
        "requires": [],  # No previous documents needed - starts the workflow
        "allowed_actions": [
            "SEARCH_MARKET",
            "ANALYZE_COMPETITORS",
            "VALIDATE_PROBLEM",
            "SUMMARIZE_INSIGHTS",
            "IDENTIFY_OPPORTUNITIES"
        ],
        "handoff_message": "Research complete. Ready for product definition.",
        "next_agent": "elon"
    },
    {
        "agent_id": "elon",
        "name": "Elon",
        "role": "Product Manager",
        "position": 2,
        "description": "Defines product requirements and features",
        "responsibilities": [
            "Write the Product Requirements Document (PRD)",
            "Define product features",
            "Identify user personas",
            "Prioritize features",
            "Define success metrics"
        ],
        "produces": ["PRD"],
        "requires": ["RESEARCH"],  # Needs Maya's research
        "allowed_actions": [
            "WRITE_PRD",
            "DEFINE_FEATURES",
            "IDENTIFY_USER_PERSONA",
            "PRIORITIZE_FEATURES",
            "DEFINE_SUCCESS_METRICS"
        ],
        "handoff_message": "PRD complete. Ready for technical design.",
        "next_agent": "jordan"
    },
    {
        "agent_id": "jordan",
        "name": "Jordan",
        "role": "Architect",
        "position": 3,
        "description": "Designs system architecture and technical approach",
        "responsibilities": [
            "Design system architecture",
            "Select technology stack",
            "Define APIs",
            "Design data model",
            "Write Technical Requirements Document (TRD)"
        ],
        "produces": ["TRD", "ARCHITECTURE"],
        "requires": ["RESEARCH", "PRD"],  # Needs research and PRD
        "allowed_actions": [
            "DESIGN_ARCHITECTURE",
            "SELECT_TECH_STACK",
            "DEFINE_APIS",
            "DESIGN_DATA_MODEL",
            "WRITE_TRD"
        ],
        "handoff_message": "Technical design complete. Ready for execution planning.",
        "next_agent": "robert"
    },
    {
        "agent_id": "robert",
        "name": "Robert",
        "role": "Execution Planner",
        "position": 4,
        "description": "Creates roadmap and task breakdown",
        "responsibilities": [
            "Create product roadmap",
            "Break down into tasks",
            "Plan sprints",
            "Estimate timelines",
            "Define dependencies"
        ],
        "produces": ["ROADMAP", "TASKS"],
        "requires": ["RESEARCH", "PRD", "TRD", "ARCHITECTURE"],  # Needs all previous work
        "allowed_actions": [
            "CREATE_ROADMAP",
            "BREAK_INTO_TASKS",
            "PLAN_SPRINTS",
            "ESTIMATE_TIMELINES",
            "DEFINE_DEPENDENCIES"
        ],
        "handoff_message": "Roadmap and tasks complete. Ready for validation.",
        "next_agent": "taylor"
    },
    {
        "agent_id": "taylor",
        "name": "Taylor",
        "role": "Validator",
        "position": 5,
        "description": "Reviews plans for quality and consistency",
        "responsibilities": [
            "Review all documents",
            "Check consistency across documents",
            "Validate claims and assumptions",
            "Identify risks",
            "Score the overall plan"
        ],
        "produces": ["VALIDATION"],
        "requires": ["RESEARCH", "PRD", "TRD", "ARCHITECTURE", "ROADMAP", "TASKS"],  # Needs everything
        "allowed_actions": [
            "REVIEW_DOCUMENTS",
            "CHECK_CONSISTENCY",
            "VALIDATE_CLAIMS",
            "IDENTIFY_RISKS",
            "SCORE_PLAN"
        ],
        "handoff_message": "Validation complete. Ready for CEO approval.",
        "next_agent": "sam"
    },
    {
        "agent_id": "sam",
        "name": "Sam",
        "role": "CEO",
        "position": 6,
        "description": "Provides final strategic approval",
        "responsibilities": [
            "Set strategic direction",
            "Review the complete plan",
            "Approve the strategy",
            "Prioritize objectives",
            "Request revisions if needed"
        ],
        "produces": ["STRATEGY"],
        "requires": ["RESEARCH", "PRD", "TRD", "ARCHITECTURE", "ROADMAP", "TASKS", "VALIDATION"],  # Needs everything
        "allowed_actions": [
            "SET_DIRECTION",
            "REVIEW_PLAN",
            "APPROVE_STRATEGY",
            "PRIORITIZE_OBJECTIVES",
            "REQUEST_REVISION"
        ],
        "handoff_message": "Strategy approved. Project ready for execution.",
        "next_agent": None  # End of workflow
    }
]

# ============================================================================
# Agent ID Type
# ============================================================================

AgentId = Literal["maya", "elon", "jordan", "robert", "taylor", "sam"]

# ============================================================================
# Helper Functions
# ============================================================================


def get_workflow_entry(agent_id: str) -> dict | None:
    """Get workflow entry for a specific agent.

    Args:
        agent_id: The agent ID to look up

    Returns:
        The workflow entry dict, or None if not found
    """
    for entry in WORKFLOW:
        if entry["agent_id"] == agent_id:
            return entry
    return None


def get_next_agent(agent_id: str) -> str | None:
    """Get the next agent in the workflow.

    Args:
        agent_id: The current agent ID

    Returns:
        The next agent ID, or None if this is the last agent
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["next_agent"]
    return None


def get_required_documents(agent_id: str) -> list[str]:
    """Get documents required by a specific agent.

    Args:
        agent_id: The agent ID

    Returns:
        List of document types required by this agent
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["requires"]
    return []


def get_produced_documents(agent_id: str) -> list[str]:
    """Get documents produced by a specific agent.

    Args:
        agent_id: The agent ID

    Returns:
        List of document types produced by this agent
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["produces"]
    return []


def get_handoff_message(agent_id: str) -> str:
    """Get the handoff message for when an agent completes.

    Args:
        agent_id: The agent ID

    Returns:
        The handoff message to pass to the next agent
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["handoff_message"]
    return ""


def get_allowed_actions(agent_id: str) -> list[str]:
    """Get allowed actions for a specific agent.

    Args:
        agent_id: The agent ID

    Returns:
        List of action types allowed for this agent
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["allowed_actions"]
    return []


def get_agent_name(agent_id: str) -> str:
    """Get the display name for an agent.

    Args:
        agent_id: The agent ID

    Returns:
        The agent's display name
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["name"]
    return agent_id


def get_agent_role(agent_id: str) -> str:
    """Get the role description for an agent.

    Args:
        agent_id: The agent ID

    Returns:
        The agent's role description
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["role"]
    return ""


def get_workflow_position(agent_id: str) -> int:
    """Get the position of an agent in the workflow (1-indexed).

    Args:
        agent_id: The agent ID

    Returns:
        The position in the workflow, or -1 if not found
    """
    entry = get_workflow_entry(agent_id)
    if entry:
        return entry["position"]
    return -1


def get_all_agent_ids() -> list[str]:
    """Get all agent IDs in workflow order.

    Returns:
        List of agent IDs in workflow order
    """
    return [entry["agent_id"] for entry in WORKFLOW]


def get_first_agent() -> str:
    """Get the first agent in the workflow.

    Returns:
        The first agent ID
    """
    if WORKFLOW:
        return WORKFLOW[0]["agent_id"]
    return ""


def is_last_agent(agent_id: str) -> bool:
    """Check if an agent is the last in the workflow.

    Args:
        agent_id: The agent ID

    Returns:
        True if this is the last agent, False otherwise
    """
    return get_next_agent(agent_id) is None


def get_workflow_summary() -> str:
    """Get a human-readable summary of the workflow.

    Returns:
        A formatted string describing the workflow
    """
    lines = ["SkyPlan Workflow:", "=" * 50]
    for entry in WORKFLOW:
        lines.append(f"\n{entry['position']}. {entry['name']} ({entry['role']})")
        lines.append(f"   Produces: {', '.join(entry['produces'])}")
        lines.append(f"   Requires: {', '.join(entry['requires']) if entry['requires'] else 'None'}")
    return "\n".join(lines)


def validate_action_for_agent(agent_id: str, action_type: str) -> bool:
    """Check if an action is allowed for a specific agent.

    Args:
        agent_id: The agent ID
        action_type: The action type to validate

    Returns:
        True if the action is allowed, False otherwise
    """
    allowed = get_allowed_actions(agent_id)
    return action_type in allowed


def get_all_document_types() -> list[str]:
    """Get all document types produced across the workflow.

    Returns:
        List of all document types
    """
    docs = set()
    for entry in WORKFLOW:
        docs.update(entry["produces"])
    return sorted(list(docs))


def get_workflow_length() -> int:
    """Get the total number of agents in the workflow.

    Returns:
        Number of agents in the workflow
    """
    return len(WORKFLOW)


# ============================================================================
# Feedback Configuration
# ============================================================================

# Feedback types for generation
COLLABORATIVE_FEEDBACK_TYPES = [
    "suggestion",
    "approval",
    "question",
    "concern"
]

VALIDATOR_FEEDBACK_TYPES = [
    "critique",
    "concern",
    "request_revision"
]

STRATEGIC_FEEDBACK_TYPES = [
    "concern",
    "request_revision"
]

# Feedback reward weights
FEEDBACK_GENERATION_REWARDS = {
    "collaborative": 0.02,
    "validator": 0.08,
    "strategic": 0.10
}

FEEDBACK_RESOLUTION_REWARDS = {
    "primary": 0.15,  # Taylor/Sam feedback
    "peer": 0.05      # Peer feedback
}

DOCUMENT_APPROVAL_REWARDS = {
    "taylor": 0.15,
    "sam": 0.15,
    "final": 0.50
}


def map_feedback_type_to_reward(feedback_type: str, from_agent: str) -> float:
    """Map feedback type to reward value.

    Args:
        feedback_type: The feedback type from FeedbackType enum
        from_agent: Agent ID that generated the feedback

    Returns:
        Reward value for generating this feedback
    """
    if from_agent == "sam":
        return FEEDBACK_GENERATION_REWARDS["strategic"]
    elif from_agent == "taylor":
        return FEEDBACK_GENERATION_REWARDS["validator"]
    else:
        return FEEDBACK_GENERATION_REWARDS["collaborative"]
