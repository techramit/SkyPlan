# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and graders for the SkyPlan environment.

Each task defines a concrete objective with a programmatic grader
that scores performance (0.0–1.0).
"""

from dataclasses import dataclass
from typing import Literal

from .models import Document


# ============================================================================
# Task Configuration
# ============================================================================


@dataclass
class TaskConfig:
    """Configuration for a task.

    Attributes:
        task_id: Unique identifier for the task
        name: Human-readable name of the task
        difficulty: Difficulty level (easy, medium, hard)
        description: The task description/prompt
        min_content_length: Minimum content length per document
        required_keywords: Keywords that must appear in documents
        min_headers: Minimum number of headers per document
        required_sections: Required sections in documents
    """

    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    min_content_length: int
    required_keywords: list[str]
    min_headers: int
    required_sections: list[str]


# Task configurations
TASKS: dict[str, TaskConfig] = {
    "easy_user_authentication": TaskConfig(
        task_id="easy_user_authentication",
        name="Simple User Authentication",
        difficulty="easy",
        description="Build a simple user authentication system with login and registration functionality.",
        min_content_length=100,
        required_keywords=["authentication", "login", "password", "user"],
        min_headers=1,
        required_sections=["overview", "implementation"],
    ),
    # Medium and hard tasks will be added later
}


# ============================================================================
# Required Documents per Task
# ============================================================================

REQUIRED_DOCUMENTS: list[str] = [
    "RESEARCH",
    "PRD",
    "TRD",
    "ARCHITECTURE",
    "ROADMAP",
    "TASKS",
    "VALIDATION",
    "STRATEGY",
]


# ============================================================================
# Agent-Specific Document Mapping
# ============================================================================

AGENT_DOCUMENTS: dict[str, list[str]] = {
    "maya": ["RESEARCH"],
    "elon": ["PRD"],
    "jordan": ["TRD", "ARCHITECTURE"],
    "robert": ["ROADMAP", "TASKS"],
    "taylor": ["VALIDATION"],
    "sam": ["STRATEGY"],
}


# ============================================================================
# Grader Functions
# ============================================================================


def grade_task(task_id: str, documents: dict[str, Document]) -> float:
    """
    Grade the final output for a task.

    The grader evaluates:
    1. Completeness (30%): Are all required documents present?
    2. Content Quality (30%): Do documents have sufficient content?
    3. Realism (40%): Are documents well-structured and relevant?

    Args:
        task_id: The task being graded
        documents: All documents produced by the agents

    Returns:
        Score in range 0.0 to 1.0
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    task = TASKS[task_id]

    # Calculate individual scores
    completeness = _calculate_completeness(documents)
    quality = _calculate_content_quality(documents, task.min_content_length)
    realism = _calculate_realism(documents, task)

    # Weighted sum
    final_score = (completeness * 0.3) + (quality * 0.3) + (realism * 0.4)

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, final_score))


def _calculate_completeness(documents: dict[str, Document]) -> float:
    """
    Calculate completeness score based on required documents.

    Args:
        documents: All documents produced

    Returns:
        Completeness score in range 0.0 to 1.0
    """
    if not REQUIRED_DOCUMENTS:
        return 1.0

    present = sum(1 for doc_type in REQUIRED_DOCUMENTS if doc_type in documents)
    return present / len(REQUIRED_DOCUMENTS)


def _calculate_content_quality(documents: dict[str, Document], min_length: int) -> float:
    """
    Calculate content quality score based on minimum content length.

    Args:
        documents: All documents produced
        min_length: Minimum content length per document

    Returns:
        Content quality score in range 0.0 to 1.0
    """
    if not documents:
        return 0.0

    quality_count = sum(
        1 for doc in documents.values() if len(doc.content) >= min_length
    )
    return quality_count / len(documents)


def _calculate_realism(documents: dict[str, Document], task: TaskConfig) -> float:
    """
    Calculate realism score based on structure and relevance.

    Args:
        documents: All documents produced
        task: Task configuration

    Returns:
        Realism score in range 0.0 to 1.0
    """
    if not documents:
        return 0.0

    structure = _calculate_structure_score(documents, task.min_headers)
    relevance = _calculate_relevance_score(documents, task.required_keywords)

    return (structure + relevance) / 2


def _calculate_structure_score(documents: dict[str, Document], min_headers: int) -> float:
    """
    Calculate structure score based on headers.

    Args:
        documents: All documents produced
        min_headers: Minimum number of headers per document

    Returns:
        Structure score in range 0.0 to 1.0
    """
    if not documents:
        return 0.0

    structure_count = sum(
        1 for doc in documents.values() if doc.content.count("##") >= min_headers
    )
    return structure_count / len(documents)


def _calculate_relevance_score(documents: dict[str, Document], keywords: list[str]) -> float:
    """
    Calculate relevance score based on required keywords.

    Args:
        documents: All documents produced
        keywords: Required keywords

    Returns:
        Relevance score in range 0.0 to 1.0
    """
    if not documents or not keywords:
        return 0.0

    relevance_count = sum(
        1
        for doc in documents.values()
        if any(keyword.lower() in doc.content.lower() for keyword in keywords)
    )
    return relevance_count / len(documents)


# ============================================================================
# Agent-Specific Grading
# ============================================================================


def grade_agent_work(agent_id: str, documents: dict[str, Document], task: TaskConfig) -> float:
    """
    Grade the work of a specific agent.

    Args:
        agent_id: The agent ID to grade
        documents: All documents produced
        task: Task configuration

    Returns:
        Score in range 0.0 to 1.0 for this agent's work
    """
    if agent_id not in AGENT_DOCUMENTS:
        return 0.0

    required_docs = AGENT_DOCUMENTS[agent_id]
    agent_docs = {doc_type: documents[doc_type] for doc_type in required_docs if doc_type in documents}

    if not agent_docs:
        return 0.0

    # Check if all required documents are present
    completeness = len(agent_docs) / len(required_docs)

    # Check content quality
    quality = _calculate_content_quality(agent_docs, task.min_content_length)

    # Check relevance
    relevance = _calculate_relevance_score(agent_docs, task.required_keywords)

    # Average the scores
    return (completeness + quality + relevance) / 3


def get_agent_checklist(agent_id: str) -> list[str]:
    """
    Get the checklist for a specific agent.

    Args:
        agent_id: The agent ID

    Returns:
        List of checklist items for this agent
    """
    checklists: dict[str, list[str]] = {
        "maya": [
            "Research authentication patterns and best practices",
            "Analyze competitors' authentication systems",
            "Identify security considerations",
            "Summarize key insights",
            "Produce RESEARCH document",
        ],
        "elon": [
            "Define authentication requirements",
            "Identify user personas",
            "Define success metrics",
            "Prioritize features",
            "Produce PRD document",
        ],
        "jordan": [
            "Design authentication architecture",
            "Select technology stack",
            "Define APIs and data model",
            "Write TRD document",
            "Produce ARCHITECTURE document",
        ],
        "robert": [
            "Create implementation roadmap",
            "Break down into tasks",
            "Plan sprints",
            "Estimate timelines",
            "Produce ROADMAP and TASKS documents",
        ],
        "taylor": [
            "Review all documents",
            "Check consistency",
            "Validate claims",
            "Identify risks",
            "Produce VALIDATION document",
        ],
        "sam": [
            "Set strategic direction",
            "Review the complete plan",
            "Approve the strategy",
            "Prioritize objectives",
            "Produce STRATEGY document",
        ],
    }

    return checklists.get(agent_id, [])


# ============================================================================
# Task Utility Functions
# ============================================================================


def get_task(task_id: str) -> TaskConfig | None:
    """
    Get task configuration by ID.

    Args:
        task_id: The task ID

    Returns:
        TaskConfig or None if not found
    """
    return TASKS.get(task_id)


def get_all_tasks() -> dict[str, TaskConfig]:
    """
    Get all task configurations.

    Returns:
        Dictionary of all tasks
    """
    return TASKS.copy()


def get_tasks_by_difficulty(difficulty: Literal["easy", "medium", "hard"]) -> list[TaskConfig]:
    """
    Get tasks by difficulty level.

    Args:
        difficulty: The difficulty level

    Returns:
        List of tasks with the specified difficulty
    """
    return [task for task in TASKS.values() if task.difficulty == difficulty]


def get_task_summary(task_id: str) -> str:
    """
    Get a human-readable summary of a task.

    Args:
        task_id: The task ID

    Returns:
        Formatted task summary
    """
    task = get_task(task_id)
    if not task:
        return f"Unknown task: {task_id}"

    lines = [
        f"Task: {task.name}",
        f"Difficulty: {task.difficulty}",
        f"Description: {task.description}",
        f"Min Content Length: {task.min_content_length}",
        f"Required Keywords: {', '.join(task.required_keywords)}",
        f"Min Headers: {task.min_headers}",
        f"Required Sections: {', '.join(task.required_sections)}",
    ]
    return "\n".join(lines)
