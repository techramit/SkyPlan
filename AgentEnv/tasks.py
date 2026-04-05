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

import json
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

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
    "medium_chat_app": TaskConfig(
        task_id="medium_chat_app",
        name="Real-time Chat Application",
        difficulty="medium",
        description="Build a real-time chat application similar to WhatsApp with messaging, online status, and notifications.",
        min_content_length=500,
        required_keywords=[
            "chat",
            "real-time",
            "websocket",
            "message",
            "online",
            "notification",
        ],
        min_headers=3,
        required_sections=[
            "overview",
            "features",
            "architecture",
            "implementation",
        ],
    ),
    # Hard task will be added later
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


def grade_task(
    task_id: str,
    documents: dict[str, Document],
    api_key: str | None = None,
    use_llm: bool = True,
) -> float:
    """
    Grade the final output for a task.

    The grader evaluates:
    1. Completeness (30%): Are all required documents present?
    2. Content Quality (30%): Do documents have sufficient content?
    3. Realism (40%): Are documents well-structured and relevant?

    Args:
        task_id: The task being graded
        documents: All documents produced by the agents
        api_key: Nvidia NIM API key for LLM grading (optional)
        use_llm: Whether to use LLM for quality assessment (default: True)

    Returns:
        Score in range 0.0 to 1.0
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    task = TASKS[task_id]

    # Calculate individual scores
    completeness = _calculate_completeness(documents)

    # Use LLM for quality assessment if API key is provided and use_llm is True
    if use_llm and api_key:
        llm_scores = _llm_grade_content(task_id, documents, api_key)
        quality = llm_scores["content_quality"]
        realism = llm_scores["realism"]
    else:
        # Fallback to rule-based grading
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
# LLM-Based Grading
# ============================================================================


def _llm_grade_content(
    task_id: str,
    documents: dict[str, Document],
    api_key: str,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    model: str = "meta/llama-3.1-405b-instruct",
) -> dict[str, float]:
    """
    Use LLM to grade content quality and realism.

    Args:
        task_id: The task being graded
        documents: All documents produced
        api_key: Nvidia NIM API key
        base_url: Nvidia NIM API base URL
        model: Model to use for grading

    Returns:
        Dictionary with content_quality and realism scores (0.0-1.0)
    """
    task = TASKS[task_id]

    # Build documents summary
    docs_summary = _build_documents_summary(documents)

    # Build the prompt
    prompt = _build_grading_prompt(task, docs_summary)

    # Call the LLM
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator of technical planning documents. Grade objectively and consistently.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # Deterministic
            response_format={"type": "json_object"},
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)

        return {
            "content_quality": float(result.get("content_quality", 0.0)),
            "realism": float(result.get("realism", 0.0)),
        }
    except Exception as e:
        # Fallback to rule-based grading on error
        print(f"LLM grading failed: {e}. Falling back to rule-based grading.")
        return {
            "content_quality": _calculate_content_quality(documents, task.min_content_length),
            "realism": _calculate_realism(documents, task),
        }


def _build_documents_summary(documents: dict[str, Document]) -> str:
    """
    Build a summary of all documents for LLM grading.

    Args:
        documents: All documents produced

    Returns:
        Formatted summary string
    """
    if not documents:
        return "No documents produced."

    lines = ["Documents produced:"]
    for doc_type, doc in documents.items():
        content_preview = doc.content[:500] if len(doc.content) > 500 else doc.content
        lines.append(f"\n{doc_type}:")
        lines.append(f"  Author: {doc.author}")
        lines.append(f"  Status: {doc.status}")
        lines.append(f"  Content: {content_preview}...")
        lines.append(f"  Length: {len(doc.content)} characters")

    return "\n".join(lines)


def _build_grading_prompt(task: TaskConfig, docs_summary: str) -> str:
    """
    Build the grading prompt for the LLM.

    Args:
        task: Task configuration
        docs_summary: Summary of documents

    Returns:
        Formatted prompt string
    """
    return f"""Grade the following planning documents on a scale of 0.0 to 1.0.

Task: {task.name}
Difficulty: {task.difficulty}
Description: {task.description}

Required Keywords: {', '.join(task.required_keywords)}
Minimum Content Length: {task.min_content_length} characters
Minimum Headers: {task.min_headers}

{docs_summary}

Evaluate on:

1. Content Quality (0.0-1.0):
   - Is the content substantive and detailed?
   - Is the reasoning sound and well-justified?
   - Are the claims supported with evidence?
   - Is the writing clear and professional?

2. Realism (0.0-1.0):
   - Is the technical plan realistic and feasible?
   - Are the timelines reasonable?
   - Did they consider edge cases and risks?
   - Is the architecture sound and appropriate?

Return a JSON object with:
{{
    "content_quality": 0.0-1.0,
    "realism": 0.0-1.0
}}

Be objective and consistent in your grading."""


# ============================================================================
# Agent-Specific Grading
# ============================================================================


def grade_agent_work(
    agent_id: str,
    documents: dict[str, Document],
    task: TaskConfig,
    api_key: str | None = None,
    use_llm: bool = True,
) -> float:
    """
    Grade the work of a specific agent.

    Args:
        agent_id: The agent ID to grade
        documents: All documents produced
        task: Task configuration
        api_key: Nvidia NIM API key for LLM grading (optional)
        use_llm: Whether to use LLM for quality assessment (default: True)

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

    # Use LLM for quality assessment if API key is provided and use_llm is True
    if use_llm and api_key:
        llm_scores = _llm_grade_content(task.task_id, agent_docs, api_key)
        quality = llm_scores["content_quality"]
        relevance = llm_scores["realism"]
    else:
        # Fallback to rule-based grading
        quality = _calculate_content_quality(agent_docs, task.min_content_length)
        relevance = _calculate_relevance_score(agent_docs, task.required_keywords)

    # Average the scores
    return (completeness + quality + relevance) / 3


def get_agent_checklist(agent_id: str, task_id: str | None = None) -> list[str]:
    """
    Get the checklist for a specific agent.

    Args:
        agent_id: The agent ID
        task_id: The task ID (optional, defaults to easy task if not provided)

    Returns:
        List of checklist items for this agent
    """
    # Default to easy task if no task_id provided
    if task_id is None:
        task_id = "easy_user_authentication"

    # Task-specific checklists
    task_checklists: dict[str, dict[str, list[str]]] = {
        "easy_user_authentication": {
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
        },
        "medium_chat_app": {
            "maya": [
                "Research real-time chat patterns and best practices",
                "Analyze competitors (WhatsApp, Slack, Discord)",
                "Identify WebSocket and messaging protocols",
                "Research online status and notification systems",
                "Summarize key insights",
                "Produce RESEARCH document",
            ],
            "elon": [
                "Define chat application requirements",
                "Identify user personas and use cases",
                "Define success metrics (latency, engagement)",
                "Prioritize features (messaging, status, notifications)",
                "Define user experience goals",
                "Produce PRD document",
            ],
            "jordan": [
                "Design real-time chat architecture",
                "Select technology stack (WebSocket, database, caching)",
                "Define APIs and data models",
                "Design message ordering and delivery system",
                "Design online status and notification system",
                "Write TRD document",
                "Produce ARCHITECTURE document",
            ],
            "robert": [
                "Create implementation roadmap",
                "Break down into tasks and sprints",
                "Plan for scalability and performance",
                "Estimate timelines and dependencies",
                "Define testing strategy",
                "Produce ROADMAP and TASKS documents",
            ],
            "taylor": [
                "Review all documents for consistency",
                "Validate technical claims",
                "Identify scalability and performance risks",
                "Check for security considerations",
                "Validate message ordering guarantees",
                "Produce VALIDATION document",
            ],
            "sam": [
                "Set strategic direction for chat platform",
                "Review the complete plan",
                "Approve the strategy and priorities",
                "Evaluate competitive positioning",
                "Prioritize objectives and milestones",
                "Produce STRATEGY document",
            ],
        },
    }

    # Get task-specific checklists, or empty list if not found
    task_checklist = task_checklists.get(task_id, {})
    return task_checklist.get(agent_id, [])


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
