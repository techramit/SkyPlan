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
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from .content_utils import (
    count_markdown_headers,
    extract_phase_labels,
    keyword_coverage_ratio,
)
from .models import Document


SCORE_EPSILON = 0.01


def _to_open_unit_interval(value: float) -> float:
    """Clamp scores to the strict open interval (0, 1)."""

    if value != value:  # NaN guard
        return SCORE_EPSILON
    if value <= 0.0:
        return SCORE_EPSILON
    if value >= 1.0:
        return 1.0 - SCORE_EPSILON
    return value


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
    "hard_saas_platform": TaskConfig(
        task_id="hard_saas_platform",
        name="Multi-tenant SaaS Platform",
        difficulty="hard",
        description="Build a multi-tenant SaaS platform with data isolation, subscription management, billing, analytics, and white-labeling capabilities.",
        min_content_length=1000,
        required_keywords=[
            "saas",
            "multi-tenant",
            "data-isolation",
            "subscription",
            "billing",
            "analytics",
            "white-label",
            "scalability",
        ],
        min_headers=5,
        required_sections=[
            "overview",
            "architecture",
            "data-model",
            "security",
            "implementation",
            "scaling",
        ],
    ),
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
# Base Grader Class
# ============================================================================


class BaseGrader(ABC):
    """Base class for grading tasks with common utility methods."""

    @staticmethod
    def check_document_length(documents: dict[str, Document], min_length: int) -> float:
        """Check if documents meet minimum length requirements.

        Args:
            documents: All documents produced
            min_length: Minimum content length per document

        Returns:
            Score in range 0.0 to 1.0
        """
        if not documents:
            return 0.0

        quality_count = sum(
            1 for doc in documents.values() if len(doc.content) >= min_length
        )
        return quality_count / len(documents)

    @staticmethod
    def check_document_structure(documents: dict[str, Document], min_headers: int) -> float:
        """Check if documents have proper structure (headers).

        Args:
            documents: All documents produced
            min_headers: Minimum number of headers per document

        Returns:
            Score in range 0.0 to 1.0
        """
        if not documents:
            return 0.0

        structure_count = sum(
            1
            for doc in documents.values()
            if count_markdown_headers(doc.content) >= min_headers
        )
        return structure_count / len(documents)

    @staticmethod
    def check_keyword_relevance(documents: dict[str, Document], keywords: list[str]) -> float:
        """Check if documents contain required keywords.

        Args:
            documents: All documents produced
            keywords: Required keywords

        Returns:
            Score in range 0.0 to 1.0
        """
        if not documents or not keywords:
            return 0.0

        coverage_scores = [
            keyword_coverage_ratio(doc.content, keywords)
            for doc in documents.values()
        ]
        return sum(coverage_scores) / len(coverage_scores)

    @staticmethod
    def check_completeness(documents: dict[str, Document], required_docs: list[str]) -> float:
        """Check if all required documents are present.

        Args:
            documents: All documents produced
            required_docs: List of required document types

        Returns:
            Score in range 0.0 to 1.0
        """
        if not required_docs:
            return 1.0

        present = sum(1 for doc_type in required_docs if doc_type in documents)
        return present / len(required_docs)

    @staticmethod
    def check_section_presence(documents: dict[str, Document], required_sections: list[str]) -> float:
        """Check if documents contain required sections.

        Args:
            documents: All documents produced
            required_sections: List of required section names

        Returns:
            Score in range 0.0 to 1.0
        """
        if not documents or not required_sections:
            return 0.0

        section_count = 0
        total_checks = len(documents) * len(required_sections)

        for doc in documents.values():
            for section in required_sections:
                if section.lower() in doc.content.lower():
                    section_count += 1

        return section_count / total_checks if total_checks > 0 else 0.0


# ============================================================================
# Grade Map for Composite Scoring
# ============================================================================


GRADE_MAP: dict[str, dict] = {
    "consistency_checks": {
        "prd_vs_trd": {
            "name": "PRD vs TRD Consistency",
            "description": "Check if PRD features align with TRD technical requirements",
            "documents": ["PRD", "TRD"],
            "check_function": "_check_prd_trd_consistency",
        },
        "architecture_vs_roadmap": {
            "name": "Architecture vs Roadmap Consistency",
            "description": "Check if architecture complexity matches roadmap timelines",
            "documents": ["ARCHITECTURE", "ROADMAP"],
            "check_function": "_check_architecture_roadmap_consistency",
        },
        "tasks_vs_roadmap": {
            "name": "Tasks vs Roadmap Consistency",
            "description": "Check if tasks align with roadmap phases",
            "documents": ["TASKS", "ROADMAP"],
            "check_function": "_check_tasks_roadmap_consistency",
        },
        "research_vs_prd": {
            "name": "Research vs PRD Consistency",
            "description": "Check if PRD addresses research findings",
            "documents": ["RESEARCH", "PRD"],
            "check_function": "_check_research_prd_consistency",
        },
        "validation_vs_all": {
            "name": "Validation vs All Documents",
            "description": "Check if validation addresses all documents",
            "documents": ["VALIDATION"],
            "check_function": "_check_validation_completeness",
        },
    },
    "agent_criteria": {
        "maya": {
            "required_sections": ["market-analysis", "competitors", "opportunities"],
            "min_findings": 3,
        },
        "elon": {
            "required_sections": ["requirements", "features", "user-personas", "success-metrics"],
            "min_features": 5,
        },
        "jordan": {
            "required_sections": ["architecture", "tech-stack", "apis", "data-model"],
            "min_components": 4,
        },
        "robert": {
            "required_sections": ["roadmap", "phases", "tasks", "timelines"],
            "min_phases": 3,
        },
        "taylor": {
            "required_sections": ["validation", "consistency", "risks", "recommendations"],
            "min_checks": 3,
        },
        "sam": {
            "required_sections": ["strategy", "priorities", "approval", "next-steps"],
            "min_objectives": 3,
        },
    },
}


# ============================================================================
# Composite Scoring Functions
# ============================================================================


def _check_prd_trd_consistency(documents: dict[str, Document]) -> float:
    """Check if PRD features align with TRD technical requirements.

    Args:
        documents: All documents produced

    Returns:
        Consistency score in range 0.0 to 1.0
    """
    if "PRD" not in documents or "TRD" not in documents:
        return 0.0

    prd = documents["PRD"].content.lower()
    trd = documents["TRD"].content.lower()

    # Check for common feature mentions
    prd_features = ["feature", "requirement", "functionality", "capability"]
    trd_features = ["implement", "api", "endpoint", "service", "component"]

    prd_has_features = sum(1 for f in prd_features if f in prd)
    trd_has_features = sum(1 for f in trd_features if f in trd)

    # Check if PRD mentions are addressed in TRD
    consistency = min(prd_has_features, trd_has_features) / max(prd_has_features, trd_has_features, 1)

    return consistency


def _check_architecture_roadmap_consistency(documents: dict[str, Document]) -> float:
    """Check if architecture complexity matches roadmap timelines.

    Args:
        documents: All documents produced

    Returns:
        Consistency score in range 0.0 to 1.0
    """
    if "ARCHITECTURE" not in documents or "ROADMAP" not in documents:
        return 0.0

    arch = documents["ARCHITECTURE"].content.lower()
    roadmap = documents["ROADMAP"].content.lower()

    # Check for complexity indicators
    arch_complexity = ["microservice", "component", "module", "service", "layer"]
    roadmap_phases = ["phase", "milestone", "sprint", "quarter", "month"]

    arch_score = sum(1 for c in arch_complexity if c in arch)
    roadmap_score = sum(1 for r in roadmap_phases if r in roadmap)

    # More complex architecture should have more phases
    if arch_score > 3 and roadmap_score < 2:
        return 0.5  # Complex architecture needs more phases
    if arch_score <= 2 and roadmap_score > 4:
        return 0.5  # Simple architecture shouldn't have too many phases

    return min(arch_score, roadmap_score) / max(arch_score, roadmap_score, 1)


def _check_tasks_vs_roadmap_consistency(documents: dict[str, Document]) -> float:
    """Check if tasks align with roadmap phases.

    Args:
        documents: All documents produced

    Returns:
        Consistency score in range 0.0 to 1.0
    """
    if "TASKS" not in documents or "ROADMAP" not in documents:
        return 0.0

    tasks = documents["TASKS"].content.lower()
    roadmap = documents["ROADMAP"].content.lower()

    # Check for task and phase mentions
    task_mentions = ["task", "story", "ticket", "item"]
    phase_mentions = ["phase", "milestone", "sprint", "iteration"]

    task_score = sum(1 for t in task_mentions if t in tasks)
    phase_score = sum(1 for p in phase_mentions if p in roadmap)
    roadmap_labels = extract_phase_labels(roadmap)
    task_labels = extract_phase_labels(tasks)

    if roadmap_labels:
        return len(roadmap_labels & task_labels) / len(roadmap_labels)

    if task_score == 0 or phase_score == 0:
        return 0.0

    return min(task_score, phase_score) / max(task_score + phase_score, 1)


def _check_research_prd_consistency(documents: dict[str, Document]) -> float:
    """Check if PRD addresses research findings.

    Args:
        documents: All documents produced

    Returns:
        Consistency score in range 0.0 to 1.0
    """
    if "RESEARCH" not in documents or "PRD" not in documents:
        return 0.0

    research = documents["RESEARCH"].content.lower()
    prd = documents["PRD"].content.lower()

    # Check for research findings in PRD
    research_keywords = ["market", "competitor", "user", "need", "problem"]
    prd_mentions = sum(1 for kw in research_keywords if kw in prd)

    return prd_mentions / len(research_keywords)


def _check_validation_completeness(documents: dict[str, Document]) -> float:
    """Check if validation addresses all documents.

    Args:
        documents: All documents produced

    Returns:
        Completeness score in range 0.0 to 1.0
    """
    if "VALIDATION" not in documents:
        return 0.0

    validation = documents["VALIDATION"].content.lower()

    # Check if validation mentions other documents
    doc_mentions = sum(
        1
        for doc_type in REQUIRED_DOCUMENTS
        if doc_type.lower() in validation
    )

    return doc_mentions / len(REQUIRED_DOCUMENTS)


def calculate_composite_score(documents: dict[str, Document]) -> float:
    """Calculate composite score based on consistency checks.

    Args:
        documents: All documents produced

    Returns:
        Composite score in range 0.0 to 1.0
    """
    consistency_checks = GRADE_MAP["consistency_checks"]
    scores = []

    for check_id, check_config in consistency_checks.items():
        check_function = check_config["check_function"]
        if check_function in globals():
            score = globals()[check_function](documents)
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def calculate_agent_criteria_score(agent_id: str, documents: dict[str, Document]) -> float:
    """Calculate score based on agent-specific criteria.

    Args:
        agent_id: The agent ID
        documents: All documents produced

    Returns:
        Score in range 0.0 to 1.0
    """
    if agent_id not in GRADE_MAP["agent_criteria"]:
        return 0.0

    criteria = GRADE_MAP["agent_criteria"][agent_id]
    agent_docs = AGENT_DOCUMENTS.get(agent_id, [])

    if not agent_docs:
        return 0.0

    total_score = 0.0
    total_checks = 0

    for doc_type in agent_docs:
        if doc_type not in documents:
            continue

        doc = documents[doc_type]
        content = doc.content.lower()

        # Check for required sections
        for section in criteria["required_sections"]:
            total_checks += 1
            if section in content:
                total_score += 1.0

    return total_score / total_checks if total_checks > 0 else 0.0


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
    1. Completeness (25%): Are all required documents present?
    2. Content Quality (25%): Do documents have sufficient content?
    3. Realism (25%): Are documents well-structured and relevant?
    4. Composite Consistency (25%): Do documents align with each other? (hard task only)

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
    resolved_api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    # Use base grader for common checks
    grader = BaseGrader()

    # Calculate individual scores
    completeness = grader.check_completeness(documents, REQUIRED_DOCUMENTS)

    # Use LLM for quality assessment if API key is provided and use_llm is True
    if use_llm and resolved_api_key:
        llm_scores = _llm_grade_content(task_id, documents, resolved_api_key)
        quality = llm_scores["content_quality"]
        realism = llm_scores["realism"]
    else:
        # Fallback to rule-based grading
        quality = grader.check_document_length(documents, task.min_content_length)
        structure = grader.check_document_structure(documents, task.min_headers)
        relevance = grader.check_keyword_relevance(documents, task.required_keywords)
        realism = (structure + relevance) / 2

    # Composite consistency score (only for hard task)
    if task.difficulty == "hard":
        composite = calculate_composite_score(documents)
        # Weighted sum with composite scoring
        final_score = (
            completeness * 0.25 +
            quality * 0.25 +
            realism * 0.25 +
            composite * 0.25
        )
    else:
        # Weighted sum without composite scoring
        final_score = (completeness * 0.3) + (quality * 0.3) + (realism * 0.4)

    # Clamp to strict open interval (0, 1).
    return _to_open_unit_interval(final_score)


def _calculate_completeness(documents: dict[str, Document]) -> float:
    """
    Calculate completeness score based on required documents.

    Args:
        documents: All documents produced

    Returns:
        Completeness score in range 0.0 to 1.0
    """
    grader = BaseGrader()
    return grader.check_completeness(documents, REQUIRED_DOCUMENTS)


def _calculate_content_quality(documents: dict[str, Document], min_length: int) -> float:
    """
    Calculate content quality score based on minimum content length.

    Args:
        documents: All documents produced
        min_length: Minimum content length per document

    Returns:
        Content quality score in range 0.0 to 1.0
    """
    grader = BaseGrader()
    return grader.check_document_length(documents, min_length)


def _calculate_realism(documents: dict[str, Document], task: TaskConfig) -> float:
    """
    Calculate realism score based on structure and relevance.

    Args:
        documents: All documents produced
        task: Task configuration

    Returns:
        Realism score in range 0.0 to 1.0
    """
    grader = BaseGrader()
    structure = grader.check_document_structure(documents, task.min_headers)
    relevance = grader.check_keyword_relevance(documents, task.required_keywords)

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
    grader = BaseGrader()
    return grader.check_document_structure(documents, min_headers)


def _calculate_relevance_score(documents: dict[str, Document], keywords: list[str]) -> float:
    """
    Calculate relevance score based on required keywords.

    Args:
        documents: All documents produced
        keywords: Required keywords

    Returns:
        Relevance score in range 0.0 to 1.0
    """
    grader = BaseGrader()
    return grader.check_keyword_relevance(documents, keywords)


# ============================================================================
# LLM-Based Grading
# ============================================================================


def _llm_grade_content(
    task_id: str,
    documents: dict[str, Document],
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
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
    resolved_base_url = base_url or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    resolved_model = model or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    client = OpenAI(api_key=api_key, base_url=resolved_base_url)

    try:
        response = client.chat.completions.create(
            model=resolved_model,
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
    total_content = sum(len(doc.content) for doc in documents.values())

    # Adjust preview size based on total content to avoid memory issues
    preview_size = min(500, max(100, total_content // len(documents)))

    for doc_type, doc in documents.items():
        content_preview = doc.content[:preview_size] if len(doc.content) > preview_size else doc.content
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

    resolved_api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    # Use LLM for quality assessment if API key is provided and use_llm is True
    if use_llm and resolved_api_key:
        llm_scores = _llm_grade_content(task.task_id, agent_docs, resolved_api_key)
        quality = llm_scores["content_quality"]
        relevance = llm_scores["realism"]
    else:
        # Fallback to rule-based grading
        quality = _calculate_content_quality(agent_docs, task.min_content_length)
        relevance = _calculate_relevance_score(agent_docs, task.required_keywords)

    # Average the scores
    return _to_open_unit_interval((completeness + quality + relevance) / 3)


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
        "hard_saas_platform": {
            "maya": [
                "Research multi-tenant SaaS patterns and best practices",
                "Analyze competitors (Salesforce, HubSpot, Stripe)",
                "Identify data isolation strategies",
                "Research subscription and billing models",
                "Research white-labeling approaches",
                "Identify compliance requirements (GDPR, SOC2)",
                "Summarize key insights",
                "Produce RESEARCH document",
            ],
            "elon": [
                "Define SaaS platform requirements",
                "Identify user personas (admin, user, guest)",
                "Define subscription tiers and pricing",
                "Define success metrics (MRR, churn, LTV)",
                "Prioritize features (multi-tenancy, billing, analytics, white-label)",
                "Define white-labeling requirements",
                "Produce PRD document",
            ],
            "jordan": [
                "Design multi-tenant architecture",
                "Select technology stack (microservices, databases, caching)",
                "Define data isolation strategy (per-tenant databases, schemas)",
                "Design billing and subscription system",
                "Design analytics and reporting system",
                "Design white-labeling system",
                "Define APIs and data models",
                "Write TRD document",
                "Produce ARCHITECTURE document",
            ],
            "robert": [
                "Create implementation roadmap",
                "Break down into tasks and sprints",
                "Plan for scalability (horizontal scaling, sharding)",
                "Plan for data migration and onboarding",
                "Estimate timelines and dependencies",
                "Define testing strategy (integration, E2E, performance)",
                "Define deployment strategy (CI/CD, blue-green deployments)",
                "Produce ROADMAP and TASKS documents",
            ],
            "taylor": [
                "Review all documents for consistency",
                "Validate technical claims",
                "Identify scalability and performance risks",
                "Check for security and compliance considerations",
                "Validate data isolation guarantees",
                "Check billing and subscription logic",
                "Validate white-labeling implementation",
                "Produce VALIDATION document",
            ],
            "sam": [
                "Set strategic direction for SaaS platform",
                "Review the complete plan",
                "Approve the strategy and priorities",
                "Evaluate competitive positioning",
                "Prioritize objectives and milestones",
                "Review go-to-market strategy",
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
