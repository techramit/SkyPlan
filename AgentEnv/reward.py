# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SkyPlan Reward System - "The Paycheck"

A comprehensive reward system that evaluates agent work across multiple dimensions:
- Quality Bonus: Document quality assessment
- Teamwork Bonus: Collaboration and reference detection
- Completion Bonus: All-or-nothing completion reward
- Constraint Penalty: Penalties for violations
- Final Score: Normalized score in [0.0, 1.0]
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from os import environ
from typing import Protocol, runtime_checkable

from openai import OpenAI

from .models import (
    Document,
    DocumentStatus,
    DocumentStatusConfig,
    SkyPlanAction,
)
from .workflow import (
    get_all_agent_ids,
    get_all_document_types,
    get_required_documents,
    get_workflow_entry,
)


# ============================================================================
# Constants
# ============================================================================

# Default number of steps in a workflow (6 agents)
DEFAULT_WORKFLOW_STEPS = 6

# Score thresholds
SCORE_THRESHOLD_EXCELLENT = 0.8
SCORE_THRESHOLD_GOOD = 0.5
SCORE_THRESHOLD_POOR = 0.3

# Feedback thresholds
FEEDBACK_THRESHOLD = 0.5
HANDOFF_QUALITY_THRESHOLD = 0.8

# Deduction values
UNPROFESSIONAL_PATTERN_DEDUCTION = 0.1
SENTENCE_STRUCTURE_BONUS = 0.1
HANDOFF_ACKNOWLEDGMENT_BONUS = 0.1
HANDOFF_SETUP_BONUS = 0.1
CONTRADICTION_PENALTY_MULTIPLIER = 0.5


# ============================================================================
# Enums
# ============================================================================


class DifficultyLevel(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReferenceLevel(str, Enum):
    """Reference quality levels."""

    NONE = "none"
    GENERIC = "generic"
    SPECIFIC = "specific"
    INTEGRATED = "integrated"


class PenaltyType(str, Enum):
    """Types of penalties."""

    LENGTH = "length"
    STRUCTURE = "structure"
    CONTENT = "content"
    ROLE = "role"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RewardConfig:
    """Configuration for reward calculation weights and thresholds.

    All values are configurable via environment variables or constructor arguments.
    """

    # Quality Bonus (0.0-0.3 per document)
    QUALITY_BONUS_MAX: float = 0.3
    QUALITY_CONTENT_DEPTH_WEIGHT: float = 0.4
    QUALITY_STRUCTURE_WEIGHT: float = 0.3
    QUALITY_RELEVANCE_WEIGHT: float = 0.2
    QUALITY_PROFESSIONALISM_WEIGHT: float = 0.1

    # Content Depth Thresholds per Difficulty
    CONTENT_DEPTH_THRESHOLDS: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            DifficultyLevel.EASY: (100, 300),
            DifficultyLevel.MEDIUM: (300, 800),
            DifficultyLevel.HARD: (800, 2000),
        }
    )

    # Structure Scoring Weights
    STRUCTURE_HAS_HEADERS_WEIGHT: float = 0.3
    STRUCTURE_HAS_LISTS_WEIGHT: float = 0.3
    STRUCTURE_HAS_PARAGRAPHS_WEIGHT: float = 0.2
    STRUCTURE_HAS_KEYWORDS_WEIGHT: float = 0.2
    STRUCTURE_MIN_PARAGRAPHS: int = 2

    # Structural Keywords
    STRUCTURAL_KEYWORDS: list[str] = field(
        default_factory=lambda: [
            "overview", "summary", "introduction", "conclusion",
            "goal", "objective", "requirement", "feature",
        ]
    )

    # Unprofessional Patterns
    UNPROFESSIONAL_PATTERNS: list[str] = field(
        default_factory=lambda: [
            "lol", "haha", "omg", "btw", "idk", "tbh",
            "!!!", "???", "??", "!!",
        ]
    )

    # Teamwork Bonus (0.0-0.2 per step)
    TEAMWORK_BONUS_MAX: float = 0.2
    TEAMWORK_REFERENCE_LEVELS: dict[str, float] = field(
        default_factory=lambda: {
            ReferenceLevel.NONE: 0.0,
            ReferenceLevel.GENERIC: 0.05,
            ReferenceLevel.SPECIFIC: 0.1,
            ReferenceLevel.INTEGRATED: 0.2,
        }
    )
    TEAMWORK_REFERENCE_THRESHOLD_LOW: float = 0.3
    TEAMWORK_REFERENCE_THRESHOLD_HIGH: float = 0.7

    # Entity Extraction
    ENTITY_MIN_WORD_LENGTH: int = 3

    # Handoff Phrases
    ACKNOWLEDGMENT_PHRASES: list[str] = field(
        default_factory=lambda: [
            "based on", "building on", "following", "as discussed",
            "as outlined", "per the", "according to",
        ]
    )
    SETUP_PHRASES: list[str] = field(
        default_factory=lambda: [
            "next step", "for the", "will be", "should",
            "recommend", "suggest",
        ]
    )

    # Completion Bonus (all-or-nothing)
    COMPLETION_BONUS: float = 0.3

    # Penalties
    PENALTY_TOO_SHORT: float = -0.1
    PENALTY_EMPTY: float = -0.2
    PENALTY_COPY_PASTE: float = -0.15
    PENALTY_NO_HEADERS: float = -0.05
    PENALTY_NO_LISTS: float = -0.05
    PENALTY_POOR_FORMATTING: float = -0.05
    PENALTY_MISSING_SECTION: float = -0.1
    PENALTY_MISSING_KEYWORD: float = -0.05
    PENALTY_CONTRADICTION: float = -0.1
    PENALTY_UNREALISTIC: float = -0.1
    PENALTY_WRONG_ROLE: float = -0.15
    PENALTY_IGNORE_ROLE: float = -0.1
    PENALTY_MAX_PER_STEP: float = -0.3

    # Approval rewards for status transition actions
    APPROVAL_BONUS_MAX: float = 0.2  # Maximum bonus per approval action
    APPROVAL_MARK_REVIEW_BONUS: float = 0.05  # Bonus for marking documents for review
    APPROVAL_APPROVE_BONUS: float = 0.15  # Bonus for approving documents
    APPROVAL_REJECT_BONUS: float = 0.02  # Small bonus for rejecting (providing feedback)
    APPROVAL_ALL_BONUS: float = 0.2  # Bonus for Sam approving all documents
    APPROVAL_FINAL_BONUS: float = 0.2  # Bonus for final CEO approval

    # Contradiction Patterns
    CONTRADICTION_PHRASES: list[str] = field(
        default_factory=lambda: [
            "however", "but", "although", "despite",
            "contrary to", "opposite of", "not consistent",
        ]
    )

    # Unrealistic Timeline Patterns
    UNREALISTIC_PATTERNS: list[str] = field(
        default_factory=lambda: [
            "in 1 day", "in one day", "tomorrow",
            "in 1 hour", "in one hour",
            "instant", "immediately", "right now",
        ]
    )

    # Minimum thresholds
    MIN_CONTENT_LENGTH: int = 50
    MIN_REASONING_LENGTH: int = 20
    MIN_HEADERS: int = 1

    # LLM Configuration
    LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL: str = "meta/llama-3.1-405b-instruct"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT: int = 10
    LLM_CONTENT_PREVIEW_LENGTH: int = 2000

    # Caching
    CACHE_TTL_HOURS: int = 24

    # Workflow configuration
    WORKFLOW_STEPS: int = DEFAULT_WORKFLOW_STEPS

    # Feedback Generation Rewards
    FEEDBACK_GENERATION_BONUS: bool = True
    COLLABORATIVE_FEEDBACK_BONUS: float = 0.02
    VALIDATOR_FEEDBACK_BONUS: float = 0.08
    STRATEGIC_FEEDBACK_BONUS: float = 0.10

    # Feedback Resolution Rewards
    FEEDBACK_RESOLUTION_BONUS: bool = True
    PRIMARY_FEEDBACK_RESOLUTION_BONUS: float = 0.15
    PEER_FEEDBACK_RESOLUTION_BONUS: float = 0.05

    # Document Approval Rewards
    DOCUMENT_APPROVAL_BONUS: bool = True
    TAYLOR_APPROVAL_BONUS: float = 0.15
    SAM_APPROVAL_BONUS: float = 0.15
    FINAL_APPROVAL_BONUS: float = 0.50

    # Normalizer bounds (calculated from config)
    NORMALIZER_MIN_POSSIBLE: float = field(init=False)
    NORMALIZER_MAX_POSSIBLE: float = field(init=False)

    def __post_init__(self):
        """Calculate derived values after initialization."""
        # Calculate normalizer bounds based on config and workflow steps
        # Worst case: all penalties, no bonuses
        self.NORMALIZER_MIN_POSSIBLE = (
            self.PENALTY_MAX_PER_STEP * self.WORKFLOW_STEPS
        )
        # Best case: all bonuses, no penalties
        self.NORMALIZER_MAX_POSSIBLE = (
            self.QUALITY_BONUS_MAX * self.WORKFLOW_STEPS
            + self.TEAMWORK_BONUS_MAX * self.WORKFLOW_STEPS
            + self.COMPLETION_BONUS
        )

    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Create configuration from environment variables.

        Environment variables:
            SKYPLAN_QUALITY_BONUS_MAX: Maximum quality bonus
            SKYPLAN_TEAMWORK_BONUS_MAX: Maximum teamwork bonus
            SKYPLAN_COMPLETION_BONUS: Completion bonus
            SKYPLAN_LLM_API_KEY: LLM API key
            SKYPLAN_LLM_BASE_URL: LLM base URL
            SKYPLAN_LLM_MODEL: LLM model name
            SKYPLAN_CACHE_TTL_HOURS: Cache TTL in hours
            SKYPLAN_WORKFLOW_STEPS: Number of workflow steps
        """
        config = cls()

        # Override with environment variables if set
        if "SKYPLAN_QUALITY_BONUS_MAX" in environ:
            config.QUALITY_BONUS_MAX = float(environ["SKYPLAN_QUALITY_BONUS_MAX"])
        if "SKYPLAN_TEAMWORK_BONUS_MAX" in environ:
            config.TEAMWORK_BONUS_MAX = float(environ["SKYPLAN_TEAMWORK_BONUS_MAX"])
        if "SKYPLAN_COMPLETION_BONUS" in environ:
            config.COMPLETION_BONUS = float(environ["SKYPLAN_COMPLETION_BONUS"])
        if "SKYPLAN_LLM_BASE_URL" in environ:
            config.LLM_BASE_URL = environ["SKYPLAN_LLM_BASE_URL"]
        if "SKYPLAN_LLM_MODEL" in environ:
            config.LLM_MODEL = environ["SKYPLAN_LLM_MODEL"]
        if "SKYPLAN_LLM_TIMEOUT" in environ:
            config.LLM_TIMEOUT = int(environ["SKYPLAN_LLM_TIMEOUT"])
        if "SKYPLAN_CACHE_TTL_HOURS" in environ:
            config.CACHE_TTL_HOURS = int(environ["SKYPLAN_CACHE_TTL_HOURS"])
        if "SKYPLAN_WORKFLOW_STEPS" in environ:
            config.WORKFLOW_STEPS = int(environ["SKYPLAN_WORKFLOW_STEPS"])

        # Re-calculate derived values
        config.__post_init__()

        return config


# Global configuration instance
reward_config = RewardConfig.from_env()

# Configure logger (module-level, not global)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class QualityScore:
    """Result of quality assessment."""

    overall: float
    content_depth: float
    structure: float
    relevance: float
    professionalism: float
    feedback: list[str] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class TeamworkScore:
    """Result of teamwork assessment."""

    overall: float
    references: dict[str, float] = field(default_factory=dict)
    handoff_quality: float = 0.0
    feedback: list[str] = field(default_factory=list)


@dataclass
class PenaltyScore:
    """Result of penalty assessment."""

    total: float
    penalties: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class StepReward:
    """Reward for a single step."""

    quality_bonus: float
    teamwork_bonus: float
    penalty: float
    total: float
    quality_score: QualityScore | None = None
    teamwork_score: TeamworkScore | None = None
    penalty_score: PenaltyScore | None = None


@dataclass
class EpisodeReward:
    """Final reward for an episode."""

    step_rewards: list[StepReward] = field(default_factory=list)
    completion_bonus: float = 0.0
    total_raw: float = 0.0
    final_score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class ScoreCalculator(Protocol):
    """Protocol for score calculators."""

    def calculate(self, *args, **kwargs) -> float:
        """Calculate a score."""
        ...


# ============================================================================
# Shared Utilities
# ============================================================================


class ContentAnalyzer:
    """Shared utilities for content analysis."""

    @staticmethod
    def has_headers(content: str) -> bool:
        """Check if content has markdown headers.

        Args:
            content: Document content

        Returns:
            True if headers present
        """
        return "##" in content

    @staticmethod
    def has_lists(content: str) -> bool:
        """Check if content has markdown lists.

        Args:
            content: Document content

        Returns:
            True if lists present
        """
        return "-" in content or "*" in content

    @staticmethod
    def count_paragraphs(content: str) -> int:
        """Count number of paragraphs in content.

        Args:
            content: Document content

        Returns:
            Number of paragraphs
        """
        return content.count("\n\n")

    @staticmethod
    def has_keyword(content: str, keyword: str, case_sensitive: bool = False) -> bool:
        """Check if content contains a keyword.

        Args:
            content: Document content
            keyword: Keyword to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            True if keyword found
        """
        if case_sensitive:
            return keyword in content
        return keyword.lower() in content.lower()

    @staticmethod
    def extract_words(content: str, min_length: int = 1) -> list[str]:
        """Extract words from content.

        Args:
            content: Document content
            min_length: Minimum word length

        Returns:
            List of words
        """
        words = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for word in line.split():
                word = word.strip(".,;:!?()[]{}\"'").strip()
                if len(word) >= min_length:
                    words.append(word)
        return words


# ============================================================================
# Cache
# ============================================================================


class RewardCache:
    """Cache for reward calculations to avoid redundant LLM calls."""

    def __init__(self, ttl_hours: int = 24):
        self._cache: dict[str, tuple[QualityScore, datetime]] = {}
        self._ttl = timedelta(hours=ttl_hours)

    def _hash_content(self, content: str) -> str:
        """Create a hash of content for caching.

        Args:
            content: Content to hash

        Returns:
            SHA256 hash
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, content: str) -> QualityScore | None:
        """Get cached score if available and not expired.

        Args:
            content: Document content to look up

        Returns:
            Cached QualityScore or None
        """
        key = self._hash_content(content)
        if key in self._cache:
            score, timestamp = self._cache[key]
            if datetime.utcnow() - timestamp < self._ttl:
                return score
            # Expired, remove
            del self._cache[key]
        return None

    def set(self, content: str, score: QualityScore) -> None:
        """Cache a quality score.

        Args:
            content: Document content
            score: QualityScore to cache
        """
        key = self._hash_content(content)
        self._cache[key] = (score, datetime.utcnow())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cached entries.

        Returns:
            Number of cached entries
        """
        return len(self._cache)


# Global cache instance
_reward_cache = RewardCache(ttl_hours=reward_config.CACHE_TTL_HOURS)


# ============================================================================
# Base Calculator
# ============================================================================


class BaseCalculator(ABC):
    """Base class for reward calculators."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or reward_config

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """Calculate the reward component.

        Returns:
            Calculated score
        """
        ...


# ============================================================================
# Quality Bonus Calculator
# ============================================================================


class QualityBonusCalculator(BaseCalculator):
    """Calculates quality bonus for document assessment."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        use_llm: bool = True,
        api_key: str | None = None,
    ):
        super().__init__(config)
        self.use_llm = use_llm
        self.api_key = api_key
        self._llm_client: OpenAI | None = None
        self._analyzer = ContentAnalyzer()

    @property
    def llm_client(self) -> OpenAI | None:
        """Lazy initialization of LLM client.

        Returns:
            OpenAI client or None
        """
        if self._llm_client is None and self.api_key and self.use_llm:
            self._llm_client = OpenAI(
                api_key=self.api_key,
                base_url=self.config.LLM_BASE_URL,
                timeout=self.config.LLM_TIMEOUT,
            )
        return self._llm_client

    def calculate(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
    ) -> QualityScore:
        """Calculate quality score for a document.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level

        Returns:
            QualityScore with detailed breakdown
        """
        # Check cache first
        cached_score = _reward_cache.get(action.content)
        if cached_score is not None:
            return cached_score

        # Try LLM-based scoring if available
        if self.use_llm and self.llm_client:
            llm_score = self._llm_quality_score(
                action, documents, task_keywords, task_difficulty
            )
            if llm_score is not None:
                # Cache the result
                _reward_cache.set(action.content, llm_score)
                return llm_score

        # Fall back to rule-based scoring
        return self._rule_based_quality_score(
            action, documents, task_keywords, task_difficulty
        )

    def _rule_based_quality_score(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
    ) -> QualityScore:
        """Calculate quality score using rule-based methods.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level

        Returns:
            QualityScore with detailed breakdown
        """
        content = action.content.lower()
        keywords = task_keywords or []

        # Calculate individual scores
        content_depth = self._score_content_depth(action.content, task_difficulty)
        structure = self._score_structure(action.content)
        relevance = self._score_relevance(content, keywords)
        professionalism = self._score_professionalism(action.content)

        # Calculate weighted overall
        overall = (
            content_depth * self.config.QUALITY_CONTENT_DEPTH_WEIGHT
            + structure * self.config.QUALITY_STRUCTURE_WEIGHT
            + relevance * self.config.QUALITY_RELEVANCE_WEIGHT
            + professionalism * self.config.QUALITY_PROFESSIONALISM_WEIGHT
        )

        # Generate feedback
        feedback = self._generate_quality_feedback(
            content_depth, structure, relevance, professionalism
        )

        return QualityScore(
            overall=overall,
            content_depth=content_depth,
            structure=structure,
            relevance=relevance,
            professionalism=professionalism,
            feedback=feedback,
            llm_used=False,
        )

    def _score_content_depth(self, content: str, difficulty: str) -> float:
        """Score content depth based on length and substance.

        Args:
            content: Document content
            difficulty: Task difficulty level

        Returns:
            Score in [0.0, 1.0]
        """
        length = len(content)

        # Get difficulty-based thresholds from config
        thresholds = self.config.CONTENT_DEPTH_THRESHOLDS
        min_len, target_len = thresholds.get(difficulty, thresholds[DifficultyLevel.MEDIUM])

        if length < min_len:
            return 0.0
        if length >= target_len:
            return 1.0

        # Linear interpolation
        return (length - min_len) / (target_len - min_len)

    def _score_structure(self, content: str) -> float:
        """Score document structure.

        Args:
            content: Document content

        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.0
        content_lower = content.lower()

        # Has headers
        if self._analyzer.has_headers(content):
            score += self.config.STRUCTURE_HAS_HEADERS_WEIGHT

        # Has lists
        if self._analyzer.has_lists(content):
            score += self.config.STRUCTURE_HAS_LISTS_WEIGHT

        # Has multiple paragraphs
        if self._analyzer.count_paragraphs(content) >= self.config.STRUCTURE_MIN_PARAGRAPHS:
            score += self.config.STRUCTURE_HAS_PARAGRAPHS_WEIGHT

        # Has structural keywords
        if any(kw in content_lower for kw in self.config.STRUCTURAL_KEYWORDS):
            score += self.config.STRUCTURE_HAS_KEYWORDS_WEIGHT

        return min(score, 1.0)

    def _score_relevance(self, content: str, keywords: list[str]) -> float:
        """Score relevance based on keyword presence.

        Args:
            content: Document content (lowercase)
            keywords: Required keywords

        Returns:
            Score in [0.0, 1.0]
        """
        if not keywords:
            return 1.0

        matches = sum(1 for kw in keywords if kw.lower() in content)
        return matches / len(keywords)

    def _score_professionalism(self, content: str) -> float:
        """Score professionalism of writing.

        Args:
            content: Document content

        Returns:
            Score in [0.0, 1.0]
        """
        score = 1.0
        content_lower = content.lower()

        # Deductions for unprofessional elements
        for pattern in self.config.UNPROFESSIONAL_PATTERNS:
            if pattern in content_lower:
                score -= UNPROFESSIONAL_PATTERN_DEDUCTION

        # Check for proper sentence structure
        sentences = content.split(".")
        if len(sentences) > 1:
            # Has multiple sentences
            score += SENTENCE_STRUCTURE_BONUS

        return max(0.0, min(score, 1.0))

    def _generate_quality_feedback(
        self,
        content_depth: float,
        structure: float,
        relevance: float,
        professionalism: float,
    ) -> list[str]:
        """Generate feedback based on scores.

        Args:
            content_depth: Content depth score
            structure: Structure score
            relevance: Relevance score
            professionalism: Professionalism score

        Returns:
            List of feedback strings
        """
        feedback = []

        if content_depth < FEEDBACK_THRESHOLD:
            feedback.append("Content could be more detailed and substantive.")
        if structure < FEEDBACK_THRESHOLD:
            feedback.append("Document structure could be improved with more headers and sections.")
        if relevance < FEEDBACK_THRESHOLD:
            feedback.append("Content could be more relevant to the task requirements.")
        if professionalism < FEEDBACK_THRESHOLD:
            feedback.append("Writing could be more professional and clear.")

        if not feedback:
            feedback.append("Good quality work.")

        return feedback

    def _llm_quality_score(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
    ) -> QualityScore | None:
        """Calculate quality score using LLM.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level

        Returns:
            QualityScore with detailed breakdown, or None if LLM fails
        """
        if not self.llm_client:
            return None

        try:
            # Get agent info
            agent_entry = get_workflow_entry(action.agent_id)
            agent_role = agent_entry["role"] if agent_entry else "Unknown"
            agent_name = agent_entry["name"] if agent_entry else action.agent_id

            # Build prompt
            prompt = self._build_quality_prompt(
                action,
                agent_name,
                agent_role,
                documents,
                task_keywords,
                task_difficulty,
            )

            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of technical planning documents. Grade objectively and consistently.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            return QualityScore(
                overall=float(result.get("overall_score", 0.0)),
                content_depth=float(result.get("content_depth", 0.0)),
                structure=float(result.get("structure", 0.0)),
                relevance=float(result.get("relevance", 0.0)),
                professionalism=float(result.get("professionalism", 0.0)),
                feedback=result.get("feedback", []),
                llm_used=True,
            )
        except Exception as e:
            # Log error and fall back to rule-based
            logger.warning(f"LLM quality scoring failed: {e}")
            return None

    def _build_quality_prompt(
        self,
        action: SkyPlanAction,
        agent_name: str,
        agent_role: str,
        documents: dict[str, Document],
        task_keywords: list[str] | None,
        task_difficulty: str,
    ) -> str:
        """Build the quality assessment prompt for LLM.

        Args:
            action: The action taken
            agent_name: Agent display name
            agent_role: Agent role description
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level

        Returns:
            Formatted prompt string
        """
        # Truncate content for prompt
        preview_length = self.config.LLM_CONTENT_PREVIEW_LENGTH
        content_preview = action.content[:preview_length]
        if len(action.content) > preview_length:
            content_preview += "..."

        keywords_str = ", ".join(task_keywords) if task_keywords else "None"

        return f"""Evaluate the quality of the following planning document.

Agent: {agent_name} ({agent_role})
Action: {action.action_type}
Task Difficulty: {task_difficulty}
Required Keywords: {keywords_str}

Document Content:
{content_preview}

Evaluate on the following dimensions (0.0 to 1.0):

1. Content Depth (0.0-1.0):
   - Is the content substantive and detailed?
   - Is there sufficient depth for the task difficulty?
   - Are claims supported with evidence?

2. Structure (0.0-1.0):
   - Is the document well-organized with clear sections?
   - Are headers and formatting used appropriately?
   - Is there a logical flow?

3. Relevance (0.0-1.0):
   - Is the content relevant to the agent's role?
   - Does it address the task requirements?
   - Are required keywords present?

4. Professionalism (0.0-1.0):
   - Is the writing clear and professional?
   - Is the tone appropriate?
   - Are there any contradictions or inconsistencies?

Return a JSON object with:
{{
    "overall_score": 0.0-1.0,
    "content_depth": 0.0-1.0,
    "structure": 0.0-1.0,
    "relevance": 0.0-1.0,
    "professionalism": 0.0-1.0,
    "feedback": ["Specific feedback points"]
}}

Be objective and consistent in your grading."""


# ============================================================================
# Teamwork Bonus Calculator
# ============================================================================


class TeamworkBonusCalculator(BaseCalculator):
    """Calculates teamwork bonus based on collaboration and references."""

    def __init__(self, config: RewardConfig | None = None):
        super().__init__(config)
        self._entity_cache: dict[str, set[str]] = {}
        self._analyzer = ContentAnalyzer()

    def calculate(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        previous_agent_id: str | None = None,
    ) -> TeamworkScore:
        """Calculate teamwork score for an action.

        Args:
            action: The action taken
            documents: All documents produced so far
            previous_agent_id: The previous agent in the workflow

        Returns:
            TeamworkScore with detailed breakdown
        """
        references: dict[str, float] = {}
        total_bonus = 0.0

        # Get required documents for this agent
        required_docs = get_required_documents(action.agent_id)

        # Check references to each required document
        for doc_type in required_docs:
            if doc_type in documents:
                ref_score = self._calculate_reference_score(
                    action.content,
                    documents[doc_type].content,
                    doc_type,
                )
                references[doc_type] = ref_score
                total_bonus += ref_score

        # Cap at maximum teamwork bonus
        total_bonus = min(total_bonus, self.config.TEAMWORK_BONUS_MAX)

        # Evaluate handoff quality
        handoff_quality = self._evaluate_handoff_quality(
            action,
            documents,
            previous_agent_id,
        )

        # Generate feedback
        feedback = self._generate_teamwork_feedback(references, handoff_quality)

        return TeamworkScore(
            overall=total_bonus,
            references=references,
            handoff_quality=handoff_quality,
            feedback=feedback,
        )

    def _calculate_reference_score(
        self,
        current_content: str,
        previous_content: str,
        doc_type: str,
    ) -> float:
        """Calculate reference score based on how well current doc references previous.

        Args:
            current_content: Current document content
            previous_content: Previous document content
            doc_type: Type of previous document

        Returns:
            Reference score in [0.0, 0.2]
        """
        current_lower = current_content.lower()

        # Extract key entities from previous document
        entities = self._extract_entities(previous_content, doc_type)

        if not entities:
            # No entities to reference
            return self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.NONE]

        # Check for references
        referenced_count = sum(
            1 for entity in entities
            if entity.lower() in current_lower
        )

        reference_ratio = referenced_count / len(entities)

        # Determine reference level using config thresholds
        if reference_ratio == 0:
            return self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.NONE]
        elif reference_ratio < self.config.TEAMWORK_REFERENCE_THRESHOLD_LOW:
            return self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.GENERIC]
        elif reference_ratio < self.config.TEAMWORK_REFERENCE_THRESHOLD_HIGH:
            return self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.SPECIFIC]
        else:
            return self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.INTEGRATED]

    def _extract_entities(self, content: str, doc_type: str) -> set[str]:
        """Extract key entities from document content.

        Args:
            content: Document content
            doc_type: Type of document

        Returns:
            Set of entity strings
        """
        # Use content hash + doc_type for cache key to avoid collisions
        cache_key = f"{doc_type}:{hashlib.sha256(content.encode()).hexdigest()}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        entities = set()
        words = self._analyzer.extract_words(
            content,
            min_length=self.config.ENTITY_MIN_WORD_LENGTH,
        )

        for word in words:
            # Keep if it's a significant term
            # A word is an entity if it meets ANY of these criteria
            is_capitalized = word[0].isupper() if word else False
            has_numbers = any(c.isdigit() for c in word)
            has_percent = "%" in word

            if is_capitalized or has_numbers or has_percent:
                entities.add(word)

        self._entity_cache[cache_key] = entities
        return entities

    def _evaluate_handoff_quality(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        previous_agent_id: str | None,
    ) -> float:
        """Evaluate the quality of handoff to next agent.

        Args:
            action: The action taken
            documents: All documents produced so far
            previous_agent_id: The previous agent in the workflow

        Returns:
            Handoff quality score in [0.0, 1.0]
        """
        if not previous_agent_id:
            return 1.0  # First agent, no handoff to evaluate

        score = 1.0
        content_lower = action.content.lower()

        # Check if agent acknowledges previous work
        if any(phrase in content_lower for phrase in self.config.ACKNOWLEDGMENT_PHRASES):
            score += HANDOFF_ACKNOWLEDGMENT_BONUS

        # Check if agent sets up next agent
        next_agent_entry = get_workflow_entry(action.agent_id)
        if next_agent_entry and next_agent_entry.get("next_agent"):
            # Agent is not last, should set up next
            if any(phrase in content_lower for phrase in self.config.SETUP_PHRASES):
                score += HANDOFF_SETUP_BONUS

        return min(score, 1.0)

    def _generate_teamwork_feedback(
        self,
        references: dict[str, float],
        handoff_quality: float,
    ) -> list[str]:
        """Generate feedback based on teamwork scores.

        Args:
            references: Reference scores by document type
            handoff_quality: Handoff quality score

        Returns:
            List of feedback strings
        """
        feedback = []

        # Check for missing references
        for doc_type, score in references.items():
            if score == 0.0:
                feedback.append(f"No reference to {doc_type} document.")
            elif score < self.config.TEAMWORK_REFERENCE_LEVELS[ReferenceLevel.SPECIFIC]:
                feedback.append(f"Could reference {doc_type} more specifically.")

        # Check handoff quality
        if handoff_quality < HANDOFF_QUALITY_THRESHOLD:
            feedback.append("Could improve handoff to next agent.")

        if not feedback:
            feedback.append("Good collaboration with previous agents.")

        return feedback


# ============================================================================
# Completion Bonus Calculator
# ============================================================================


class CompletionBonusCalculator(BaseCalculator):
    """Calculates completion bonus (all-or-nothing)."""

    def calculate(
        self,
        documents: dict[str, Document],
        required_document_types: list[str] | None = None,
    ) -> float:
        """Calculate completion bonus.

        Args:
            documents: All documents produced
            required_document_types: Required document types

        Returns:
            Completion bonus (0.0 or COMPLETION_BONUS)
        """
        if required_document_types is None:
            required_document_types = get_all_document_types()

        # Check if all required documents are present
        for doc_type in required_document_types:
            if doc_type not in documents:
                return 0.0

            # Check if document meets minimum quality
            doc = documents[doc_type]
            if len(doc.content) < self.config.MIN_CONTENT_LENGTH:
                return 0.0

        return self.config.COMPLETION_BONUS


# ============================================================================
# Penalty Calculator
# ============================================================================


class PenaltyCalculator(BaseCalculator):
    """Calculates penalties for constraint violations."""

    def __init__(self, config: RewardConfig | None = None):
        super().__init__(config)
        self._analyzer = ContentAnalyzer()

    def calculate(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        required_sections: list[str] | None = None,
    ) -> PenaltyScore:
        """Calculate penalties for an action.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            required_sections: Required sections for the document

        Returns:
            PenaltyScore with detailed breakdown
        """
        penalties: dict[str, float] = {}
        reasons: list[str] = []

        # Length penalties
        length_penalty = self._check_length_penalty(action.content)
        if length_penalty < 0:
            penalties[PenaltyType.LENGTH] = length_penalty
            if length_penalty == self.config.PENALTY_EMPTY:
                reasons.append("Empty content.")
            elif length_penalty == self.config.PENALTY_TOO_SHORT:
                reasons.append("Content too short.")

        # Structure penalties
        structure_penalty = self._check_structure_penalty(action.content)
        if structure_penalty < 0:
            if not self._analyzer.has_headers(action.content):
                penalties["no_headers"] = self.config.PENALTY_NO_HEADERS
                reasons.append("No headers in document.")
            if not self._analyzer.has_lists(action.content):
                penalties["no_lists"] = self.config.PENALTY_NO_LISTS
                reasons.append("No lists or tables where expected.")

        # Content penalties
        content_penalties = self._check_content_penalties(
            action,
            documents,
            task_keywords,
            required_sections,
        )
        for penalty_name, penalty_value in content_penalties.items():
            if penalty_value < 0:
                penalties[penalty_name] = penalty_value
                if "missing_section" in penalty_name:
                    reasons.append(f"Missing required section: {penalty_name.replace('missing_section_', '')}.")
                elif "missing_keyword" in penalty_name:
                    reasons.append(f"Missing required keyword: {penalty_name.replace('missing_keyword_', '')}.")
                elif "contradiction" in penalty_name:
                    reasons.append("Potential contradiction with previous work.")
                elif "unrealistic" in penalty_name:
                    reasons.append("Unrealistic claims detected.")

        # Role penalties
        role_penalty = self._check_role_penalty(action)
        if role_penalty < 0:
            penalties[PenaltyType.ROLE] = role_penalty
            reasons.append("Action not aligned with agent role.")

        # Calculate total and cap
        total = sum(penalties.values())
        total = max(total, self.config.PENALTY_MAX_PER_STEP)

        return PenaltyScore(
            total=total,
            penalties=penalties,
            reasons=reasons,
        )

    def _check_length_penalty(self, content: str) -> float:
        """Check for length-related penalties.

        Args:
            content: Document content

        Returns:
            Penalty value (negative or zero)
        """
        if not content or not content.strip():
            return self.config.PENALTY_EMPTY

        if len(content.strip()) < self.config.MIN_CONTENT_LENGTH:
            return self.config.PENALTY_TOO_SHORT

        return 0.0

    def _check_structure_penalty(self, content: str) -> float:
        """Check for structure-related penalties.

        Args:
            content: Document content

        Returns:
            Penalty value (negative or zero)
        """
        penalty = 0.0

        # Check for headers
        if not self._analyzer.has_headers(content):
            penalty += self.config.PENALTY_NO_HEADERS

        # Check for lists
        if not self._analyzer.has_lists(content):
            penalty += self.config.PENALTY_NO_LISTS

        return penalty

    def _check_content_penalties(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        required_sections: list[str] | None = None,
    ) -> dict[str, float]:
        """Check for content-related penalties.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            required_sections: Required sections for the document

        Returns:
            Dictionary of penalty values
        """
        penalties: dict[str, float] = {}
        content_lower = action.content.lower()

        # Check for missing keywords
        if task_keywords:
            missing_keywords = [
                kw for kw in task_keywords
                if kw.lower() not in content_lower
            ]
            if missing_keywords:
                penalties["missing_keywords"] = (
                    len(missing_keywords) * self.config.PENALTY_MISSING_KEYWORD
                )

        # Check for missing sections
        if required_sections:
            missing_sections = [
                section for section in required_sections
                if section.lower() not in content_lower
            ]
            if missing_sections:
                penalties["missing_sections"] = (
                    len(missing_sections) * self.config.PENALTY_MISSING_SECTION
                )

        # Check for contradictions with previous documents
        contradiction_penalty = self._check_contradictions(action, documents)
        if contradiction_penalty < 0:
            penalties["contradiction"] = contradiction_penalty

        # Check for unrealistic claims
        unrealistic_penalty = self._check_unrealistic_claims(action.content)
        if unrealistic_penalty < 0:
            penalties["unrealistic"] = unrealistic_penalty

        return penalties

    def _check_contradictions(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
    ) -> float:
        """Check for contradictions with previous documents.

        Args:
            action: The action taken
            documents: All documents produced so far

        Returns:
            Penalty value (negative or zero)
        """
        content_lower = action.content.lower()

        # Check for contradiction phrases from config
        for phrase in self.config.CONTRADICTION_PHRASES:
            if phrase in content_lower:
                # Potential contradiction, but could be legitimate
                # Just a small penalty for now
                return self.config.PENALTY_CONTRADICTION * CONTRADICTION_PENALTY_MULTIPLIER

        return 0.0

    def _check_unrealistic_claims(self, content: str) -> float:
        """Check for unrealistic claims.

        Args:
            content: Document content

        Returns:
            Penalty value (negative or zero)
        """
        content_lower = content.lower()

        # Check for unrealistic timeline patterns from config
        for pattern in self.config.UNREALISTIC_PATTERNS:
            if pattern in content_lower:
                return self.config.PENALTY_UNREALISTIC

        return 0.0

    def _check_role_penalty(self, action: SkyPlanAction) -> float:
        """Check if action is aligned with agent's role.

        Args:
            action: The action taken

        Returns:
            Penalty value (negative or zero)
        """
        # Get agent's allowed actions
        from .workflow import get_allowed_actions

        allowed_actions = get_allowed_actions(action.agent_id)

        if action.action_type not in allowed_actions:
            return self.config.PENALTY_WRONG_ROLE

        return 0.0


# ============================================================================
# Score Normalizer
# ============================================================================


    def _calculate_approval_bonus(self, action: SkyPlanAction, documents: dict[str, Document]) -> float:
        """Calculate approval bonus for status-changing actions.

        Args:
            action: The action taken
            documents: All documents with current status

        Returns:
            Approval bonus value (0.0 to APPROVAL_FINAL_BONUS)
        """
        # Check if action changes document status
        status_change = DocumentStatusConfig.STATUS_TRANSITIONS.get(action.action_type)
        if not status_change:
            return 0.0

        # Determine bonus based on action type
        bonus_map = {
            "MARK_DOCUMENT_REVIEW": self.config.APPROVAL_MARK_REVIEW_BONUS,
            "APPROVE_DOCUMENT": self.config.APPROVAL_APPROVE_BONUS,
            "REJECT_DOCUMENT": self.config.APPROVAL_REJECT_BONUS,
            "APPROVE_ALL_DOCUMENTS": self.config.APPROVAL_ALL_BONUS,
            "FINAL_APPROVAL": self.config.APPROVAL_FINAL_BONUS,
        }

        return bonus_map.get(action.action_type, 0.0)

class ScoreNormalizer:
    """Normalizes raw reward scores to [0.0, 1.0] range."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or reward_config

    def normalize(
        self,
        raw_score: float,
        min_possible: float | None = None,
        max_possible: float | None = None,
    ) -> float:
        """Normalize raw score to [0.0, 1.0] range.

        Args:
            raw_score: Raw score to normalize
            min_possible: Minimum possible raw score (uses config if None)
            max_possible: Maximum possible raw score (uses config if None)

        Returns:
            Normalized score in [0.0, 1.0]
        """
        # Use config values if not provided
        if min_possible is None:
            min_possible = self.config.NORMALIZER_MIN_POSSIBLE
        if max_possible is None:
            max_possible = self.config.NORMALIZER_MAX_POSSIBLE

        # Scale to [0, 1] range
        if max_possible > min_possible:
            scaled = (raw_score - min_possible) / (max_possible - min_possible)
        else:
            scaled = 0.5  # Default to middle if range is invalid

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, scaled))


# ============================================================================
# Main Reward Calculator
# ============================================================================


class RewardCalculator:
    """Main reward calculator that orchestrates all components."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        use_llm: bool = True,
        api_key: str | None = None,
    ):
        """Initialize the reward calculator.

        Args:
            config: Reward configuration
            use_llm: Whether to use LLM for quality assessment
            api_key: API key for LLM service (overrides env var)
        """
        self.config = config or reward_config
        self.use_llm = use_llm
        # Use provided API key or get from environment
        self.api_key = api_key or environ.get("SKYPLAN_LLM_API_KEY")

        # Initialize component calculators
        self.quality_calculator = QualityBonusCalculator(
            config=self.config,
            use_llm=self.use_llm,
            api_key=self.api_key,
        )
        self.teamwork_calculator = TeamworkBonusCalculator(config=self.config)
        self.completion_calculator = CompletionBonusCalculator(config=self.config)
        self.penalty_calculator = PenaltyCalculator(config=self.config)
        self.normalizer = ScoreNormalizer(config=self.config)
        self.approval_calculator = ApprovalBonusCalculator(config=self.config)
        self._approved_documents = set()

        # Episode tracking
        self._step_rewards: list[StepReward] = []
        self._previous_agent_id: str | None = None

    def calculate_step_reward(
        self,
        action: SkyPlanAction,
        documents: dict[str, Document],
        task_keywords: list[str] | None = None,
        task_difficulty: str = "medium",
        required_sections: list[str] | None = None,
        feedback_generated: list | None = None,
        feedback_resolved: list | None = None,
        new_approvals: list | None = None,
    ) -> StepReward:
        """Calculate reward for a single step."""
        """Calculate reward for a single step.

        Args:
            action: The action taken
            documents: All documents produced so far
            task_keywords: Required keywords for the task
            task_difficulty: Task difficulty level
            required_sections: Required sections for the document
            feedback_generated: List of feedback items generated this step
            feedback_resolved: List of feedback items resolved this step
            new_approvals: List of (document_type, approver_agent) tuples for approvals

        Returns:
            StepReward with detailed breakdown
        """
        # Calculate quality bonus
        quality_score = self.quality_calculator.calculate(
            action,
            documents,
            task_keywords,
            task_difficulty,
        )
        quality_bonus = quality_score.overall * self.config.QUALITY_BONUS_MAX

        # Calculate teamwork bonus
        teamwork_score = self.teamwork_calculator.calculate(
            action,
            documents,
            self._previous_agent_id,
        )
        teamwork_bonus = teamwork_score.overall

        # Calculate penalties
        penalty_score = self.penalty_calculator.calculate(
            action,
            documents,
            task_keywords,
            required_sections,
        )
        penalty = penalty_score.total

        # Calculate total step reward
        # Calculate approval bonus for status-changing actions
        approval_bonus = self._calculate_approval_bonus(action, documents)
        
        # Sum all reward components
        total = quality_bonus + teamwork_bonus + penalty + approval_bonus + feedback_generation_reward + feedback_resolution_reward + document_approval_reward

        # Calculate feedback generation reward if feedback was generated\n        feedback_generation_reward = 0.0\n        if feedback_generated:\n            feedback_generation_reward = self.calculate_feedback_generation_reward(feedback_generated)\n\n        # Calculate feedback resolution reward if feedback was resolved\n        feedback_resolution_reward = 0.0\n        if feedback_resolved:\n            feedback_resolution_reward = self.calculate_feedback_resolution_reward(feedback_resolved)\n\n        # Calculate document approval reward if new approvals occurred\n        document_approval_reward = 0.0\n        if new_approvals:\n            document_approval_reward = self.calculate_document_approval_reward(new_approvals)

        # Create step reward
        step_reward = StepReward(
            quality_bonus=quality_bonus,
            teamwork_bonus=teamwork_bonus,
            penalty=penalty,
            total=total,
            quality_score=quality_score,
            teamwork_score=teamwork_score,
            penalty_score=penalty_score,

		feedback_generation_reward=feedback_generation_reward,
		feedback_resolution_reward=feedback_resolution_reward,
		document_approval_reward=document_approval_reward
        )

        # Track for episode
        self._step_rewards.append(step_reward)
        self._previous_agent_id = action.agent_id

        return step_reward

    def calculate_feedback_generation_reward(self, feedback_list: list) -> float:
        """Calculate reward for feedback generation.

        Args:
            feedback_list: List of feedback generated this step

        Returns:
            Total reward for feedback generation
        """
        if not self.config.FEEDBACK_GENERATION_BONUS or not feedback_list:
            return 0.0

        total_reward = 0.0
        for feedback in feedback_list:
            if feedback.from_agent == "sam":
                total_reward += self.config.STRATEGIC_FEEDBACK_BONUS
            elif feedback.from_agent == "taylor":
                total_reward += self.config.VALIDATOR_FEEDBACK_BONUS
            else:
                total_reward += self.config.COLLABORATIVE_FEEDBACK_BONUS

        return total_reward

    def calculate_feedback_resolution_reward(self, resolved_feedback: list) -> float:
        """Calculate reward for resolving feedback.

        Args:
            resolved_feedback: List of feedback items resolved this step

        Returns:
            Total reward for feedback resolution
        """
        if not self.config.FEEDBACK_RESOLUTION_BONUS or not resolved_feedback:
            return 0.0

        total_reward = 0.0
        for feedback in resolved_feedback:
            if feedback.from_agent in ["taylor", "sam"]:
                total_reward += self.config.PRIMARY_FEEDBACK_RESOLUTION_BONUS
            else:
                total_reward += self.config.PEER_FEEDBACK_RESOLUTION_BONUS

        return total_reward

    def calculate_document_approval_reward(self, new_approvals: list) -> float:
        """Calculate reward for document approvals.

        Args:
            new_approvals: List of (document_type, approver_agent) tuples

        Returns:
            Total reward for document approvals
        """
        if not self.config.DOCUMENT_APPROVAL_BONUS or not new_approvals:
            return 0.0

        total_reward = 0.0
        for doc_type, approver in new_approvals:
            if approver == "taylor":
                total_reward += self.config.TAYLOR_APPROVAL_BONUS
            elif approver == "sam":
                total_reward += self.config.SAM_APPROVAL_BONUS

        return total_reward

    def calculate_episode_reward(
        self,
        documents: dict[str, Document],
        required_document_types: list[str] | None = None,
    ) -> EpisodeReward:
        """Calculate final episode reward.

        Args:
            documents: All documents produced
            required_document_types: Required document types

        Returns:
            EpisodeReward with final score
        """
        # Calculate completion bonus
        completion_bonus = self.completion_calculator.calculate(
            documents,
            required_document_types,
        )

        # Sum step rewards
        step_total = sum(step.total for step in self._step_rewards)

        # Calculate raw total
        raw_total = step_total + completion_bonus

        # Normalize to [0.0, 1.0]
        final_score = self.normalizer.normalize(raw_total)

        # Create breakdown
        breakdown = {
            "quality_bonus": sum(step.quality_bonus for step in self._step_rewards),
            "teamwork_bonus": sum(step.teamwork_bonus for step in self._step_rewards),
            "penalty": sum(step.penalty for step in self._step_rewards),
            "completion_bonus": completion_bonus,
        }

        # Create episode reward
        episode_reward = EpisodeReward(
            step_rewards=self._step_rewards.copy(),
            completion_bonus=completion_bonus,
            total_raw=raw_total,
            final_score=final_score,
            breakdown=breakdown,
        )

        return episode_reward

    def reset(self) -> None:
        """Reset the calculator for a new episode."""
        self._step_rewards.clear()
        self._previous_agent_id = None

    def get_step_count(self) -> int:
        """Get the number of steps processed.

        Returns:
            Number of steps processed
        """
        return len(self._step_rewards)

    def get_current_total(self) -> float:
        """Get the current total reward (before completion bonus).

        Returns:
            Current total reward
        """
        return sum(step.total for step in self._step_rewards)


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_reward(
    action: SkyPlanAction,
    documents: dict[str, Document],
    task_keywords: list[str] | None = None,
    task_difficulty: str = "medium",
    required_sections: list[str] | None = None,
    use_llm: bool = True,
    api_key: str | None = None,
) -> StepReward:
    """Convenience function to calculate step reward.

    Args:
        action: The action taken
        documents: All documents produced so far
        task_keywords: Required keywords for the task
        task_difficulty: Task difficulty level
        required_sections: Required sections for the document
        use_llm: Whether to use LLM for quality assessment
        api_key: API key for LLM service

    Returns:
        StepReward with detailed breakdown
    """
    calculator = RewardCalculator(
        use_llm=use_llm,
        api_key=api_key,
    )
    return calculator.calculate_step_reward(
        action,
        documents,
        task_keywords,
        task_difficulty,
        required_sections,
    )


def clear_reward_cache() -> None:
    """Clear the reward cache."""
    _reward_cache.clear()


def get_cache_size() -> int:
    """Get the current cache size.

    Returns:
        Number of cached entries
    """
    return _reward_cache.size()
