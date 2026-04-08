"""Shared content-analysis helpers used by graders, rewards, and observations."""

from __future__ import annotations

import re


HEADER_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
PHASE_LABEL_PATTERN = re.compile(r"\b(?:phase|sprint|iteration|milestone)\s+\d+\b", re.IGNORECASE)


def has_markdown_headers(content: str) -> bool:
    """Return whether the content contains at least one Markdown header line."""

    return bool(HEADER_PATTERN.search(content))


def count_markdown_headers(content: str) -> int:
    """Count Markdown header lines."""

    return len(HEADER_PATTERN.findall(content))


def has_markdown_lists(content: str) -> bool:
    """Return whether the content contains an ordered or unordered Markdown list."""

    return bool(UNORDERED_LIST_PATTERN.search(content) or ORDERED_LIST_PATTERN.search(content))


def count_paragraph_blocks(content: str) -> int:
    """Count non-empty paragraph blocks separated by blank lines."""

    return len([block for block in PARAGRAPH_SPLIT_PATTERN.split(content) if block.strip()])


def keyword_coverage_ratio(content: str, keywords: list[str]) -> float:
    """Measure the fraction of required keywords present in the content."""

    if not keywords:
        return 0.0

    content_lower = content.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
    return matches / len(keywords)


def extract_phase_labels(content: str) -> set[str]:
    """Extract explicit phase/sprint/milestone labels for cross-document consistency checks."""

    return {match.group(0).lower() for match in PHASE_LABEL_PATTERN.finditer(content)}
