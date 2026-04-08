"""Tests for grading and shared content-analysis heuristics."""

from AgentEnv.content_utils import count_markdown_headers, has_markdown_headers, has_markdown_lists
from AgentEnv.models import Document
from AgentEnv.tasks import BaseGrader, _check_tasks_vs_roadmap_consistency


def test_shared_markdown_structure_helpers_require_real_markdown_syntax():
    """Header/list detection should look for Markdown structure, not raw punctuation anywhere."""

    content = "# Overview\n\n1. First task\n- Second task\n\nParagraph text."

    assert has_markdown_headers(content) is True
    assert count_markdown_headers(content) == 1
    assert has_markdown_lists(content) is True
    assert has_markdown_headers("plain text with #inline hash") is False
    assert has_markdown_lists("asterisk*insideword") is False


def test_keyword_relevance_scores_keyword_coverage_instead_of_any_single_hit():
    """A document should not get full keyword credit for matching only one required keyword."""

    documents = {
        "PRD": Document.create("PRD", "# PRD\n\nContains authentication only.", "elon"),
        "TRD": Document.create("TRD", "# TRD\n\nContains security only.", "jordan"),
    }

    score = BaseGrader.check_keyword_relevance(
        documents,
        ["authentication", "security", "passwordless"],
    )

    assert 0.0 < score < 1.0


def test_tasks_vs_roadmap_consistency_needs_more_than_generic_term_overlap():
    """Generic mentions alone should not produce a perfect consistency score."""

    documents = {
        "TASKS": Document.create(
            "TASKS",
            "# Tasks\n\n- task item one\n- task item two\n\nstory and ticket planning",
            "robert",
        ),
        "ROADMAP": Document.create(
            "ROADMAP",
            "# Roadmap\n\nphase planning with sprint setup and milestone review",
            "robert",
        ),
    }

    score = _check_tasks_vs_roadmap_consistency(documents)

    assert 0.0 < score < 1.0
