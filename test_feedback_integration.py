"""Feedback integration tests for the SkyPlan environment."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from AgentEnv.client import AgentenvEnv
from AgentEnv.models import Document, Feedback, LastAction, SkyPlanAction
from AgentEnv.tasks import TASKS


class _StubRewardCalculator:
    """Lightweight reward stub for deterministic environment tests."""

    def __init__(self, *args, **kwargs):
        self._step_total = 0.25

    def reset(self) -> None:
        """Reset state between episodes."""

    def calculate_step_reward(self, **kwargs) -> SimpleNamespace:
        """Return a fixed step reward without external dependencies."""

        return SimpleNamespace(total=self._step_total)


@pytest.fixture
def make_env(monkeypatch):
    """Create a SkyPlanEnvironment with reward calculation stubbed out."""

    env_module = import_module("AgentEnv.server.AgentEnv_environment")
    monkeypatch.setattr(env_module, "RewardCalculator", _StubRewardCalculator)

    def _make_env():
        return env_module.SkyPlanEnvironment(use_llm_reward=False)

    return _make_env


def _make_action(
    agent_id: str,
    action_type: str,
    content: str,
    reasoning: str = "Addressing the task with clear follow-up and explicit revisions.",
) -> SkyPlanAction:
    """Build a valid SkyPlanAction for tests."""

    return SkyPlanAction(
        agent_id=agent_id,
        action_type=action_type,
        reasoning=reasoning,
        content=content,
    )


def _make_document(
    doc_type: str,
    author: str,
    content: str,
    status: str = "draft",
) -> Document:
    """Build a document with deterministic status for feedback tests."""

    document = Document.create(doc_type=doc_type, content=content, author=author)
    document.status = status
    return document


def _short_workflow_content(label: str, task_name: str) -> str:
    """Generate intentionally short content to trigger feedback paths."""

    return f"# {label}\n{task_name} notes"


def test_generate_collaborative_feedback_for_low_quality_action(make_env):
    """Low-quality action output should generate collaborative feedback."""

    env = make_env()
    action = _make_action(
        agent_id="maya",
        action_type="SEARCH_MARKET",
        content="# Research\nthin notes",
    )

    feedback = env._generate_collaborative_feedback(action)

    assert feedback is not None
    assert feedback.from_agent == "maya"
    assert feedback.document_type == "RESEARCH"
    assert feedback.feedback_type == "suggestion"
    assert "Document quality score" in feedback.comment


def test_generate_validation_feedback_updates_statuses_and_targets_authors(make_env):
    """Taylor should generate targeted feedback and update document review status."""

    env = make_env()
    env._documents = {
        "RESEARCH": _make_document("RESEARCH", "maya", "brief"),
        "PRD": _make_document("PRD", "elon", "brief"),
        "TRD": _make_document("TRD", "jordan", "brief"),
        "ARCHITECTURE": _make_document("ARCHITECTURE", "jordan", "brief"),
        "ROADMAP": _make_document("ROADMAP", "robert", "brief"),
        "TASKS": _make_document("TASKS", "robert", "brief"),
        "VALIDATION": _make_document("VALIDATION", "taylor", "# Validation\nreview summary"),
    }

    feedback_list = env._generate_validation_feedback("taylor")

    assert len(feedback_list) == 6
    assert all(feedback.from_agent == "taylor" for feedback in feedback_list)
    assert all(feedback.feedback_type == "concern" for feedback in feedback_list)
    assert {feedback.to_agent for feedback in feedback_list} == {
        "maya",
        "elon",
        "jordan",
        "robert",
    }
    assert all(
        env._documents[doc_type].status == "rejected"
        for doc_type in ("RESEARCH", "PRD", "TRD", "ARCHITECTURE", "ROADMAP", "TASKS")
    )
    assert env._documents["VALIDATION"].status == "in_review"


def test_generate_strategic_feedback_flags_missing_docs_and_approval_gaps(
    make_env,
    monkeypatch,
):
    """Sam should raise missing-doc, consistency, and approval-gap feedback."""

    env = make_env()
    env._documents = {
        "RESEARCH": _make_document("RESEARCH", "maya", "complete", status="approved"),
        "PRD": _make_document("PRD", "elon", "needs revision", status="draft"),
        "VALIDATION": _make_document("VALIDATION", "taylor", "pending", status="draft"),
        "STRATEGY": _make_document("STRATEGY", "sam", "pending", status="in_review"),
    }
    monkeypatch.setattr(env, "_check_document_consistency", lambda: 0.4)

    feedback_list = env._generate_strategic_feedback("sam")

    assert len(feedback_list) == 3
    assert any(
        feedback.feedback_type == "concern"
        and "Missing required documents" in feedback.comment
        for feedback in feedback_list
    )
    assert any(
        feedback.feedback_type == "critique"
        and feedback.to_agent == "taylor"
        and "Document consistency low" in feedback.comment
        for feedback in feedback_list
    )
    assert any(
        feedback.feedback_type == "request_revision"
        and "Final strategic approval is blocked" in feedback.comment
        for feedback in feedback_list
    )
    assert env._documents["STRATEGY"].status == "rejected"


def test_process_feedback_resolutions_marks_matching_feedback_as_resolved(make_env):
    """Addressed feedback should be marked resolved and tracked for rewards."""

    env = make_env()
    feedback = Feedback.create(
        from_agent="taylor",
        to_agent="elon",
        document_type="PRD",
        feedback_type="request_revision",
        comment="Add headers and expand overview details",
    )
    env._feedback = [feedback]
    action = _make_action(
        agent_id="elon",
        action_type="WRITE_PRD",
        content="# PRD\n\nAdd headers and expand overview details with milestones and acceptance criteria.",
        reasoning="Addressing feedback from Taylor with clearer headers and overview coverage.",
    )

    env._process_feedback_resolutions("elon", action)

    assert feedback.resolved is True
    assert feedback.addressed_by == "elon"
    assert feedback.resolution_timestamp.endswith("Z")
    assert env._feedback_resolved_this_step == [feedback]
    assert env._primary_feedback_addressed_this_step is True


def test_client_parses_feedback_and_status_fields_into_typed_models():
    """The client should deserialize typed observation fields, not leave raw dicts."""

    client = AgentenvEnv(base_url="http://localhost:8000")
    payload = {
        "observation": {
            "task_description": "Plan a chat app",
            "result": "ok",
            "reasoning": "moving forward",
            "current_agent": "elon",
            "step_number": 2,
            "total_steps": 6,
            "documents": {
                "PRD": {
                    "type": "PRD",
                    "content": "# PRD\nhello",
                    "author": "elon",
                    "created_at": "2026-04-08T00:00:00Z",
                    "updated_at": "2026-04-08T00:00:00Z",
                    "status": "draft",
                }
            },
            "feedback": [
                {
                    "from_agent": "taylor",
                    "to_agent": "elon",
                    "document_type": "PRD",
                    "feedback_type": "critique",
                    "comment": "Expand the requirements section",
                    "timestamp": "2026-04-08T00:00:00Z",
                    "resolution_timestamp": "",
                    "addressed_by": "",
                    "resolved": False,
                }
            ],
            "last_action_result": {
                "agent_id": "maya",
                "action_type": "SEARCH_MARKET",
                "result": "success",
                "message": "Research done",
                "timestamp": "2026-04-08T00:00:00Z",
                "resolved_feedback_count": 0,
                "primary_feedback_addressed": False,
                "feedback_generated_count": 1,
            },
            "current_state": {"status": "in_progress"},
            "document_status_summary": {"draft": 1, "in_review": 0, "approved": 0, "rejected": 0},
            "documents_awaiting_review": ["PRD (needs review)"],
            "errors": [],
            "step_count": 1,
        },
        "reward": 0.25,
        "done": False,
    }

    result = client._parse_result(payload)

    assert isinstance(result.observation.documents["PRD"], Document)
    assert isinstance(result.observation.feedback[0], Feedback)
    assert isinstance(result.observation.last_action_result, LastAction)
    assert result.observation.document_status_summary["draft"] == 1
    assert result.observation.documents_awaiting_review == ["PRD (needs review)"]


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_complete_workflow_generates_feedback_for_all_tasks(make_env, task_id):
    """Run the full six-agent workflow and verify feedback integration end to end."""

    env = make_env()
    task = TASKS[task_id]
    observation = env.reset(
        task_description=task.description,
        task_keywords=task.required_keywords,
        task_difficulty=task.difficulty,
        required_sections=task.required_sections,
        task_id=task_id,
    )

    workflow_actions = [
        ("maya", "SEARCH_MARKET", "Research"),
        ("elon", "WRITE_PRD", "PRD"),
        ("jordan", "WRITE_TRD", "TRD"),
        ("robert", "BREAK_INTO_TASKS", "Tasks"),
        ("taylor", "REVIEW_DOCUMENTS", "Validation"),
        ("sam", "APPROVE_STRATEGY", "Strategy"),
    ]

    for step_index, (agent_id, action_type, label) in enumerate(workflow_actions, start=1):
        observation = env.step(
            _make_action(
                agent_id=agent_id,
                action_type=action_type,
                content=_short_workflow_content(label, task.name),
            )
        )
        assert observation.errors == []
        assert observation.step_count == step_index

    assert observation.done is True
    assert {
        "RESEARCH",
        "PRD",
        "TRD",
        "ARCHITECTURE",
        "ROADMAP",
        "TASKS",
        "VALIDATION",
        "STRATEGY",
    } <= set(observation.documents)
    assert any(feedback.from_agent == "taylor" for feedback in observation.feedback)
    assert any(feedback.from_agent == "sam" for feedback in observation.feedback)
    assert observation.document_status_summary
    assert isinstance(observation.documents_awaiting_review, list)
    assert observation.last_action_result is not None
    assert observation.last_action_result.feedback_generated_count >= 1
    assert any(
        document.status in {"rejected", "in_review", "approved"}
        for doc_type, document in observation.documents.items()
        if doc_type in {"RESEARCH", "PRD", "TRD", "ARCHITECTURE", "ROADMAP", "TASKS"}
    )
