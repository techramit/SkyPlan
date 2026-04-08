"""Tests for the SkyPlan inference runner contract."""

import asyncio

import inference
from AgentEnv.models import Document, SkyPlanObservation


def _make_observation() -> SkyPlanObservation:
    """Build a minimal observation for prompt-contract tests."""

    documents = {
        "RESEARCH": Document.create("RESEARCH", "# Research\n\n- auth\n\nDetailed research notes.", "maya"),
        "PRD": Document.create("PRD", "# PRD\n\n- auth\n\nDetailed product notes.", "elon"),
        "TRD": Document.create("TRD", "# TRD\n\n- auth\n\nDetailed technical notes.", "jordan"),
        "ROADMAP": Document.create("ROADMAP", "# Roadmap\n\n- auth\n\nDetailed roadmap notes.", "robert"),
        "TASKS": Document.create("TASKS", "# Tasks\n\n- auth\n\nDetailed task notes.", "robert"),
    }
    return SkyPlanObservation(
        task_description="Plan authentication",
        result="ok",
        reasoning="ready",
        current_agent="elon",
        step_number=2,
        total_steps=6,
        documents=documents,
        feedback=[
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
            },
            {
                "from_agent": "sam",
                "to_agent": "robert",
                "document_type": "ROADMAP",
                "feedback_type": "question",
                "comment": "Clarify milestone sequencing",
                "timestamp": "2026-04-08T00:00:00Z",
                "resolution_timestamp": "",
                "addressed_by": "",
                "resolved": False,
            },
        ],
        last_action_result=None,
        current_state={"phase": "product"},
        document_status_summary={"draft": 5, "in_review": 0, "approved": 0, "rejected": 0},
        documents_awaiting_review=[],
        errors=[],
        step_count=1,
        done=False,
        reward=0.0,
    )


def test_build_user_prompt_filters_context_to_relevant_docs_and_feedback():
    """Prompt construction should focus on relevant context instead of dumping everything."""

    prompt = inference.build_user_prompt(
        step=2,
        agent_id="elon",
        observation=_make_observation(),
        task_description="Plan authentication",
    )

    assert "RESEARCH (draft)" in prompt
    assert "ROADMAP (draft)" in prompt
    assert "TASKS (draft)" in prompt
    assert "PRD (draft)" not in prompt
    assert "TRD (draft)" not in prompt
    assert "Expand the requirements section" in prompt
    assert "Clarify milestone sequencing" not in prompt


def test_main_logs_end_after_env_close(monkeypatch):
    """Each episode should emit [END] only after the environment is closed."""

    events: list[str] = []

    class FakeEnv:
        async def close(self):
            events.append("close")

    async def fake_from_docker_image(image_name):
        del image_name
        events.append("create")
        return FakeEnv()

    async def fake_run_episode(client, env, task_id, task_config):
        del client, env, task_id, task_config
        events.append("run")
        return {"success": True, "steps": 1, "rewards": [0.5]}

    monkeypatch.setattr(inference, "HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(inference, "resolve_task_ids", lambda: ["easy_user_authentication"])
    monkeypatch.setattr(inference.AgentenvEnv, "from_docker_image", staticmethod(fake_from_docker_image))
    monkeypatch.setattr(inference, "run_episode", fake_run_episode)
    monkeypatch.setattr(inference, "log_start", lambda **kwargs: events.append("start"))
    monkeypatch.setattr(inference, "log_end", lambda **kwargs: events.append("end"))
    monkeypatch.setattr(inference, "log_error", lambda message: events.append(f"error:{message}"))

    asyncio.run(inference.main())

    assert events.index("close") < events.index("end")


def test_main_still_logs_end_when_episode_execution_raises(monkeypatch):
    """The inference loop should still emit an end line when an episode fails."""

    events: list[str] = []

    class FakeEnv:
        async def close(self):
            events.append("close")

    async def fake_from_docker_image(image_name):
        del image_name
        return FakeEnv()

    async def fake_run_episode(client, env, task_id, task_config):
        del client, env, task_id, task_config
        raise RuntimeError("episode failed")

    monkeypatch.setattr(inference, "HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(inference, "resolve_task_ids", lambda: ["easy_user_authentication"])
    monkeypatch.setattr(inference.AgentenvEnv, "from_docker_image", staticmethod(fake_from_docker_image))
    monkeypatch.setattr(inference, "run_episode", fake_run_episode)
    monkeypatch.setattr(inference, "log_start", lambda **kwargs: events.append("start"))
    monkeypatch.setattr(inference, "log_end", lambda **kwargs: events.append("end"))
    monkeypatch.setattr(inference, "log_error", lambda message: events.append(f"error:{message}"))

    asyncio.run(inference.main())

    assert "end" in events
    assert events.index("close") < events.index("end")
