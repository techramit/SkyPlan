"""Tests for the SkyPlan inference runner contract."""

import asyncio
from types import SimpleNamespace

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

    monkeypatch.setattr(inference, "load_env_file", lambda path=".env": None)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(inference.AgentenvEnv, "from_docker_image", staticmethod(fake_from_docker_image))
    monkeypatch.setattr(inference, "run_episode", fake_run_episode)
    monkeypatch.setattr(inference, "log_start", lambda **kwargs: events.append("start"))
    monkeypatch.setattr(inference, "log_end", lambda **kwargs: events.append("end"))
    monkeypatch.setattr(inference, "log_error", lambda message: events.append(f"error:{message}"))

    asyncio.run(inference.main(["--task", "easy_user_authentication"]))

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

    monkeypatch.setattr(inference, "load_env_file", lambda path=".env": None)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(inference.AgentenvEnv, "from_docker_image", staticmethod(fake_from_docker_image))
    monkeypatch.setattr(inference, "run_episode", fake_run_episode)
    monkeypatch.setattr(inference, "log_start", lambda **kwargs: events.append("start"))
    monkeypatch.setattr(inference, "log_end", lambda **kwargs: events.append("end"))
    monkeypatch.setattr(inference, "log_error", lambda message: events.append(f"error:{message}"))

    asyncio.run(inference.main(["--task", "easy_user_authentication"]))

    assert "end" in events
    assert events.index("close") < events.index("end")


def test_load_env_file_populates_missing_variables(tmp_path, monkeypatch):
    """Local .env loading should populate unset variables without overwriting explicit shell env."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=from-file\nMODEL_NAME=file-model\nSKYPLAN_TASK=medium_chat_app\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("SKYPLAN_TASK", "easy_user_authentication")

    inference.load_env_file(env_file)

    assert inference.get_runtime_config().api_key == "from-file"
    assert inference.get_runtime_config().model_name == "file-model"
    assert inference.get_runtime_config().task_selector == "easy_user_authentication"


def test_parse_cli_task_override_wins_over_environment(monkeypatch):
    """--task should override SKYPLAN_TASK for direct script invocation."""

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("SKYPLAN_TASK", "easy_user_authentication")

    runtime_config = inference.get_runtime_config("all")

    assert inference.resolve_task_ids(runtime_config.task_selector) == list(inference.TASKS.keys())


def test_runtime_config_uses_legacy_skyplan_endpoint_when_only_legacy_key_exists(monkeypatch):
    """Legacy SkyPlan inference credentials should resolve to the compatible endpoint."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("SKYPLAN_LLM_API_KEY", "legacy-token")
    monkeypatch.setenv("SKYPLAN_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("SKYPLAN_LLM_MODEL", "meta/llama-3.1-405b-instruct")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    runtime_config = inference.get_runtime_config()

    assert runtime_config.api_base_url == "https://integrate.api.nvidia.com/v1"
    assert runtime_config.api_key == "legacy-token"
    assert runtime_config.model_name == "meta/llama-3.1-405b-instruct"


def test_run_episode_fails_fast_when_model_request_fails(monkeypatch):
    """Model request errors should stop the episode and avoid fake-success results."""

    async def fake_reset(**kwargs):
        del kwargs
        return SimpleNamespace(observation=_make_observation())

    class FakeEnv:
        reset = staticmethod(fake_reset)

    runtime_config = inference.RuntimeConfig(
        api_base_url="https://router.huggingface.co/v1",
        model_name="test-model",
        api_key="test-token",
        image_name="skyplan-env",
        task_selector="easy_user_authentication",
        temperature=0.0,
        max_tokens=2000,
    )
    events: list[str] = []

    monkeypatch.setattr(
        inference,
        "get_model_message",
        lambda **kwargs: (_ for _ in ()).throw(inference.ModelRequestError("maya: 401 invalid")),
    )
    monkeypatch.setattr(inference, "log_error", lambda message: events.append(f"error:{message}"))
    monkeypatch.setattr(
        inference,
        "log_step",
        lambda **kwargs: events.append(
            f"step:{kwargs['action']}:{kwargs['reward']:.2f}:{kwargs['error']}"
        ),
    )

    result = asyncio.run(
        inference.run_episode(
            client=object(),
            env=FakeEnv(),
            task_id="easy_user_authentication",
            task_config=inference.TASKS["easy_user_authentication"],
            runtime_config=runtime_config,
        )
    )

    assert result == {"success": False, "steps": 1, "rewards": [0.0]}
    assert any("[model-error] maya: 401 invalid" in event for event in events)
    assert any("MODEL_REQUEST_FAILED" in event for event in events)
