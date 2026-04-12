<h1 align="center">SkyPlan</h1>

<p align="center">
	<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0EA5E9,100:0467DF&height=140&section=header&text=SkyPlan&fontSize=54&fontColor=ffffff&fontAlignY=42&desc=Multi-Agent%20Autonomous%20Planning%20Benchmark&descAlignY=72&descSize=16" alt="SkyPlan banner" />
</p>

<p align="center">
	<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
	<a href="https://github.com/meta-pytorch/OpenEnv"><img src="https://img.shields.io/badge/OpenEnv-Compatible-0EA5E9" alt="OpenEnv Compatible" /></a>
	<a href="https://huggingface.co/spaces"><img src="https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Spaces" /></a>
	<a href="AboutHackathon/TASK.md"><img src="https://img.shields.io/badge/Meta-Hackathon%202k26-0467DF" alt="Meta Hackathon 2k26" /></a>
	<a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--Style-4CAF50" alt="BSD Style License" /></a>
</p>

<p align="center">
	<strong>Plan like a product org. Execute like an agent swarm.</strong><br/>
	<sub>SkyPlan is built on OpenEnv and designed for real planning workflows: research, product definition, architecture, execution, validation, and strategy.</sub>
</p>

---

## Quick Links

<p align="center">
	<a href="#what-skyplan-produces"><strong>Artifacts</strong></a> •
	<a href="#workflow"><strong>Workflow</strong></a> •
	<a href="#implementation-status"><strong>Status</strong></a> •
	<a href="#tasks-and-grading"><strong>Grading</strong></a> •
	<a href="#repository-structure"><strong>Structure</strong></a> •
	<a href="#local-setup"><strong>Setup</strong></a> •
	<a href="#run-inference"><strong>Inference</strong></a> •
	<a href="#testing"><strong>Testing</strong></a> •
	<a href="#deployment"><strong>Deploy</strong></a>
</p>

## What SkyPlan Produces

For each episode, agents collaboratively build and iterate on:

- Research summary
- Product Requirements Document (PRD)
- Technical Requirements Document (TRD)
- System architecture
- Roadmap and milestone plan
- Task and sprint breakdown
- Validation report
- Strategic approval decision

## Workflow

SkyPlan uses a role-based handoff pipeline:

1. Maya (Research Analyst)
2. Elon (Product Manager)
3. Jordan (Architect)
4. Robert (Execution Planner)
5. Taylor (Validator)
6. Sam (CEO)

The environment supports revision loops when quality gates are not met. Validation and strategy steps can route work back to targeted agents before episode completion.

## Implementation Status

Current codebase capabilities:

- Typed models for actions, observations, documents, and feedback in [AgentEnv/models.py](AgentEnv/models.py)
- Dynamic workflow orchestration and handoff rules in [AgentEnv/workflow.py](AgentEnv/workflow.py)
- Document lifecycle transitions: `draft -> in_review -> approved/rejected`
- Cross-agent feedback generation and resolution tracking
- Reward shaping for step quality and episode-level completion in [AgentEnv/reward.py](AgentEnv/reward.py)
- FastAPI/OpenEnv server implementation in [AgentEnv/server/AgentEnv_environment.py](AgentEnv/server/AgentEnv_environment.py)
- Judge-compatible inference runner and protocol logging in [AgentEnv/inference.py](AgentEnv/inference.py)

## Tasks and Grading

SkyPlan ships with three graded tasks in [AgentEnv/tasks.py](AgentEnv/tasks.py):

- `easy_user_authentication`
- `medium_chat_app`
- `hard_saas_platform`

Grading combines:

- Artifact completeness
- Document structure quality
- Keyword and requirement relevance
- Agent-specific quality criteria
- Cross-document consistency checks

## Repository Structure

```text
SkyPlan/
├── AgentEnv/
│   ├── client.py
│   ├── inference.py
│   ├── models.py
│   ├── prompts.py
│   ├── reward.py
│   ├── tasks.py
│   ├── workflow.py
│   └── server/
│       ├── AgentEnv_environment.py
│       └── app.py
├── inference.py
├── test_feedback_integration.py
├── test_grading_quality.py
└── test_inference_contract.py
```

## Local Setup

```bash
cd AgentEnv
uv sync --extra dev
```

Start the environment server:

```bash
uv run --project AgentEnv server --host 0.0.0.0 --port 8000
```

Validate OpenEnv packaging and spec:

```bash
cd AgentEnv
openenv validate
```

## Run Inference

From repository root:

```bash
export HF_TOKEN=your_token_here
python inference.py
```

Useful runtime environment variables:

- `HF_TOKEN` or `API_KEY` (required)
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME`
- `SKYPLAN_TASK` (`all` by default)

Protocol output format used by the hackathon evaluator:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## Testing

Run the full test suite:

```bash
uv run --project AgentEnv pytest -q
```

Targeted suites:

- [test_feedback_integration.py](test_feedback_integration.py)
- [test_grading_quality.py](test_grading_quality.py)
- [test_inference_contract.py](test_inference_contract.py)

## Deployment

Deploy to Hugging Face Spaces through OpenEnv:

```bash
cd AgentEnv
openenv push
```

## License

This project is released under the BSD-style license in [LICENSE](LICENSE).
