---
title: SkyPlan Environment Server
emoji: "\U0001F6F0\uFE0F"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# SkyPlan Environment

SkyPlan is an OpenEnv environment for autonomous product planning. It simulates a six-agent planning team that collaborates on research, product definition, architecture, execution planning, validation, and strategic approval.

## Core models

The typed OpenEnv models live in [models.py](/D:/Docs/Hackathons/Meta%20Hackathon%202k26/AgentEnv/models.py):
- `SkyPlanAction`: `agent_id`, `action_type`, `reasoning`, `content`
- `SkyPlanObservation`: task state, documents, feedback, last action result, status summary, reward, and done flag

The workflow is enforced by [workflow.py](/D:/Docs/Hackathons/Meta%20Hackathon%202k26/AgentEnv/workflow.py), and the environment implementation is in [server/AgentEnv_environment.py](/D:/Docs/Hackathons/Meta%20Hackathon%202k26/AgentEnv/server/AgentEnv_environment.py).

## Environment behavior

Each session starts with Maya and moves through this sequence:
1. Maya produces research.
2. Elon writes the PRD.
3. Jordan creates the TRD and architecture.
4. Robert creates the roadmap and task breakdown.
5. Taylor validates the package and generates structured feedback.
6. Sam approves the strategy or sends the plan back for revision.

Documents transition through `draft`, `in_review`, `approved`, and `rejected`, and the observation surface includes both the status summary and the documents still awaiting review.

## Tasks

Three benchmark tasks are defined in [tasks.py](/D:/Docs/Hackathons/Meta%20Hackathon%202k26/AgentEnv/tasks.py):
- `easy_user_authentication`
- `medium_chat_app`
- `hard_saas_platform`

## Local usage

Install dependencies:

```bash
uv sync --extra dev
```

Run the FastAPI server:

```bash
uv run --project . server --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate
```

Run the feedback workflow tests from the repo root:

```bash
uv run --project AgentEnv pytest test_feedback_integration.py -q
```

## Docker

Build:

```bash
docker build -t AgentEnv-env:latest -f server/Dockerfile .
```

Run:

```bash
docker run -p 8000:8000 AgentEnv-env:latest
```

## Baseline inference

The baseline runner is [../inference.py](/D:/Docs/Hackathons/Meta%20Hackathon%202k26/inference.py). It uses the OpenAI client, supports all three tasks, and emits the hackathon-required stdout line format.
