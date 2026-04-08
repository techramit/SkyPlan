# SkyPlan

SkyPlan is a multi-agent OpenEnv environment for autonomous product planning. Six specialized agents collaborate to turn a product idea into a research summary, PRD, technical design, roadmap, task breakdown, validation report, and final strategy approval.

## Why this environment

Product planning is a real workflow teams perform before implementation. SkyPlan models that work directly: market research, requirement definition, architecture design, execution planning, validation, and executive approval. The environment rewards useful partial progress, tracks document quality, and now includes explicit feedback and document-status transitions across the workflow.

## Action and observation space

Actions are represented by `SkyPlanAction` in [models.py](/AgentEnv/models.py). Each action includes:
- `agent_id`: one of `maya`, `elon`, `jordan`, `robert`, `taylor`, `sam`
- `action_type`: a role-valid action such as `SEARCH_MARKET`, `WRITE_PRD`, or `APPROVE_STRATEGY`
- `reasoning`: the agent's rationale
- `content`: the document content or review output

Observations are represented by `SkyPlanObservation` in [models.py](/AgentEnv/models.py). Agents receive:
- task description and current phase
- shared planning documents with statuses
- feedback history and unresolved feedback
- last action result
- document status summary and documents awaiting review
- reward and done signals

## Workflow

The workflow order is defined in [workflow.py](/AgentEnv/workflow.py):
1. Maya researches the market and problem space.
2. Elon writes the PRD.
3. Jordan produces the TRD and architecture.
4. Robert creates the roadmap and task plan.
5. Taylor validates the package and issues structured feedback.
6. Sam provides strategic approval or requests revision.

Documents move through `draft -> in_review -> approved/rejected`, and feedback can be generated, targeted, and later resolved by downstream actions.

## Tasks

The benchmark ships with three graded tasks in [tasks.py](/AgentEnv/tasks.py):
- `easy_user_authentication`: simple authentication planning
- `medium_chat_app`: real-time chat application planning
- `hard_saas_platform`: multi-tenant SaaS platform planning

Each task includes deterministic grading inputs such as required keywords, required sections, and difficulty-specific expectations.

## Setup

```bash
cd AgentEnv
uv sync --extra dev
```

Run the server locally:

```bash
uv run --project AgentEnv server --port 8000
```

Validate the OpenEnv package:

```bash
cd AgentEnv
openenv validate
```

Run the baseline inference script from the repo root:

```bash
set HF_TOKEN=...
python inference.py
```

By default, `inference.py` runs all three tasks and emits the required `[START]`, `[STEP]`, and `[END]` lines for each episode. Set `SKYPLAN_TASK` to a specific task id to run a single task.

## Testing

Feedback integration and end-to-end workflow coverage live in [test_feedback_integration.py](/test_feedback_integration.py).

```bash
uv run --project AgentEnv pytest test_feedback_integration.py -q
```

## Baseline note

The baseline runner is reproducible through [inference.py](/inference.py), but model-dependent scores are not checked into the repo because they depend on the configured remote model and token. Use the command above to record the current baseline for your chosen `MODEL_NAME`.
