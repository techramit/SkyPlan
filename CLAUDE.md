# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Meta Hackathon 2k26 submission for **SkyPlan** - a multi-agent autonomous planning copilot for vibecoders. The project implements an OpenEnv environment where AI agents collaborate to transform an idea into structured planning documents (PRD, TRD, roadmap, tasks).

The environment simulates a real-world task: autonomous product planning using a multi-agent swarm. Six specialized agents (Sam/CEO, Elon/Product Manager, Maya/Research Analyst, Jordan/Architect, Robert/Execution Planner, Taylor/Validator) collaborate to produce planning artifacts.

## Architecture

### OpenEnv Environment Structure

The project follows the OpenEnv framework pattern with client-server separation:

```
AgentEnv/
├── models.py              # Action, Observation dataclasses (typed models)
├── client.py              # AgentenvEnv client (EnvClient subclass)
├── openenv.yaml           # Environment manifest
├── pyproject.toml         # Dependencies and package config
└── server/
    ├── AgentEnv_environment.py  # Core environment logic (Environment subclass)
    ├── app.py             # FastAPI application (HTTP + WebSocket)
    ├── Dockerfile         # Container image definition
    └── requirements.txt   # Server dependencies
```

### Key Components

**Server-side (`AgentEnv_environment.py`)**:
- `AgentenvEnvironment(Environment)`: Implements `reset()`, `step(action)`, `state` property
- `SUPPORTS_CONCURRENT_SESSIONS: bool = True`: Enables multiple WebSocket sessions
- Maintains `State(episode_id, step_count)` for episode tracking

**Client-side (`client.py`)**:
- `AgentenvEnv(EnvClient)`: Async client with WebSocket connection
- `_step_payload(action)`: Convert Action to JSON
- `_parse_result(payload)`: Parse server response to StepResult
- `_parse_state(payload)`: Parse state response
- `from_docker_image()`: Auto-start container and connect

**Models (`models.py`)**:
- `AgentenvAction(Action)`: Input dataclass with Pydantic Fields
- `AgentenvObservation(Observation)`: Output dataclass with Pydantic Fields

### Agent System

Six agents collaborate in a defined workflow:
1. Maya (Research Analyst) → Research Summary
2. Elon (Product Manager) → PRD, feature list
3. Jordan (Architect) → TRD, system architecture
4. Robert (Execution Planner) → Roadmap, sprint backlog
5. Taylor (Validator) → Validation report
6. Sam (CEO) → Final strategic approval

Each agent has defined actions (e.g., `WRITE_PRD`, `DESIGN_ARCHITECTURE`, `CREATE_ROADMAP`).

## Development Commands

### Local Development

```bash
# Install environment in editable mode
cd AgentEnv
pip install -e .

# Or using uv (faster)
uv pip install -e .

# Run server locally (from AgentEnv directory)
uv run server --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn server.app:app --reload
```

### Docker

```bash
# Build Docker image
docker build -t AgentEnv-env:latest -f server/Dockerfile .

# Run container
docker run -p 8000:8000 AgentEnv-env:latest
```

### Hugging Face Deployment

```bash
# Deploy to Hugging Face Spaces
cd AgentEnv
openenv push

# Or with options
openenv push --repo-id my-org/skyplan --private
```

### Testing

```bash
# Run environment directly (no HTTP server)
python3 server/AgentEnv_environment.py

# Run tests (if pytest is installed)
pytest tests/ -v
```

### OpenEnv Commands

#### Usage: openenv [OPTIONS] COMMAND [ARGS]...

OpenEnv - An e2e framework for creating, deploying and using isolated execution environments for agentic RL training

#### Options
```bash
# Install completion for the current shell. 
openenv --install-completion

# Show completion for the current shell, to copy it or customize the installation.
openenv --show-completion

# Show help message and exit.
openenv --help

```

#### Commands
```bash
# Initialize a new OpenEnv environment
openenv init

# Build Docker images for OpenEnv environments
openenv build

# Validate environment structure and deployment readiness
openenv validate

# Push an OpenEnv environment to Hugging Face Spaces or custom registry
openenv push

# Serve environments locally (TODO: Phase 4)
openenv serve

# Fork (duplicate) a Hugging Face Space to your account
openenv fork

# Manage OpenEnv skills for AI assistants
openenv skills
```

## Hackathon Requirements

### Mandatory Submission Checklist

1. **HF Space deploys**: Space must return 200 and respond to `reset()`
2. **OpenEnv spec compliance**: Validate `openenv.yaml`, typed models, `step()/reset()/state()` endpoints
3. **Dockerfile builds**: Automated docker build must succeed
4. **Baseline reproduces**: Inference script must complete and produce scores
5. **3+ tasks with graders**: Minimum 3 tasks (easy → medium → hard) with scores 0.0–1.0
6. **Runtime < 20min**: Inference script must complete within 20 minutes
7. **Resource limits**: Must run on vcpu=2, memory=8gb

### Inference Script Requirements

The inference script (`inference.py` in project root) must:
- Use OpenAI Client for all LLM calls
- Emit stdout in exact format:
  ```
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
  ```
- Use environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `IMAGE_NAME`

### Evaluation Criteria

- **Real-world utility (30%)**: Does it model a genuine task?
- **Task & grader quality (25%)**: Well-defined tasks, accurate graders, difficulty progression
- **Environment design (20%)**: Clean state management, sensible action/observation spaces
- **Code quality & spec compliance (15%)**: Follows OpenEnv spec, clean structure
- **Creativity & novelty (10%)**: Novel domain, interesting mechanics

## Key Design Patterns

### OpenEnv Environment Implementation

When implementing the environment:
1. Subclass `Environment` from `openenv.core.env_server.interfaces`
2. Implement `reset()` → returns initial `Observation`
3. Implement `step(action)` → returns resulting `Observation`
4. Implement `state` property → returns `State(episode_id, step_count)`
5. Set `SUPPORTS_CONCURRENT_SESSIONS = True` for multi-client support

### Client Implementation

When implementing the client:
1. Subclass `EnvClient[Action, Observation, State]`
2. Implement `_step_payload(action)` → convert Action to JSON
3. Implement `_parse_result(payload)` → parse JSON to StepResult
4. Implement `_parse_state(payload)` → parse JSON to State

### Models

Use Pydantic Fields for type safety:
```python
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class MyAction(Action):
    field_name: str = Field(..., description="Description")
```

## Reference Repository: OpenOfficeRL

**Repository**: https://github.com/bvsbharat/OpenOfficeRL

This is a similar multi-agent OpenEnv project that can be used as a reference for implementing SkyPlan.

### Key Patterns to Adapt

**Multi-Agent System:**
- 7 agents with role-specific actions (CEO, Dev, Marketing, Sales, Content, HR, Customer)
- Asymmetric observations - each agent sees different data based on their role
- LLM-powered agents with role-specific system prompts

**Environment Structure:**
- 90 simulated days, 4 phases per day (morning_standup, execution, review, planning)
- Event engine for scenario-specific events
- 5 scenarios: Baseline GTM Launch, Competitor Launch, Series A Pressure, Churn Spike, Viral Moment

**Reward Function (6 components):**
- Pipeline stage rewards, KPI delta rewards, action rewards
- Collaboration bonuses, constraint penalties, base shaping

**Training Pipeline:**
- TRL + Unsloth with GRPO + LoRA
- Base model: Qwen 2.5 14B
- Trajectory collector for capturing agent turns

### Reference Files

| File | Purpose |
|------|---------|
| `office_os_environment.py` | Core env with reset/step/state |
| `models.py` | Action/Observation dataclasses |
| `market/config.py` | Agent roles, actions, pipeline stages |
| `market/simulator.py` | Action execution logic |
| `market/metrics.py` | Reward calculation |
| `agents/llm_agent.py` | LLM-powered agent implementation |
| `agents/prompts.py` | Role-specific system prompts |

### For SkyPlan Adaptation

Replace the office simulation with:
- **6 Planning Agents**: Maya (Research), Elon (PM), Jordan (Architect), Robert (Execution), Taylor (Validator), Sam (CEO)
- **Planning Pipeline**: Research → PRD → TRD → Roadmap → Tasks → Validation
- **3 Tasks with Graders**: Easy (simple feature), Medium (complex feature), Hard (multi-product)
- **Reward Function**: Document quality scores, collaboration bonuses, constraint penalties

## Important Notes

- The current `AgentEnv` is a template/echo environment - needs to be replaced with the actual SkyPlan multi-agent planning environment
- The environment must implement at least 3 tasks with graders (easy, medium, hard)
- Rewards must be in 0.0–1.0 range with meaningful partial progress signals
- All environment-specific dependencies go in `pyproject.toml`
- Server dependencies go in `server/requirements.txt`

## Session Context & Implementation Status

### Current Implementation (As of 2026-04-08)

**Completed Components:**

1. **Models (`AgentEnv/models.py`)** - Fully implemented with:
- `SkyPlanAction`: agent_id, action_type, reasoning, content
- `SkyPlanObservation`: task_description, result, reasoning, current_agent, step_number, total_steps, documents, feedback, last_action_result, current_state, errors, step_count
- `AgentId` enum: 6 agents (maya, elon, jordan, robert, taylor, sam) with workflow order
- `ActionType` enum: 30 action types across 6 categories
- `DocumentType` enum: 8 document types (RESEARCH, PRD, TRD, ARCHITECTURE, ROADMAP, TASKS, VALIDATION, STRATEGY)
- `Document` model: type, content, author, created_at, updated_at, status
- `Feedback` model: from_agent, to_agent, document_type, feedback_type, comment, timestamp, resolved
- `LastAction` model: agent_id, action_type, result, message, timestamp
- `ActionResult` enum: success, failure, partial, rejected, pending
- `FeedbackType` enum: suggestion, critique, question, approval, concern, request_revision
- Configuration classes: `ValidationConfig`, `RewardConfig`, `WorkflowConfig`
- `ACTION_TO_DOCUMENT` mapping: 30 actions → 8 document types

2. **Environment (`AgentEnv/server/AgentEnv_environment.py`)** - Fully implemented as `SkyPlanEnvironment`:
- `SUPPORTS_CONCURRENT_SESSIONS = True`
- `reset()`: Initializes new episode, sets Maya as first agent, returns initial observation
- `step(action)`: Validates action, files document, calculates reward, moves to next agent
- `state` property: Returns `State(episode_id, step_count)`
- Private methods: `_is_valid_agent()`, `_validate_action()`, `_file_document()`, `_calculate_reward()`, `_calculate_structure_bonus()`, `_build_observation()`, `_create_error_observation()`, `_get_current_phase()`

3. **Tasks & Graders (`AgentEnv/tasks.py`)** - Fully implemented:
- **3 tasks**: easy_user_authentication, medium_chat_app, hard_saas_platform
- Comprehensive grading system with rule-based and LLM-based scoring
- Completeness checks, content quality, structure validation, keyword relevance
- Composite consistency checks between documents
- Agent-specific quality criteria
- Detailed checklists for each agent per task

4. **Agent Prompts (`AgentEnv/prompts.py`)** - Fully implemented:
- 867 lines of detailed system prompts
- Professional identity, philosophy, and instructions for each agent
- Quality standards, collaboration guidelines, common pitfalls

5. **Inference Script (`inference.py`)** - Fully implemented:
- Complete async episode loop with `run_episode()`
- Full model integration via OpenAI client
- Proper logging format for judges
- Error handling and fallbacks
- 517 lines, production-ready

6. **Server (`AgentEnv/server/app.py`)** - Updated to use `SkyPlanEnvironment`

7. **Client (`AgentEnv/client.py`)** - Uses `SkyPlanAction`/`SkyPlanObservation`

**Still To Build:**

- Document status transitions (draft → in_review → approved/rejected)
- Feedback system integration (feedback model exists but workflow integration partial)
- Client update for new model fields (may need validation)
- End-to-end testing across all 3 tasks
- Deployment to Hugging Face Space

### Key Design Decisions

**Observation Structure ("Briefing Status Report"):**
- `task_description`: The work order/goal
- `result`: What happened
- `reasoning`: Why it happened
- `current_agent`: Who's next (e.g., "maya")
- `step_number`: Progress (e.g., Step 4 of 10)
- `total_steps`: Total steps
- `documents`: Shared folder (dict of Document objects)
- `feedback`: Peer reviews/critiques
- `last_action_result`: Success/failure check
- `current_state`: Document state
- `errors`: Any issues
- `step_count`: Episode step count

**Action Structure:**
- `agent_id`: Who is working (maya, elon, jordan, robert, taylor, sam)
- `action_type`: What action (30 types across 6 categories)
- `reasoning`: Why they took this action (required)
- `content`: The work output

**Workflow Order:**
Maya (Research) → Elon (PM) → Jordan (Architect) → Robert (Planner) → Taylor (Validator) → Sam (CEO)

**Reward Calculation (0.0–1.0):**
- Base reward: 0.1
- Content length bonus: up to 0.3
- Reasoning quality bonus: up to 0.2
- Document structure bonus: up to 0.4 (headers, lists, paragraphs, keywords)

**Configuration Classes (No Hardcoding):**
- `ValidationConfig`: MIN_CONTENT_LENGTH, MIN_REASONING_LENGTH
- `RewardConfig`: BASE_REWARD, CONTENT_LENGTH_WEIGHT, CONTENT_LENGTH_TARGET, REASONING_WEIGHT, REASONING_TARGET, STRUCTURE_WEIGHT, MAX_REWARD
- `WorkflowConfig`: PHASES, DEFAULT_TOTAL_STEPS
- `ACTION_TO_DOCUMENT`: 30 action → document mappings
