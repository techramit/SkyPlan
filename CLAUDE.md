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
