# SkyPlan Inference Script

## Overview

The `inference.py` script is the "Master Switch" that runs the SkyPlan environment with LLM-powered agents. It manages the complete workflow from start to finish, emitting standardized output for hackathon evaluation.

## Usage

### Basic Usage

```bash
# Set environment variables
export HF_TOKEN="your-api-key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
export IMAGE_NAME="AgentEnv-env:latest"

# Run inference
python3 inference.py
```

### With Custom Configuration

```bash
# Set custom task
export SKYPLAN_TASK=medium_chat_app

# Set custom model
export MODEL_NAME="meta-llama/Llama-3.1-405B-Instruct"

# Run inference
python3 inference.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | API key for LLM | Required |
| `IMAGE_NAME` | Docker image name | `AgentEnv-env:latest` |
| `SKYPLAN_TASK` | Task to run | `easy_user_authentication` |
| `SKYPLAN_MAX_STEPS` | Max steps per episode | `6` |
| `SKYPLAN_TEMPERATURE` | LLM temperature | `0.0` |
| `SKYPLAN_MAX_TOKENS` | Max tokens per LLM call | `2000` |
| `SKYPLAN_SUCCESS_THRESHOLD` | Success score threshold | `0.5` |

## Available Tasks

| Task ID | Name | Difficulty |
|---------|------|------------|
| `easy_user_authentication` | Simple User Authentication | Easy |
| `medium_chat_app` | Real-time Chat Application | Medium |
| `hard_saas_platform` | Multi-tenant SaaS Platform | Hard |

## Output Format

The script emits standardized output in the following format:

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

### Output Rules

- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after `env.close()`, always emitted (even on exception)
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the raw last_action_error string, or `null` if none
- All fields on a single line with no newlines within a line
- Score in `[0.0, 1.0]`

### Example Output

```
[START] task=easy_user_authentication env=skyplan model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=maya:SEARCH_MARKET reward=0.15 done=false error=null
[STEP] step=2 action=elon:WRITE_PRD reward=0.25 done=false error=null
[STEP] step=3 action=jordan:DESIGN_ARCHITECTURE reward=0.20 done=false error=null
[STEP] step=4 action=robert:CREATE_ROADMAP reward=0.18 done=false error=null
[STEP] step=5 action=taylor:REVIEW_DOCUMENTS reward=0.12 done=false error=null
[STEP] step=6 action=sam:APPROVE_STRATEGY reward=0.10 done=true error=null
[END] success=true steps=6 score=0.833 rewards=0.15,0.25,0.20,0.18,0.12,0.10
```

## How It Works

1. **Initialization**: The script reads environment variables and initializes the OpenAI client
2. **Connection**: Connects to the SkyPlan environment (via Docker or local)
3. **Episode Start**: Logs `[START]` with task, environment, and model info
4. **Agent Loop**:
   - For each agent in the workflow (Maya → Elon → Jordan → Robert → Taylor → Sam):
     - Get current observation from environment
     - Build prompt for the agent with context and previous work
     - Get action from LLM (JSON format with action_type, reasoning, content)
     - Parse and validate the action
     - Step the environment with the action
     - Log `[STEP]` with action, reward, done status, and error
     - Check if episode is done
5. **Episode End**: Calculate final score and log `[END]` with results
6. **Cleanup**: Close the environment connection

## Agent Workflow

The script follows the predefined workflow order:

1. **Maya (Research Analyst)**: Market research, competitive analysis
2. **Elon (Product Manager)**: PRD, feature definition
3. **Jordan (Architect)**: Architecture, TRD, tech stack
4. **Robert (Execution Planner)**: Roadmap, tasks, timelines
5. **Taylor (Validator)**: Document review, consistency checks
6. **Sam (CEO)**: Strategic approval, final sign-off

## Error Handling

The script handles errors gracefully:

- **Model request failures**: Falls back to default action
- **Environment failures**: Logs error and exits with failure status
- **Invalid actions**: Logs error and continues with next agent
- **JSON parsing errors**: Falls back to free-form response parsing

## Scoring

The final score is calculated as:

```
score = sum(rewards) / number_of_steps
score = clamp(score, 0.0, 1.0)
```

A score of `0.5` or higher is considered successful.

## Troubleshooting

### Connection Issues

If you see connection errors:

1. Verify the environment is running: `docker ps | grep AgentEnv`
2. Check the image name matches: `docker images | grep AgentEnv`
3. Verify the API key is valid

### Model Issues

If you see model request failures:

1. Verify the model name is correct
2. Check your API key has sufficient credits
3. Try reducing `MAX_TOKENS` if you're hitting token limits

### Task Issues

If you see "Unknown task" errors:

1. Verify the task ID is one of:
   - `easy_user_authentication`
   - `medium_chat_app`
   - `hard_saas_platform`

## Development

### Running Locally

```bash
# Start the environment server
cd AgentEnv
uv run --project . server

# In another terminal, run inference
python3 inference.py
```

### Running with Docker

```bash
# Build the image
docker build -t AgentEnv-env:latest -f AgentEnv/server/Dockerfile .

# Run the container
docker run -p 8000:8000 AgentEnv-env:latest

# Run inference (in another terminal)
python3 inference.py
```

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
