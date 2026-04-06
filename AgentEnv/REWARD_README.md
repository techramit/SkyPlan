# SkyPlan Reward System

The "Paycheck" system for evaluating agent work quality and collaboration.

## Overview

The reward system evaluates agent work across four dimensions:

1. **Quality Bonus** (0.0–0.3 per document): Document quality assessment
2. **Teamwork Bonus** (0.0–0.2 per step): Collaboration and reference detection
3. **Completion Bonus** (0.0 or 0.3): All-or-nothing for completing all documents
4. **Constraint Penalty** (-0.3–0.0 per step): Penalties for violations

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example` for template):

```bash
# LLM Configuration
SKYPLAN_LLM_API_KEY=your_api_key_here
SKYPLAN_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
SKYPLAN_LLM_MODEL=meta/llama-3.1-405b-instruct
SKYPLAN_LLM_TIMEOUT=10

# Reward System
SKYPLAN_QUALITY_BONUS_MAX=0.3
SKYPLAN_TEAMWORK_BONUS_MAX=0.2
SKYPLAN_COMPLETION_BONUS=0.3
SKYPLAN_CACHE_TTL_HOURS=24
```

### Programmatic Configuration

```python
from AgentEnv.reward import RewardCalculator, RewardConfig

# Create custom config
config = RewardConfig(
    QUALITY_BONUS_MAX=0.4,
    TEAMWORK_BONUS_MAX=0.25,
    COMPLETION_BONUS=0.35,
)

# Or load from environment
config = RewardConfig.from_env()

# Create calculator
calculator = RewardCalculator(
    config=config,
    use_llm=True,
    api_key="your-api-key",
)
```

## Usage

### Step Reward Calculation

```python
from AgentEnv.reward import RewardCalculator
from AgentEnv.models import SkyPlanAction, Document

calculator = RewardCalculator(use_llm=True)

action = SkyPlanAction(
    agent_id="maya",
    action_type="SEARCH_MARKET",
    reasoning="I need to research the market.",
    content="# Market Research\n\n## Overview\n...",
)

documents = {}

step_reward = calculator.calculate_step_reward(
    action=action,
    documents=documents,
    task_keywords=["authentication", "security"],
    task_difficulty="medium",
)

print(f"Quality Bonus: {step_reward.quality_bonus:.3f}")
print(f"Teamwork Bonus: {step_reward.teamwork_bonus:.3f}")
print(f"Penalty: {step_reward.penalty:.3f}")
print(f"Total: {step_reward.total:.3f}")
```

### Episode Reward Calculation

```python
# After all steps are complete
episode_reward = calculator.calculate_episode_reward(
    documents=documents,
    required_document_types=["RESEARCH", "PRD", "TRD", "ARCHITECTURE", "ROADMAP", "TASKS", "VALIDATION", "STRATEGY"],
)

print(f"Final Score: {episode_reward.final_score:.3f}")
print(f"Breakdown: {episode_reward.breakdown}")
```

### Convenience Function

```python
from AgentEnv.reward import calculate_reward

step_reward = calculate_reward(
    action=action,
    documents=documents,
    task_keywords=["authentication"],
    task_difficulty="easy",
)
```

## Components

### QualityBonusCalculator

Evaluates document quality across four dimensions:

- **Content Depth** (40%): Substantive analysis, evidence-backed claims
- **Structure** (30%): Headers, sections, logical flow
- **Relevance** (20%): On-topic, addresses requirements
- **Professionalism** (10%): Clear writing, appropriate tone

Supports both rule-based and LLM-based evaluation.

### TeamworkBonusCalculator

Evaluates collaboration quality:

- **Reference Detection**: Checks if current agent references previous work
- **Handoff Quality**: Evaluates acknowledgment and setup for next agent
- **Entity Extraction**: Identifies key terms for reference matching

### CompletionBonusCalculator

All-or-nothing bonus for completing all required documents.

### PenaltyCalculator

Penalizes constraint violations:

- **Length**: Too short or empty content
- **Structure**: Missing headers or lists
- **Content**: Missing keywords or sections
- **Role**: Actions not aligned with agent role
- **Contradictions**: Inconsistencies with previous work
- **Unrealistic Claims**: Impossible timelines or claims

### ScoreNormalizer

Normalizes raw scores to [0.0, 1.0] range.

## Caching

The reward system caches LLM evaluations to avoid redundant calls:

```python
from AgentEnv.reward import clear_reward_cache, get_cache_size

# Get cache size
size = get_cache_size()

# Clear cache
clear_reward_cache()
```

## Quality Dimensions

### Content Depth

Scored based on document length relative to task difficulty:

| Difficulty | Min Length | Target Length |
|------------|------------|---------------|
| Easy       | 100        | 300           |
| Medium     | 300        | 800           |
| Hard       | 800        | 2000          |

### Structure

Scored based on:
- Presence of headers (## or ###)
- Presence of lists (- or *)
- Multiple paragraphs
- Structural keywords (overview, summary, etc.)

### Relevance

Scored based on keyword presence in document content.

### Professionalism

Scored based on:
- Absence of unprofessional patterns (lol, haha, etc.)
- Proper sentence structure
- Clear, professional tone

## Teamwork Scoring

### Reference Levels

| Level | Reference Ratio | Bonus |
|-------|-----------------|-------|
| 0     | 0%              | 0.0   |
| 1     | < 30%           | 0.05  |
| 2     | 30-70%          | 0.1   |
| 3     | > 70%           | 0.2   |

### Handoff Quality

Evaluated based on:
- Acknowledgment phrases ("based on", "building on", etc.)
- Setup phrases ("next step", "recommend", etc.)

## Penalties

| Violation | Penalty |
|-----------|---------|
| Empty content | -0.2 |
| Too short | -0.1 |
| No headers | -0.05 |
| No lists | -0.05 |
| Missing section | -0.1 each |
| Missing keyword | -0.05 each |
| Contradiction | -0.1 |
| Unrealistic claim | -0.1 |
| Wrong role | -0.15 |

Maximum penalty per step: -0.3

## Final Score Calculation

```
Step Reward = Quality Bonus + Teamwork Bonus + Penalty
Episode Raw = Sum(Step Rewards) + Completion Bonus
Final Score = Normalize(Episode Raw, min_possible, max_possible)
```

Where:
- `min_possible` = PENALTY_MAX_PER_STEP × 6 = -1.8
- `max_possible` = (QUALITY_BONUS_MAX + TEAMWORK_BONUS_MAX) × 6 + COMPLETION_BONUS = 3.3

## LLM Integration

The reward system supports LLM-based quality assessment with automatic fallback to rule-based scoring.

### LLM Prompt

The LLM is prompted to evaluate on four dimensions:

1. Content Depth (0.0-1.0)
2. Structure (0.0-1.0)
3. Relevance (0.0-1.0)
4. Professionalism (0.0-1.0)

And returns a JSON object with scores and feedback.

### Fallback

If LLM fails, the system automatically falls back to rule-based scoring.

## Logging

The reward system uses Python's logging module:

```python
import logging

# Configure log level
logging.getLogger("AgentEnv.reward").setLevel(logging.DEBUG)
```

## Testing

Run the verification script:

```bash
python3 verify_reward.py
```

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
