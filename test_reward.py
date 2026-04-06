#!/usr/bin/env python3
"""
Test script for the SkyPlan Reward System.

This script tests the reward calculator with sample actions and documents.
"""

import sys
from pathlib import Path
import importlib.util

# Add AgentEnv to path
agentenv_path = str(Path(__file__).parent / "AgentEnv")
sys.path.insert(0, agentenv_path)

# Create mock modules for openenv
import types

openenv_mock = types.ModuleType("openenv")
openenv_core_mock = types.ModuleType("openenv.core")
openenv_types_mock = types.ModuleType("openenv.core.env_server.types")
openenv_interfaces_mock = types.ModuleType("openenv.core.env_server.interfaces")

# Create base classes
class Action:
    pass

class Observation:
    pass

class State:
    def __init__(self, episode_id: str, step_count: int):
        self.episode_id = episode_id
        self.step_count = step_count

class Environment:
    pass

# Add to mock modules
openenv_types_mock.Action = Action
openenv_types_mock.Observation = Observation
openenv_interfaces_mock.Environment = Environment
openenv_core_mock.env_server = types.ModuleType("openenv.core.env_server")
openenv_core_mock.env_server.types = openenv_types_mock
openenv_core_mock.env_server.interfaces = openenv_interfaces_mock
openenv_mock.core = openenv_core_mock

# Register mocks
sys.modules["openenv"] = openenv_mock
sys.modules["openenv.core"] = openenv_core_mock
sys.modules["openenv.core.env_server"] = openenv_core_mock.env_server
sys.modules["openenv.core.env_server.types"] = openenv_types_mock
sys.modules["openenv.core.env_server.interfaces"] = openenv_interfaces_mock

# Import modules directly by file path to avoid __init__.py
models_path = Path(__file__).parent / "AgentEnv" / "models.py"
reward_path = Path(__file__).parent / "AgentEnv" / "reward.py"
workflow_path = Path(__file__).parent / "AgentEnv" / "workflow.py"

# Load models module
spec = importlib.util.spec_from_file_location("models", models_path)
models = importlib.util.module_from_spec(spec)
sys.modules["models"] = models
spec.loader.exec_module(models)

# Load workflow module (needed by reward)
spec = importlib.util.spec_from_file_location("workflow", workflow_path)
workflow = importlib.util.module_from_spec(spec)
sys.modules["workflow"] = workflow
spec.loader.exec_module(workflow)

# Load reward module
spec = importlib.util.spec_from_file_location("reward", reward_path)
reward = importlib.util.module_from_spec(spec)
sys.modules["reward"] = reward
spec.loader.exec_module(reward)

# Get the classes we need
SkyPlanAction = models.SkyPlanAction
Document = models.Document
RewardCalculator = reward.RewardCalculator
StepReward = reward.StepReward
calculate_reward = reward.calculate_reward
clear_reward_cache = reward.clear_reward_cache
get_cache_size = reward.get_cache_size


def test_quality_bonus():
    """Test quality bonus calculation."""
    print("=" * 60)
    print("Test 1: Quality Bonus Calculation")
    print("=" * 60)

    action = SkyPlanAction(
        agent_id="maya",
        action_type="SEARCH_MARKET",
        reasoning="I need to research the market to understand the problem space.",
        content="""# Market Research

## Overview
This document provides an analysis of the current market for authentication systems.

## Key Findings
- 60% of users prefer passwordless authentication
- Security is the top concern for 80% of enterprises
- Biometric authentication is growing at 25% annually

## Competitors
- Auth0: Market leader with 40% share
- Okta: Strong enterprise presence
- Firebase: Popular for mobile apps

## Opportunities
- Passwordless authentication
- Biometric integration
- Multi-factor authentication
""",
    )

    documents = {}

    calculator = RewardCalculator(use_llm=False)
    step_reward = calculator.calculate_step_reward(
        action=action,
        documents=documents,
        task_keywords=["authentication", "security", "password"],
        task_difficulty="easy",
    )

    print(f"Quality Bonus: {step_reward.quality_bonus:.3f}")
    print(f"Teamwork Bonus: {step_reward.teamwork_bonus:.3f}")
    print(f"Penalty: {step_reward.penalty:.3f}")
    print(f"Total Step Reward: {step_reward.total:.3f}")

    if step_reward.quality_score:
        print(f"\nQuality Breakdown:")
        print(f"  Content Depth: {step_reward.quality_score.content_depth:.3f}")
        print(f"  Structure: {step_reward.quality_score.structure:.3f}")
        print(f"  Relevance: {step_reward.quality_score.relevance:.3f}")
        print(f"  Professionalism: {step_reward.quality_score.professionalism:.3f}")
        print(f"  Feedback: {step_reward.quality_score.feedback}")

    print()


def test_teamwork_bonus():
    """Test teamwork bonus calculation."""
    print("=" * 60)
    print("Test 2: Teamwork Bonus Calculation")
    print("=" * 60)

    # First, Maya creates research
    maya_action = SkyPlanAction(
        agent_id="maya",
        action_type="SEARCH_MARKET",
        reasoning="I need to research the market.",
        content="""# Market Research

## Key Findings
- 60% of users prefer passwordless authentication
- Security is the top concern for 80% of enterprises
""",
    )

    documents = {}
    calculator = RewardCalculator(use_llm=False)

    # Maya's step (no previous documents)
    maya_reward = calculator.calculate_step_reward(
        action=maya_action,
        documents=documents,
    )

    # Add Maya's document
    documents["RESEARCH"] = Document(
        type="RESEARCH",
        content=maya_action.content,
        author="maya",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        status="draft",
    )

    # Now Elon creates PRD, referencing Maya's research
    elon_action = SkyPlanAction(
        agent_id="elon",
        action_type="WRITE_PRD",
        reasoning="I need to write the PRD based on Maya's research.",
        content="""# Product Requirements Document

## Overview
Based on Maya's research showing that 60% of users prefer passwordless authentication,
we will build a passwordless authentication system.

## Features
- Passwordless login
- Biometric authentication
- Multi-factor authentication

## Security
Security is the top concern for 80% of enterprises, so we will prioritize
enterprise-grade security features.
""",
    )

    elon_reward = calculator.calculate_step_reward(
        action=elon_action,
        documents=documents,
    )

    print(f"Maya's Teamwork Bonus: {maya_reward.teamwork_bonus:.3f}")
    print(f"Elon's Teamwork Bonus: {elon_reward.teamwork_bonus:.3f}")

    if elon_reward.teamwork_score:
        print(f"\nElon's Teamwork Breakdown:")
        for doc_type, score in elon_reward.teamwork_score.references.items():
            print(f"  Reference to {doc_type}: {score:.3f}")
        print(f"  Handoff Quality: {elon_reward.teamwork_score.handoff_quality:.3f}")
        print(f"  Feedback: {elon_reward.teamwork_score.feedback}")

    print()


def test_penalty_calculation():
    """Test penalty calculation."""
    print("=" * 60)
    print("Test 3: Penalty Calculation")
    print("=" * 60)

    # Test with poor quality content
    action = SkyPlanAction(
        agent_id="elon",
        action_type="WRITE_PRD",
        reasoning="I need to write the PRD.",
        content="This is a PRD.",  # Too short, no structure
    )

    documents = {}
    calculator = RewardCalculator(use_llm=False)

    step_reward = calculator.calculate_step_reward(
        action=action,
        documents=documents,
        task_keywords=["authentication", "security"],
        required_sections=["overview", "features"],
    )

    print(f"Quality Bonus: {step_reward.quality_bonus:.3f}")
    print(f"Teamwork Bonus: {step_reward.teamwork_bonus:.3f}")
    print(f"Penalty: {step_reward.penalty:.3f}")
    print(f"Total Step Reward: {step_reward.total:.3f}")

    if step_reward.penalty_score:
        print(f"\nPenalty Breakdown:")
        for penalty_name, penalty_value in step_reward.penalty_score.penalties.items():
            print(f"  {penalty_name}: {penalty_value:.3f}")
        print(f"  Reasons: {step_reward.penalty_score.reasons}")

    print()


def test_episode_reward():
    """Test episode reward calculation."""
    print("=" * 60)
    print("Test 4: Episode Reward Calculation")
    print("=" * 60)

    calculator = RewardCalculator(use_llm=False)

    # Simulate a complete episode
    actions = [
        SkyPlanAction(
            agent_id="maya",
            action_type="SEARCH_MARKET",
            reasoning="Research the market.",
            content="# Market Research\n\n## Overview\nMarket analysis for authentication systems.",
        ),
        SkyPlanAction(
            agent_id="elon",
            action_type="WRITE_PRD",
            reasoning="Write the PRD.",
            content="# PRD\n\n## Overview\nProduct requirements for authentication system.",
        ),
        SkyPlanAction(
            agent_id="jordan",
            action_type="DESIGN_ARCHITECTURE",
            reasoning="Design the architecture.",
            content="# Architecture\n\n## Overview\nSystem architecture design.",
        ),
    ]

    documents = {}
    for i, action in enumerate(actions):
        step_reward = calculator.calculate_step_reward(
            action=action,
            documents=documents,
        )
        print(f"Step {i+1} ({action.agent_id}): {step_reward.total:.3f}")

        # Add document
        doc_type = "RESEARCH" if action.agent_id == "maya" else (
            "PRD" if action.agent_id == "elon" else "ARCHITECTURE"
        )
        documents[doc_type] = Document(
            type=doc_type,
            content=action.content,
            author=action.agent_id,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            status="draft",
        )

    # Calculate episode reward
    episode_reward = calculator.calculate_episode_reward(documents=documents)

    print(f"\nEpisode Summary:")
    print(f"  Step Count: {episode_reward.step_count}")
    print(f"  Total Raw: {episode_reward.total_raw:.3f}")
    print(f"  Completion Bonus: {episode_reward.completion_bonus:.3f}")
    print(f"  Final Score: {episode_reward.final_score:.3f}")
    print(f"\nBreakdown:")
    for key, value in episode_reward.breakdown.items():
        print(f"  {key}: {value:.3f}")

    print()


def test_cache():
    """Test caching functionality."""
    print("=" * 60)
    print("Test 5: Cache Functionality")
    print("=" * 60)

    action = SkyPlanAction(
        agent_id="maya",
        action_type="SEARCH_MARKET",
        reasoning="Research the market.",
        content="# Market Research\n\n## Overview\nMarket analysis for authentication systems.",
    )

    documents = {}

    # First call - should cache
    calculator = RewardCalculator(use_llm=False)
    step_reward_1 = calculator.calculate_step_reward(
        action=action,
        documents=documents,
    )

    cache_size_1 = get_cache_size()
    print(f"First call - Cache size: {cache_size_1}")

    # Second call with same content - should use cache
    calculator2 = RewardCalculator(use_llm=False)
    step_reward_2 = calculator2.calculate_step_reward(
        action=action,
        documents=documents,
    )

    cache_size_2 = get_cache_size()
    print(f"Second call - Cache size: {cache_size_2}")

    # Clear cache
    clear_reward_cache()
    cache_size_3 = get_cache_size()
    print(f"After clear - Cache size: {cache_size_3}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SkyPlan Reward System Tests")
    print("=" * 60 + "\n")

    try:
        test_quality_bonus()
        test_teamwork_bonus()
        test_penalty_calculation()
        test_episode_reward()
        test_cache()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
