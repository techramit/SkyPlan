#!/usr/bin/env python3
"""
Verification script for the SkyPlan Reward System.

This script verifies the code structure and exports without running the full tests.
"""

import sys
from pathlib import Path

print("=" * 60)
print("SkyPlan Reward System Verification")
print("=" * 60)

# Check if files exist
print("\n1. Checking file existence...")
files_to_check = [
    "AgentEnv/reward.py",
    "AgentEnv/models.py",
    "AgentEnv/workflow.py",
    "AgentEnv/server/AgentEnv_environment.py",
    "AgentEnv/__init__.py",
]

base_path = Path(__file__).parent
all_exist = True
for file_path in files_to_check:
    full_path = base_path / file_path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {file_path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Some files are missing!")
    sys.exit(1)

# Check syntax
print("\n2. Checking Python syntax...")
import py_compile

syntax_ok = True
for file_path in files_to_check:
    full_path = base_path / file_path
    try:
        py_compile.compile(str(full_path), doraise=True)
        print(f"  ✓ {file_path}")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {file_path}: {e}")
        syntax_ok = False

if not syntax_ok:
    print("\n✗ Syntax errors found!")
    sys.exit(1)

# Check reward.py exports
print("\n3. Checking reward.py exports...")
reward_path = base_path / "AgentEnv" / "reward.py"
with open(reward_path) as f:
    reward_content = f.read()

exports_to_check = [
    "RewardCalculator",
    "RewardConfig",
    "StepReward",
    "EpisodeReward",
    "QualityScore",
    "TeamworkScore",
    "PenaltyScore",
    "ScoreNormalizer",
    "TeamworkBonusCalculator",
    "calculate_reward",
    "clear_reward_cache",
    "get_cache_size",
]

all_exports_found = True
for export in exports_to_check:
    found = f"class {export}" in reward_content or f"def {export}" in reward_content
    status = "✓" if found else "✗"
    print(f"  {status} {export}")
    if not found:
        all_exports_found = False

if not all_exports_found:
    print("\n✗ Some exports are missing!")
    sys.exit(1)

# Check reward calculator components
print("\n4. Checking reward calculator components...")
components_to_check = [
    "QualityBonusCalculator",
    "TeamworkBonusCalculator",
    "CompletionBonusCalculator",
    "PenaltyCalculator",
    "ScoreNormalizer",
    "RewardCache",
]

all_components_found = True
for component in components_to_check:
    found = f"class {component}" in reward_content
    status = "✓" if found else "✗"
    print(f"  {status} {component}")
    if not found:
        all_components_found = False

if not all_components_found:
    print("\n✗ Some components are missing!")
    sys.exit(1)

# Check environment integration
print("\n5. Checking environment integration...")
env_path = base_path / "AgentEnv" / "server" / "AgentEnv_environment.py"
with open(env_path) as f:
    env_content = f.read()

integration_checks = [
    ("RewardCalculator import", "from ..reward import RewardCalculator" in env_content or "from reward import RewardCalculator" in env_content),
    ("RewardCalculator in __init__", "self._reward_calculator = RewardCalculator" in env_content),
    ("calculate_step_reward call", "calculate_step_reward" in env_content),
    ("get_episode_reward method", "def get_episode_reward" in env_content),
]

all_integration_ok = True
for check_name, check_result in integration_checks:
    status = "✓" if check_result else "✗"
    print(f"  {status} {check_name}")
    if not check_result:
        all_integration_ok = False

if not all_integration_ok:
    print("\n✗ Some integration checks failed!")
    sys.exit(1)

# Check __init__.py exports
print("\n6. Checking __init__.py exports...")
init_path = base_path / "AgentEnv" / "__init__.py"
with open(init_path) as f:
    init_content = f.read()

init_exports_to_check = [
    "RewardCalculator",
    "RewardConfig",
    "StepReward",
    "EpisodeReward",
    "QualityScore",
    "TeamworkScore",
    "PenaltyScore",
    "ScoreNormalizer",
    "TeamworkBonusCalculator",
    "calculate_reward",
    "clear_reward_cache",
    "get_cache_size",
]

all_init_exports_found = True
for export in init_exports_to_check:
    found = export in init_content
    status = "✓" if found else "✗"
    print(f"  {status} {export}")
    if not found:
        all_init_exports_found = False

if not all_init_exports_found:
    print("\n✗ Some __init__.py exports are missing!")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ All verification checks passed!")
print("=" * 60)
print("\nReward System Components:")
print("  - Quality Bonus Calculator (0.0-0.3 per document)")
print("  - Teamwork Bonus Calculator (0.0-0.2 per step)")
print("  - Completion Bonus Calculator (0.0 or 0.3)")
print("  - Penalty Calculator (up to -0.3 per step)")
print("  - Score Normalizer (0.0-1.0 range)")
print("\nFeatures:")
print("  - LLM-based quality assessment (with fallback)")
print("  - Reference detection for collaboration scoring")
print("  - Caching for performance optimization")
print("  - Detailed feedback and breakdown")
print("\nIntegration:")
print("  - Environment uses RewardCalculator")
print("  - Episode reward includes completion bonus")
print("  - All components exported via __init__.py")
print("=" * 60)

sys.exit(0)
