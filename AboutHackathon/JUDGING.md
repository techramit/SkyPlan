# How Judging works

---

## Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

## Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

## Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

## Disqualification Criteria

* Environment does not deploy or respond
* Plagiarized or trivially modified existing environments
* Graders that always return the same score
* No baseline inference script