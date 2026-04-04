# Round 1 — Problem Statement

## The Task
Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.

---

## Key Requirements at a Glance
- Must simulate a real-world task (not games or toys)
- Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml
- Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)
- Meaningful reward function with partial progress signals
- Baseline inference script with reproducible scores
- Deploy to Hugging Face Spaces + working Dockerfile
- README with environment description, action/observation spaces, setup instructions