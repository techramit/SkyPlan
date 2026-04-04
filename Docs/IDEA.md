# IDEA.md

## Overview
SkyPlan is a **multi-agent autonomous planning system** designed for vibecoders.  
Instead of manually planning projects, users provide an idea and the system generates structured planning documents such as research, product requirements, technical design, architecture, roadmap, and validation reports.

The system operates as an **OpenEnv multi-agent environment** where specialized agents collaborate to convert a raw idea into a clear execution plan.

---

## The Core Idea
Many vibecoders jump directly into coding without structured planning.  
This often results in:

- unclear product scope
- weak technical design
- unrealistic timelines
- wasted development effort

SkyPlan solves this by using **multiple AI agents with specialized roles** to simulate a small startup planning team.

Each agent focuses on a specific responsibility, and together they produce a full project plan.

---

## The Agents

| Name | Role | Responsibility |
|-----|-----|-----|
| Sam | CEO | Defines strategy and approves the final plan |
| Elon | Product Manager | Creates the PRD and defines product features |
| Maya | Research Analyst | Performs research and validates the idea |
| Jordan | Architect | Designs system architecture and technical approach |
| Robert | Execution Planner | Converts plans into roadmap and tasks |
| Taylor | Validator | Reviews outputs and checks consistency |

Each agent produces artifacts that are used by the next agent in the pipeline.

---

## The Workflow

1. User submits a project idea.
2. **Maya (Research)** analyzes the idea and gathers insights.
3. **Elon (Product Manager)** defines product requirements and features.
4. **Jordan (Architect)** designs the technical architecture.
5. **Robert (Execution Planner)** creates roadmap and task breakdown.
6. **Taylor (Validator)** reviews the plan for issues.
7. **Sam (CEO)** approves the final strategy.

---

## System Outputs

SkyPlan generates structured planning files:

- `RESEARCH.md`
- `PRD.md`
- `TRD.md`
- `ARCHITECTURE.md`
- `ROADMAP.md`
- `VALIDATION.md`

These files form a **complete execution blueprint** for building the project.

---

## Why This Matters
SkyPlan acts like a **virtual startup planning team** for vibecoders.

Instead of spending hours planning manually, developers can instantly generate structured plans and focus on execution.

---

## In Summary
SkyPlan transforms a simple idea into a **complete development plan** using a multi-agent system that simulates real product planning roles.