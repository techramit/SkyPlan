# Agent System Overview
This project uses a multi-agent swarm to transform an idea into a validated execution plan (PRD, TRD, roadmap, and tasks).  
Each agent has a clear responsibility and defined actions.

---

## Agents

| Name | Role | Responsibility | Key Actions | Main Outputs |
|-----|-----|-----|-----|-----|
| Sam | CEO | Strategic direction and final approval | SET_DIRECTION, REVIEW_PLAN, APPROVE_STRATEGY, PRIORITIZE_OBJECTIVES, REQUEST_REVISION | Final strategic alignment |
| Elon | Product Manager | Defines product requirements and user value | WRITE_PRD, DEFINE_FEATURES, IDENTIFY_USER_PERSONA, PRIORITIZE_FEATURES, DEFINE_SUCCESS_METRICS | PRD, feature list |
| Maya | Research Analyst | Performs research and validates ideas | SEARCH_MARKET, ANALYZE_COMPETITORS, VALIDATE_PROBLEM, SUMMARIZE_INSIGHTS, IDENTIFY_OPPORTUNITIES | Research summary |
| Jordan | Architect | Designs system architecture and technical approach | DESIGN_ARCHITECTURE, SELECT_TECH_STACK, DEFINE_APIS, DESIGN_DATA_MODEL, WRITE_TRD | TRD, system architecture |
| Robert | Execution Planner | Converts plans into execution tasks and roadmap | CREATE_ROADMAP, BREAK_INTO_TASKS, PLAN_SPRINTS, ESTIMATE_TIMELINES, DEFINE_DEPENDENCIES | Roadmap, sprint backlog |
| Taylor | Validator | Reviews plans for quality and consistency | REVIEW_DOCUMENTS, CHECK_CONSISTENCY, VALIDATE_CLAIMS, IDENTIFY_RISKS, SCORE_PLAN | Validation report |

---

## Agent Workflow

1. **Maya (Research Analyst)** gathers research and validates the idea.
2. **Elon (Product Manager)** creates the PRD and defines product scope.
3. **Jordan (Architect)** designs the technical architecture and TRD.
4. **Robert (Execution Planner)** generates roadmap, milestones, and tasks.
5. **Taylor (Validator)** reviews outputs and flags issues.
6. **Sam (CEO)** reviews the full plan and approves the final strategy.

---

## Expected Artifacts

- Research Summary
- Product Requirements Document (PRD)
- Technical Requirements Document (TRD)
- Product Roadmap
- Task / Sprint Plan
- Validation Report