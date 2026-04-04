# Pre-Submission Checklist  — all must pass or you're disqualified

---

## HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()

## OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints

## Dockerfile builds

Automated docker build on the submitted repo

## Baseline reproduces

Run the submitted inference script — must complete without error and produce scores

## 3+ tasks with graders

Enumerate tasks, run each grader, verify scores in 0.0–1.0 range

## Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:  

API_BASE_URL   The API endpoint for the LLM.  
MODEL_NAME     The model identifier to use for inference. 
HF_TOKEN       Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project
Participants must use OpenAI Client for all LLM calls using above variables

## Infra Restrictions

Runtime of inference script should be less than 20min 
Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

## Validator

Run the pre-submission validation script before submitting