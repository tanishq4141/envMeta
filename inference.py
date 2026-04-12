import asyncio
import os
import textwrap
import json
from typing import List, Optional, Dict, Any

from openai import OpenAI
from models import LegalAction
from environment import LegalDocumentReviewEnv

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["easy_clause_detection", "medium_risk_analysis", "hard_full_review"]
BENCHMARK = os.getenv("LEGAL_ENV_BENCHMARK", "legal_doc_review")
MAX_STEPS = 15
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a legal AI agent. You interact with an environment.
    Rules:
    You can output ONLY a valid JSON object representing the action you want to take.
    Valid Action Types and Payloads:
    1. {"action_type": "find_risk_clause", "payload": {"clause": "<clause_name>"}}
    2. {"action_type": "suggest_edit", "payload": {"clause": "<clause_name>", "suggestion": "<text>"}}
    3. {"action_type": "classify_contract", "payload": {"contract_type": "<type>"}}
    4. {"action_type": "approve", "payload": {}}
    5. {"action_type": "reject", "payload": {}}

    Output only JSON without markdown wrappers.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, doc_text: str, current_obs: Dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Contract Text:
        {doc_text}

        Current Observation:
        {json.dumps(current_obs, indent=2)}

        Previous steps:
        {history_block}
        Send your next action as JSON.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, doc_text: str, current_obs: Dict, history: List[str]) -> Dict:
    user_prompt = build_user_prompt(step, doc_text, current_obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        if text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "reject", "payload": {}}

async def run_task(client: OpenAI, task_name: str) -> None:
    env = LegalDocumentReviewEnv()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Use the new OpenEnv-compatible reset signature
        obs = env.reset(task_name=task_name)
        obs_dict = {
            "document_text": obs.document_text,
            "contract_type": obs.contract_type,
            "identified_risks": obs.identified_risks,
            "suggestions": obs.suggestions,
            "status": obs.status,
            "feedback": obs.feedback
        }
        doc_text = obs.document_text

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_dict = get_model_action(client, step, doc_text, obs_dict, history)
            action_str = f"{action_dict.get('action_type', 'unknown')}({json.dumps(action_dict.get('payload', {}))})"

            try:
                legal_action = LegalAction(**action_dict)
                obs = env.step(legal_action)

                obs_dict = {
                    "document_text": obs.document_text,
                    "contract_type": obs.contract_type,
                    "identified_risks": obs.identified_risks,
                    "suggestions": obs.suggestions,
                    "status": obs.status,
                    "feedback": obs.feedback
                }
                reward = obs.reward if obs.reward is not None else 0.0
                done = obs.done
                error = None
            except Exception as set_err:
                reward = 0.0
                done = True
                error = str(set_err)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                score = reward  # final cumulative reward is in the last observation
                break

        if not obs.done:
            score = obs.reward if obs.reward is not None else 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_name in TASKS:
        await run_task(client, task_name)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
