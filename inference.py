import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# Environment variables
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-chat-hf")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "triage")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
You are assisting in a healthcare triage environment.
Each turn you must suggest a decision for the current patient.
Decisions can be: "Healthy", "Monitor", "Refer", "RequestTest".
Your goal is to maximize reward by making clinically appropriate decisions.
Reply with exactly one decision string — no quotes, no prefixes.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, patient: dict, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    vitals = f"age={patient['age']}, hb={patient['hemoglobin']}, bp={patient['blood_pressure']}, symptoms={patient['symptoms']}"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val} patient=({vitals})", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, patient: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
    Step: {step}
    Patient: age={patient['age']}, hemoglobin={patient['hemoglobin']}, bp={patient['blood_pressure']}, symptoms={patient['symptoms']}
    Last reward: {last_reward:.2f}
    Previous steps:
    {history_block}
    Suggest next decision (Healthy, Monitor, Refer, RequestTest).
    """).strip()

def get_model_decision(client: OpenAI, step: int, patient: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, patient, last_reward, history)
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
        return text if text else "Monitor"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "Monitor"

async def main() -> None:
    if not API_KEY:
        raise RuntimeError("No API key found. Please set HF_TOKEN or OPENAI_API_KEY.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MyEnvV4Env()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        state = result["state"]
        patients = state["patients"]
        current_index = state["current_index"]
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            patient = patients[current_index]
            decision = get_model_decision(client, step, patient, last_reward, history)
            action = MyEnvV4Action(decision=decision)

            result = env.step(action)
            reward = result["reward"]
            done = result["done"]
            state = result["state"]
            patients = state["patients"]
            current_index = state["current_index"]

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=decision, reward=reward, done=done, patient=patient, error=None)
            history.append(f"Step {step}: decision={decision} -> reward {reward:+.2f}")

            if done:
                break

        max_total_reward = MAX_STEPS * 1.0  # rewards normalized around 1.0
        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())