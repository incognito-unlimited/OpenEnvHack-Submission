"""
inference.py — Email Triage OpenEnv Baseline Inference Script
=============================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face API key (required, no default)
    ENV_BASE_URL   URL for the Email Triage env server (default: http://localhost:7860)
    LOCAL_IMAGE_NAME Optional local image name if using from_docker_image()

STDOUT FORMAT:
    [START] task=<task_name> env=email-triage model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import EmailTriageEnv
from models import TriageAction
from tasks_data import ALL_TASK_NAMES

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "email-triage"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STEPS_PER_TASK = 12  # Safety cap

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant. When given an email, respond ONLY with a
    valid JSON object and nothing else. No markdown, no explanation, no backticks.

    Think through the decision internally before answering, but do NOT reveal any scratchpad,
    chain-of-thought, or reasoning text. The final response must be JSON only.

    Required JSON schema:
    {
      "category": "<spam|work|personal|newsletter|urgent|social>",
      "priority": <integer 1-5>,
      "action": "<delete|archive|reply|forward|flag|mark_read>",
      "response_draft": "<string or null>"
    }

    Guidelines:
    - category: classify the email's nature accurately
      * spam: unsolicited/phishing/scam emails
      * urgent: time-sensitive, requires immediate attention, high business impact
      * work: professional emails that don't require immediate action
      * personal: emails from friends/family
      * newsletter: subscription content, digests, marketing
      * social: social media notifications
    - priority: 1=ignore, 2=low, 3=medium, 4=high, 5=critical/immediate
    - action: most appropriate response
      * delete: spam, unwanted
      * archive: newsletters, social, low-value
      * reply: emails expecting a response from you
      * forward: emails that should go to someone else
      * flag: important emails to revisit later
      * mark_read: informational emails, no action needed
    - response_draft: REQUIRED if action is "reply" or "forward". Write a professional,
      concise response (2-4 sentences) addressing the main points. Otherwise set to null.

        Boundary rules:
        - If an email contains a deadline within 24 hours, the category MUST be urgent and the
            priority MUST be 5.
        - If the category is spam, the priority MUST be 1 and the action MUST be delete.
        - Automated system updates are work, not newsletter.

        Examples:
        Email: "Hey, are we still on for lunch today?"
        Correct Output: {"category":"personal","priority":3,"action":"reply","response_draft":"Sure, lunch still works for me today. See you then."}

        Email: "Your account will be suspended in 12 hours unless you verify immediately."
        Correct Output: {"category":"urgent","priority":5,"action":"flag","response_draft":null}

        Email: "Weekly product update: deployment completed successfully."
        Correct Output: {"category":"work","priority":2,"action":"mark_read","response_draft":null}
""").strip()


def build_user_prompt(observation_dict: dict) -> str:
    email = observation_dict["email"]
    return textwrap.dedent(f"""
        Email #{observation_dict['step_number']} of {observation_dict['context'].get('total_emails', '?')}
        Inbox remaining after this: {observation_dict['inbox_remaining']}

        From: {email['sender']}
        Subject: {email['subject']}
        Timestamp: {email['timestamp']}
        Has Attachment: {email['has_attachment']}
        Thread Length: {email['thread_length']} email(s)

        Body:
        {email['body']}

        Respond with ONLY a JSON object matching the schema. No other text.
    """).strip()


def call_llm(client: OpenAI, user_prompt: str) -> dict:
    """Call the LLM and parse the triage JSON response."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent decisions
            max_tokens=300,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a safe default action
        return {"category": "work", "priority": 2, "action": "mark_read", "response_draft": None}
    except Exception:
        return {"category": "work", "priority": 2, "action": "mark_read", "response_draft": None}


# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_name: str, session_id: str) -> dict:
    """Run a single task episode and return final metrics."""
    env = EmailTriageEnv(
        base_url=ENV_BASE_URL,
        task_name=task_name,
        session_id=session_id,
    )
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset()
        obs = reset_result.observation

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if obs is None:
                break

            # Build prompt from observation
            user_prompt = build_user_prompt(obs.model_dump())

            # Get LLM decision
            llm_response = call_llm(client, user_prompt)

            # Build action
            action = TriageAction(
                category=str(llm_response.get("category", "work")).lower(),
                priority=int(llm_response.get("priority", 2)),
                action=str(llm_response.get("action", "mark_read")).lower(),
                response_draft=llm_response.get("response_draft"),
            )

            # Execute step
            step_result = await env.step(action)

            reward = step_result.reward
            done = step_result.done
            error = step_result.info.get("error")

            rewards.append(reward)
            steps_taken = step

            action_summary = f"cat={action.category},pri={action.priority},act={action.action}"
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

            if done:
                break

            obs = step_result.observation

        # Compute normalized score
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        pass
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for i, task_name in enumerate(ALL_TASK_NAMES):
        session_id = f"inference-{task_name}-{i}"
        await run_task(client, task_name, session_id)


if __name__ == "__main__":
    asyncio.run(main())
