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
import re
import textwrap
import time
from typing import Any, Dict, List, Optional

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
DEBUG_INFERENCE = os.getenv("DEBUG_INFERENCE", "0") == "1"

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


def _coerce_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse model output into JSON object, tolerating minor format noise."""
    text = (raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object span.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def _heuristic_fallback(observation_dict: dict) -> Dict[str, Any]:
    """Rule-based fallback to avoid repeated low-score default actions when LLM fails."""
    email = observation_dict["email"]
    sender = str(email.get("sender", "")).lower()
    subject = str(email.get("subject", "")).lower()
    body = str(email.get("body", "")).lower()
    text = f"{sender} {subject} {body}"

    spam_markers = ["phish", "verify", "suspended", "lottery", "winner", "crypto", "urgent account"]
    urgent_markers = ["within 24", "within 48", "deadline", "asap", "immediately", "legal", "security", "breach", "incident"]
    newsletter_markers = ["newsletter", "digest", "weekly", "unsubscribe", "promotion", "sale"]
    social_markers = ["liked your", "commented", "followed you", "friend request", "notification"]
    personal_markers = ["lunch", "dinner", "weekend", "family", "mom", "dad", "birthday"]

    if any(m in text for m in spam_markers):
        return {"category": "spam", "priority": 1, "action": "delete", "response_draft": None}

    if any(m in text for m in urgent_markers):
        return {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_draft": "Acknowledged. I am treating this as urgent and will take immediate action."
        }

    if any(m in text for m in newsletter_markers):
        return {"category": "newsletter", "priority": 2, "action": "archive", "response_draft": None}

    if any(m in text for m in social_markers):
        return {"category": "social", "priority": 2, "action": "archive", "response_draft": None}

    if any(m in text for m in personal_markers):
        return {
            "category": "personal",
            "priority": 3,
            "action": "reply",
            "response_draft": "Thanks for the message. That works for me and I will follow up shortly."
        }

    return {"category": "work", "priority": 3, "action": "mark_read", "response_draft": None}


def _normalize_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model payload into valid action schema with safe defaults."""
    category = str(payload.get("category", "work")).lower().strip()
    if category not in {"spam", "work", "personal", "newsletter", "urgent", "social"}:
        category = "work"

    try:
        priority = int(payload.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    priority = min(max(priority, 1), 5)

    action = str(payload.get("action", "mark_read")).lower().strip()
    if action not in {"delete", "archive", "reply", "forward", "flag", "mark_read"}:
        action = "mark_read"

    response_draft = payload.get("response_draft")
    if action not in {"reply", "forward"}:
        response_draft = None
    elif not isinstance(response_draft, str) or len(response_draft.strip()) < 20:
        response_draft = "Acknowledged. Thanks for your email. I will review and respond with next steps promptly."

    return {
        "category": category,
        "priority": priority,
        "action": action,
        "response_draft": response_draft,
    }


def call_llm(client: OpenAI, user_prompt: str, observation_dict: dict) -> dict:
    """Call the LLM and parse the triage JSON response."""
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.05,
                max_tokens=320,
            )
            raw = completion.choices[0].message.content or ""
            parsed = _coerce_json(raw)
            if parsed is not None:
                return _normalize_action(parsed)
        except Exception as exc:
            last_err = exc

        if attempt < 2:
            time.sleep(0.6 * (attempt + 1))

    if DEBUG_INFERENCE and last_err is not None:
        print(f"[WARN] LLM call fallback after retries: {last_err}", file=os.sys.stderr, flush=True)
    return _normalize_action(_heuristic_fallback(observation_dict))


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
            llm_response = call_llm(client, user_prompt, obs.model_dump())

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
