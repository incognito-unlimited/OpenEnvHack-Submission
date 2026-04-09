"""
Core Email Triage Environment logic.
Manages state, computes rewards, and grades agent actions.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Email,
    ResetResult,
    StateResult,
    StepResult,
    TriageAction,
    TriageObservation,
    TriageReward,
)
from tasks_data import TASK_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Reward clipping for OpenEnv compliance
# ─────────────────────────────────────────────────────────────────────────────

SCORE_MIN = 0.01
SCORE_MAX = 0.99

def clip_score(score: float) -> float:
    """Clip score to strictly open interval (0, 1) to satisfy OpenEnv requirements."""
    if score <= 0.0:
        return SCORE_MIN
    if score >= 1.0:
        return SCORE_MAX
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Grading helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"spam", "work", "personal", "newsletter", "urgent", "social"}
VALID_ACTIONS = {"delete", "archive", "reply", "forward", "flag", "mark_read"}

# Semantic category groups (partial credit for close misses)
CATEGORY_SIMILARITY: Dict[str, List[str]] = {
    "urgent": ["work"],
    "work": ["urgent"],
    "spam": [],
    "newsletter": ["social"],
    "social": ["newsletter"],
    "personal": [],
}

# Which actions are reasonable for which categories (partial action credit)
REASONABLE_ACTIONS: Dict[str, List[str]] = {
    "spam": ["delete", "archive"],
    "urgent": ["reply", "forward", "flag"],
    "work": ["reply", "forward", "flag", "mark_read"],
    "personal": ["reply", "flag", "archive", "mark_read"],
    "newsletter": ["archive", "delete", "mark_read"],
    "social": ["archive", "mark_read", "delete"],
}


def grade_category(predicted: str, ground_truth: str) -> Tuple[float, str]:
    """Return (score 0-1, feedback)."""
    predicted = predicted.lower().strip()
    if predicted == ground_truth:
        return 1.0, "exact match"
    if predicted in CATEGORY_SIMILARITY.get(ground_truth, []):
        return 0.4, f"close match (expected '{ground_truth}')"
    if predicted not in VALID_CATEGORIES:
        return 0.0, f"invalid category '{predicted}'"
    return 0.0, f"wrong category (expected '{ground_truth}', got '{predicted}')"


def grade_priority(predicted: int, ground_truth: int) -> Tuple[float, str]:
    """Priority score: exact=1.0, off-by-1=0.6, off-by-2=0.3, else=0."""
    diff = abs(predicted - ground_truth)
    if diff == 0:
        return 1.0, "exact"
    if diff == 1:
        return 0.6, f"off by 1 (expected {ground_truth})"
    if diff == 2:
        return 0.3, f"off by 2 (expected {ground_truth})"
    return 0.0, f"far off (expected {ground_truth}, got {predicted})"


def grade_action(predicted: str, category: str, gt_action: str) -> Tuple[float, str]:
    """Grade the chosen action."""
    predicted = predicted.lower().strip()
    if predicted == gt_action:
        return 1.0, "exact match"
    if predicted in REASONABLE_ACTIONS.get(category, []):
        return 0.5, f"reasonable but not ideal (expected '{gt_action}')"
    if predicted not in VALID_ACTIONS:
        return 0.0, f"invalid action '{predicted}'"
    return 0.0, f"poor action choice (expected '{gt_action}')"


def grade_response(
    response_draft: Optional[str],
    response_required: bool,
    response_keywords: List[str],
    category: str,
) -> Tuple[float, str]:
    """
    Grade response quality based on:
    - Whether a response was provided when needed
    - Keyword coverage (relevance to the email)
    - Minimum length (too short = low effort)
    - Professionalism heuristics
    """
    if not response_required:
        if response_draft:
            return 0.8, "response provided but not needed (minor penalty)"
        return 1.0, "no response needed, none provided"

    if not response_draft or len(response_draft.strip()) < 20:
        return 0.0, "response required but not provided (or too short)"

    text = response_draft.lower()
    score = 0.3  # Base score for providing any response

    # Length check: good responses are 30-300 chars
    length = len(response_draft.strip())
    if length >= 50:
        score += 0.2
    if length >= 100:
        score += 0.1

    # Keyword coverage
    if response_keywords:
        matched = sum(1 for kw in response_keywords if kw.lower() in text)
        coverage = matched / len(response_keywords)
        score += 0.4 * coverage
    else:
        score += 0.4  # No keywords required = full keyword score

    return min(score, 1.0), f"response score: {score:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    Stateful email triage environment.
    Each session independently tracks progress through a task's email inbox.
    """

    def __init__(self, task_name: str = "easy_triage"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASK_REGISTRY.keys())}"
            )
        self.task_name = task_name
        self.task_config = TASK_REGISTRY[task_name]
        self.session_id = str(uuid.uuid4())[:8]

        # State
        self._emails: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._step_rewards: List[float] = []
        self._done: bool = False

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reset the environment to the initial state."""
        self._emails = list(self.task_config["emails"])
        self._current_index = 0
        self._step_rewards = []
        self._done = False
        self.session_id = str(uuid.uuid4())[:8]

        return ResetResult(
            observation=self._make_observation(),
            info={"session_id": self.session_id, "total_emails": len(self._emails)},
        )

    def step(self, action: TriageAction) -> StepResult:
        """Process one triage action and advance to next email."""
        if self._done:
            return StepResult(
                observation=None,
                reward=SCORE_MIN,
                done=True,
                info={"error": "Episode already done. Call reset() first."},
            )

        # Validate inputs
        action.category = action.category.lower().strip()
        action.action = action.action.lower().strip()
        if action.category not in VALID_CATEGORIES:
            action.category = "work"  # default on invalid
        if action.action not in VALID_ACTIONS:
            action.action = "mark_read"

        # Grade current email
        email_data = self._emails[self._current_index]
        gt = email_data["ground_truth"]
        weights = self.task_config["weights"]

        cat_score, cat_fb = grade_category(action.category, gt["category"])
        pri_score, pri_fb = grade_priority(action.priority, gt["priority"])
        act_score, act_fb = grade_action(action.action, gt["category"], gt["action"])
        res_score, res_fb = grade_response(
            action.response_draft,
            gt.get("response_required", False),
            gt.get("response_keywords", []),
            gt["category"],
        )

        reward = (
            weights["category"] * cat_score
            + weights["priority"] * pri_score
            + weights["action"] * act_score
            + weights["response"] * res_score
        )

        # Bonus: correct spam detection is rewarded extra (safety signal)
        if gt["category"] == "spam" and action.category == "spam":
            reward = min(1.0, reward + 0.05)

        # Penalty: classifying spam as urgent (dangerous action)
        if gt["category"] == "spam" and action.category == "urgent":
            reward = max(0.0, reward - 0.2)

        # Clip reward to strictly open interval (0, 1)
        reward = clip_score(reward)

        # Keep all score-like outputs strictly in (0, 1) for validator compatibility.
        cat_score_out = clip_score(cat_score)
        pri_score_out = clip_score(pri_score)
        act_score_out = clip_score(act_score)
        res_score_out = clip_score(res_score)

        reward = round(float(reward), 4)
        self._step_rewards.append(reward)

        triage_reward = TriageReward(
            value=reward,
            category_score=cat_score_out,
            priority_score=pri_score_out,
            action_score=act_score_out,
            response_score=res_score_out,
            breakdown={
                "category_feedback": cat_fb,
                "priority_feedback": pri_fb,
                "action_feedback": act_fb,
                "response_feedback": res_fb,
                "ground_truth_category": gt["category"],
                "ground_truth_priority": gt["priority"],
            },
        )

        # Advance
        self._current_index += 1
        done = self._current_index >= len(self._emails)
        self._done = done

        next_obs = None if done else self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "reward_breakdown": triage_reward.breakdown,
                "detailed_scores": {
                    "category": round(cat_score_out, 4),
                    "priority": round(pri_score_out, 4),
                    "action": round(act_score_out, 4),
                    "response": round(res_score_out, 4),
                },
                "email_id": email_data["id"],
            },
        )

    def state(self) -> StateResult:
        """Return the current environment state."""
        return StateResult(
            task_name=self.task_name,
            step=self._current_index,
            total_emails=len(self._emails),
            emails_processed=self._current_index,
            cumulative_reward=round(sum(self._step_rewards), 4),
            done=self._done,
            step_scores=list(self._step_rewards),
            session_id=self.session_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_observation(self) -> TriageObservation:
        email_data = self._emails[self._current_index]
        email = Email(
            id=email_data["id"],
            sender=email_data["sender"],
            subject=email_data["subject"],
            body=email_data["body"],
            timestamp=email_data["timestamp"],
            has_attachment=email_data.get("has_attachment", False),
            thread_length=email_data.get("thread_length", 1),
        )
        return TriageObservation(
            email=email,
            inbox_remaining=len(self._emails) - self._current_index - 1,
            step_number=self._current_index + 1,
            task_name=self.task_name,
            instructions=self.task_config["instructions"],
            context={
                "total_emails": len(self._emails),
                "session_id": self.session_id,
            },
        )

    @property
    def final_score(self) -> float:
        """Normalized score over the full episode (0.0 - 1.0)."""
        if not self._step_rewards:
            return SCORE_MIN
        return round(clip_score(sum(self._step_rewards) / len(self._step_rewards)), 4)
