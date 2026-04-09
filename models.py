"""
Pydantic models for the Email Triage OpenEnv environment.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Domain objects ────────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1


# ── OpenEnv Observation ───────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """Observation returned to the agent at each step."""
    email: Email
    inbox_remaining: int = Field(description="How many emails remain after this one")
    step_number: int
    task_name: str
    instructions: str
    valid_categories: List[str] = Field(
        default_factory=lambda: ["spam", "work", "personal", "newsletter", "urgent", "social"]
    )
    valid_actions: List[str] = Field(
        default_factory=lambda: ["delete", "archive", "reply", "forward", "flag", "mark_read"]
    )
    context: Dict[str, Any] = Field(default_factory=dict)


# ── OpenEnv Action ────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """Action the agent takes to triage a single email."""
    category: str = Field(description="One of: spam | work | personal | newsletter | urgent | social")
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = highest")
    action: str = Field(description="One of: delete | archive | reply | forward | flag | mark_read")
    response_draft: Optional[str] = Field(
        default=None,
        description="Optional response draft (required for urgent/work emails in hard task)"
    )


# ── OpenEnv Reward ────────────────────────────────────────────────────────────

class TriageReward(BaseModel):
    value: float = Field(gt=0.0, lt=1.0)
    category_score: float = Field(gt=0.0, lt=1.0)
    priority_score: float = Field(gt=0.0, lt=1.0)
    action_score: float = Field(gt=0.0, lt=1.0)
    response_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    breakdown: Dict[str, Any] = Field(default_factory=dict)


# ── Step / Reset / State results ──────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Optional[TriageObservation] = None
    reward: float = Field(gt=0.0, lt=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: TriageObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    task_name: str
    step: int
    total_emails: int
    emails_processed: int
    cumulative_reward: float = Field(gt=0.0, lt=1.0)
    done: bool
    step_scores: List[float] = Field(default_factory=list)
    session_id: str = ""
