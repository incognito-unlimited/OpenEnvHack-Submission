"""
Python client for the Email Triage OpenEnv server.
Wraps HTTP calls to the server into a clean async interface.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

from models import ResetResult, StateResult, StepResult, TriageAction


class EmailTriageEnv:
    """
    Async client for the Email Triage OpenEnv server.

    Usage:
        env = EmailTriageEnv(base_url="http://localhost:7860", task_name="easy_triage")
        result = await env.reset()
        obs = result.observation
        while True:
            action = TriageAction(category="work", priority=3, action="reply")
            result = await env.step(action)
            if result.done:
                break
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task_name: str = "easy_triage",
        session_id: str = "default",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.task_name = task_name
        self.session_id = session_id
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(self) -> ResetResult:
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_name": self.task_name, "session_id": self.session_id},
        )
        resp.raise_for_status()
        return ResetResult(**resp.json())

    async def step(self, action: TriageAction) -> StepResult:
        resp = await self._client.post(
            f"{self.base_url}/step",
            json={
                "action": action.model_dump(),
                "session_id": self.session_id,
            },
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    async def state(self) -> StateResult:
        resp = await self._client.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        return StateResult(**resp.json())

    async def close(self):
        await self._client.aclose()

    @classmethod
    def from_env(cls, task_name: str = "easy_triage") -> "EmailTriageEnv":
        """Create a client from environment variables."""
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        return cls(base_url=base_url, task_name=task_name)


# ── Synchronous convenience wrapper ──────────────────────────────────────────

class SyncEmailTriageEnv:
    """Synchronous wrapper around EmailTriageEnv."""

    def __init__(self, base_url: str = "http://localhost:7860", task_name: str = "easy_triage"):
        self._async_env = EmailTriageEnv(base_url=base_url, task_name=task_name)

    def reset(self) -> ResetResult:
        return asyncio.run(self._async_env.reset())

    def step(self, action: TriageAction) -> StepResult:
        return asyncio.run(self._async_env.step(action))

    def state(self) -> StateResult:
        return asyncio.run(self._async_env.state())
