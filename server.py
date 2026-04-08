"""
FastAPI server implementing the OpenEnv HTTP API for the Email Triage environment.

Endpoints:
  POST /reset           → Start a new episode
  POST /step            → Take a triage action
  GET  /state           → Inspect current state
  GET  /tasks           → List available tasks
  GET  /health          → Health check
  GET  /                → Landing page
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import EmailTriageEnvironment
from models import (
    ResetResult,
    StateResult,
    StepResult,
    TriageAction,
    TriageObservation,
)
from tasks_data import ALL_TASK_NAMES, TASK_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email-triage-env")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv environment for training AI agents on email triage tasks. "
        "Agents learn to categorize, prioritize, and respond to realistic emails."
    ),
    version="1.0.0",
)

# Session storage: session_id → environment instance
_sessions: Dict[str, EmailTriageEnvironment] = {}
_default_session_id = "default"


def _get_or_create_session(session_id: str, task_name: str) -> EmailTriageEnvironment:
    if session_id not in _sessions or _sessions[session_id].task_name != task_name:
        _sessions[session_id] = EmailTriageEnvironment(task_name=task_name)
    return _sessions[session_id]


def _get_session(session_id: str) -> EmailTriageEnvironment:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return _sessions[session_id]


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "easy_triage"
    session_id: Optional[str] = _default_session_id


class StepRequest(BaseModel):
    action: TriageAction
    session_id: Optional[str] = _default_session_id


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    tasks_html = "".join(
        f"<li><b>{name}</b> ({cfg['difficulty']}): {cfg['description']}</li>"
        for name, cfg in TASK_REGISTRY.items()
    )
    return f"""
    <html><head><title>Email Triage OpenEnv</title></head>
    <body style="font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px">
      <h1>📧 Email Triage OpenEnv</h1>
      <p>An OpenEnv environment for training AI agents on email triage. 
         Agents categorize, prioritize, and respond to realistic emails.</p>
      <h2>Available Tasks</h2>
      <ul>{tasks_html}</ul>
      <h2>API Endpoints</h2>
      <ul>
        <li><code>POST /reset</code> — start new episode</li>
        <li><code>POST /step</code> — submit triage action</li>
        <li><code>GET /state</code> — inspect environment state</li>
        <li><code>GET /tasks</code> — list all tasks</li>
        <li><a href="/docs">Interactive API docs (Swagger)</a></li>
      </ul>
    </body></html>
    """


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "version": "1.0.0", "tasks": ALL_TASK_NAMES}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    return {
        "name": "email-triage-env",
        "description": (
            "A real-world OpenEnv simulation for email triage with classification, "
            "prioritization, action selection, and response drafting."
        ),
        "version": "1.0.0",
        "tasks": ALL_TASK_NAMES,
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    return {
        "action": TriageAction.model_json_schema(),
        "observation": TriageObservation.model_json_schema(),
        "state": StateResult.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "ok": True,
            "method": method,
            "message": "MCP endpoint reachable",
        },
    }


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": name,
                "description": cfg["description"],
                "difficulty": cfg["difficulty"],
                "num_emails": len(cfg["emails"]),
            }
            for name, cfg in TASK_REGISTRY.items()
        ]
    }


@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest = ResetRequest()) -> ResetResult:
    """
    Reset the environment and start a new episode.
    Pass task_name to select which task to run.
    """
    task_name = request.task_name or "easy_triage"
    session_id = request.session_id or _default_session_id

    if task_name not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid tasks: {ALL_TASK_NAMES}",
        )

    env = _get_or_create_session(session_id, task_name)
    result = env.reset()
    logger.info(f"[{session_id}] Reset task='{task_name}'")
    return result


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest) -> StepResult:
    """
    Submit a triage action for the current email.
    Returns observation (next email), reward, and done flag.
    """
    session_id = request.session_id or _default_session_id
    env = _get_session(session_id)

    result = env.step(request.action)
    logger.info(
        f"[{session_id}] Step {env._current_index} "
        f"action={request.action.category}/{request.action.priority} "
        f"reward={result.reward:.3f} done={result.done}"
    )
    return result


@app.get("/state", response_model=StateResult)
async def state(session_id: str = Query(default=_default_session_id)) -> StateResult:
    """Return the current environment state without advancing it."""
    env = _get_session(session_id)
    return env.state()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
