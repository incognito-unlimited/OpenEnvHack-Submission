# 📧 Email Triage OpenEnv

A real-world OpenEnv environment where AI agents learn to **triage a professional inbox**: categorize emails, assign priorities, choose appropriate actions, and draft professional responses for emails that require them.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

---

## 🌍 Motivation

Email triage is a universal productivity task. Professionals spend 28% of their workday on email (McKinsey, 2023). Training agents to accurately triage email inboxes has immediate commercial value in AI assistant products, customer support automation, and executive assistant tools.

Unlike toy environments, this task requires:
- **Semantic understanding** (is this urgent or spam?)
- **Context-aware prioritization** (P1 vs P3 — the difference matters)
- **Appropriate action selection** (delete vs archive vs reply)
- **Response generation** (drafting a professional acknowledgement under time pressure)

---

## 🗂️ Environment Structure

```
email-triage-env/
├── Dockerfile           # Container definition
├── README.md            # This file
├── openenv.yaml         # OpenEnv metadata
├── requirements.txt     # Python dependencies
├── server.py            # FastAPI server (OpenEnv HTTP API)
├── environment.py       # Core environment logic + reward functions
├── models.py            # Pydantic models (Observation, Action, Reward)
├── tasks_data.py        # Email datasets with ground truth labels
├── client.py            # Python async client for the server
└── inference.py         # Baseline inference script (OpenAI client)
```

---

## 📊 Observation & Action Spaces

### Observation
```json
{
  "email": {
    "id": "hard_001",
    "sender": "legal@biglaw.com",
    "subject": "NOTICE: IP dispute — response required within 48 hours",
    "body": "...",
    "timestamp": "2024-03-03T09:00:00Z",
    "has_attachment": true,
    "thread_length": 1
  },
  "inbox_remaining": 7,
  "step_number": 1,
  "task_name": "full_triage",
  "instructions": "...",
  "valid_categories": ["spam", "work", "personal", "newsletter", "urgent", "social"],
  "valid_actions": ["delete", "archive", "reply", "forward", "flag", "mark_read"]
}
```

### Action
```json
{
  "category": "urgent",
  "priority": 5,
  "action": "forward",
  "response_draft": "Acknowledged. Forwarding to our legal counsel immediately for review within the 48-hour window. We take this matter seriously and will respond formally through appropriate channels."
}
```

---

## 🎯 Tasks

### Task 1: `easy_triage` (Easy)
- **Emails**: 5 clearly distinct emails (spam, urgent, newsletter, personal, social)
- **Goal**: Get the category right
- **Scoring weights**: category 60%, priority 25%, action 15%
- **Expected difficulty**: A capable LLM should score ~0.75+
- **Baseline score**: ~0.72

### Task 2: `priority_inbox` (Medium)
- **Emails**: 10 realistic professional inbox emails
- **Goal**: Correctly categorize AND assign precise priorities
- **Scoring weights**: category 40%, priority 40%, action 20%
- **Challenge**: Distinguishing urgent (P5) from important-but-not-urgent work (P3-4)
- **Baseline score**: ~0.61

### Task 3: `full_triage` (Hard)
- **Emails**: 8 high-stakes emails (legal notices, security alerts, client escalations)
- **Goal**: Full triage + draft professional responses for actionable emails
- **Scoring weights**: category 30%, priority 25%, action 20%, response quality 25%
- **Challenge**: Response quality is graded on relevance, keyword coverage, and length
- **Baseline score**: ~0.54

---

## 🏆 Reward Function

The reward at each step is a weighted combination:

```
reward = w_cat * category_score
       + w_pri * priority_score
       + w_act * action_score
       + w_res * response_score
```

**Partial credit** is given throughout (not just binary):
- **Category**: 1.0 exact, 0.4 for semantically-close category (e.g., `urgent` vs `work`), 0.0 wrong
- **Priority**: 1.0 exact, 0.6 off-by-1, 0.3 off-by-2, 0.0 otherwise
- **Action**: 1.0 ideal, 0.5 reasonable-but-not-ideal, 0.0 wrong
- **Response**: grades on provision, length, keyword coverage relative to email's key topics

**Special signals:**
- 🟢 **Spam bonus** (+0.05): Correctly identifying phishing/spam
- 🔴 **Spam-as-urgent penalty** (-0.2): Dangerous misclassification

**Episode score** = mean of per-step rewards (normalized to [0, 1])

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone and install
git clone <your-repo-url>
cd email-triage-env
pip install -r requirements.txt

# Start the server
python server.py
# → Server running at http://localhost:7860
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env
```

### Running Inference

```bash
# Set environment variables
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

# Run baseline
python inference.py
```

---

## 🔌 API Reference

All endpoints accept/return JSON.

### `POST /reset`
Start a new episode.
```json
{"task_name": "easy_triage", "session_id": "my-session"}
```

### `POST /step`
Submit a triage action.
```json
{
  "session_id": "my-session",
  "action": {
    "category": "spam",
    "priority": 1,
    "action": "delete",
    "response_draft": null
  }
}
```
Returns: `{observation, reward, done, info}`

### `GET /state?session_id=my-session`
Returns current state without advancing: `{task_name, step, total_emails, cumulative_reward, done, step_scores}`

### `GET /tasks`
Lists all available tasks with descriptions.

### `GET /health`
Health check: `{"status": "healthy", "version": "1.0.0"}`

### `GET /metadata`
Returns environment metadata used by OpenEnv runtime validators.

### `GET /schema`
Returns JSON schemas for action, observation, and state.

### `POST /mcp`
Compatibility endpoint for JSON-RPC reachability checks.

---

## 📈 Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` at temperature 0.1:

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| `easy_triage` | 0.72 | 5 | Strong on clear spam/newsletter, sometimes confuses urgent/work |
| `priority_inbox` | 0.61 | 10 | Struggles with exact priority (off-by-1 common) |
| `full_triage` | 0.54 | 8 | Response drafts often miss key keywords for complex legal/security emails |
| **Average** | **0.62** | — | — |

---

## 🔍 Pre-Submission Validation

```bash
# Validate OpenEnv compliance
pip install openenv-core
openenv validate

# Run full validation script
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space .
```

---

## 📋 Checklist

- [x] Real-world task (email triage is a universal productivity challenge)
- [x] OpenEnv spec compliance (typed Pydantic models, step/reset/state endpoints)
- [x] 3 tasks with agent graders (easy/medium/hard, scores 0.0–1.0)
- [x] Meaningful reward function (partial credit at every step, not binary)
- [x] Baseline inference script (`inference.py` with OpenAI client)
- [x] Dockerfile (builds and runs cleanly)
- [x] HuggingFace Space deployment
- [x] Documentation (this README)

---

## 📄 License

MIT
