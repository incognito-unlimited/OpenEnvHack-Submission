"""
Email datasets with ground truth labels for all three tasks.
Ground truth was established by human labelers and is deterministic.
"""

from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY: Simple 5-email categorization
# Agent must correctly categorize emails. Priority/action graded leniently.
# ─────────────────────────────────────────────────────────────────────────────

EASY_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "easy_001",
        "sender": "promo@mega-lottery-winner.net",
        "subject": "CONGRATULATIONS!! You WON $1,500,000 — CLAIM NOW!!!",
        "body": (
            "Dear Lucky Winner, You have been randomly selected from millions of entries "
            "to receive $1,500,000 USD. To claim your prize, click here: http://scam-link.biz "
            "and provide your bank details. Offer expires in 24 hours. Act NOW!"
        ),
        "timestamp": "2024-03-01T08:02:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "spam",
            "priority": 1,
            "action": "delete",
            "response_required": False,
        },
    },
    {
        "id": "easy_002",
        "sender": "sarah.chen@company.com",
        "subject": "Q1 Budget Review — Action Required by EOD",
        "body": (
            "Hi team, please review the attached Q1 budget spreadsheet and send me your "
            "department's variance explanations before end of day today. The CFO needs this "
            "for tomorrow's board meeting. Let me know if you have questions. — Sarah"
        ),
        "timestamp": "2024-03-01T09:15:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
        },
    },
    {
        "id": "easy_003",
        "sender": "newsletter@techdigest.io",
        "subject": "This Week in Tech: AI Breakthroughs & Startup News",
        "body": (
            "Welcome to Tech Digest Weekly! In this issue: OpenAI releases new model, "
            "top 10 startups to watch in 2024, and a deep dive into quantum computing. "
            "Unsubscribe | View in browser | Privacy Policy"
        ),
        "timestamp": "2024-03-01T07:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 1,
            "action": "archive",
            "response_required": False,
        },
    },
    {
        "id": "easy_004",
        "sender": "mike.johnson@gmail.com",
        "subject": "Weekend BBQ at my place — you coming?",
        "body": (
            "Hey! Having a few friends over for BBQ this Saturday around 4pm. "
            "Can you make it? Let me know and I'll add you to the headcount. "
            "Bringing your famous potato salad would be a bonus 😄"
        ),
        "timestamp": "2024-03-01T11:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "personal",
            "priority": 2,
            "action": "reply",
            "response_required": False,
        },
    },
    {
        "id": "easy_005",
        "sender": "notifications@linkedin.com",
        "subject": "You have 3 new connection requests",
        "body": (
            "People you may know want to connect with you on LinkedIn. "
            "John Smith (Software Engineer at Google), Emma Davis (Product Manager at Meta), "
            "Alex Kumar (Data Scientist). View all requests on LinkedIn."
        ),
        "timestamp": "2024-03-01T10:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "social",
            "priority": 1,
            "action": "archive",
            "response_required": False,
        },
    },
]

EASY_TASK_CONFIG = {
    "name": "easy_triage",
    "description": "Categorize 5 emails into the correct category. Priority and action are leniently graded.",
    "difficulty": "easy",
    "max_steps": 5,
    "emails": EASY_EMAILS,
    "instructions": (
        "You are an email triage assistant. For each email, determine:\n"
        "1. category: one of [spam, work, personal, newsletter, urgent, social]\n"
        "2. priority: 1 (lowest) to 5 (highest)\n"
        "3. action: one of [delete, archive, reply, forward, flag, mark_read]\n"
        "Focus on getting the category right."
    ),
    "weights": {"category": 0.6, "priority": 0.25, "action": 0.15, "response": 0.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: 10-email inbox, category + priority both matter
# ─────────────────────────────────────────────────────────────────────────────

MEDIUM_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "med_001",
        "sender": "ceo@acme-corp.com",
        "subject": "URGENT: Server outage affecting all customers — need immediate response",
        "body": (
            "We have a P0 production outage. All customer-facing services are down. "
            "Revenue impact is ~$50k/hour. Engineering leads need to join the bridge call "
            "immediately: +1-800-555-0100 code 1234. This is your top priority right now."
        ),
        "timestamp": "2024-03-02T14:05:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
        },
    },
    {
        "id": "med_002",
        "sender": "phishing@secure-bank-verify.net",
        "subject": "⚠️ Your account has been SUSPENDED — verify immediately",
        "body": (
            "Dear Valued Customer, your bank account has been suspended due to suspicious "
            "activity. Click here IMMEDIATELY to verify your identity and restore access: "
            "http://totally-not-a-scam.net/verify. Failure to act within 2 hours will result "
            "in permanent account closure. — Security Team"
        ),
        "timestamp": "2024-03-02T08:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "spam",
            "priority": 1,
            "action": "delete",
            "response_required": False,
        },
    },
    {
        "id": "med_003",
        "sender": "recruiter@bigtech.com",
        "subject": "Exciting opportunity at BigTech — Senior Engineer role",
        "body": (
            "Hi, I came across your profile and I'm reaching out about a Senior Software "
            "Engineer role at BigTech. The compensation is $250k+ base. If you're open to "
            "a conversation, let me know a good time to chat. Best, Rachel (Recruiter)"
        ),
        "timestamp": "2024-03-02T10:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "work",
            "priority": 2,
            "action": "flag",
            "response_required": False,
        },
    },
    {
        "id": "med_004",
        "sender": "mom@familymail.com",
        "subject": "Re: Thanksgiving plans",
        "body": (
            "Honey, just confirming you're coming for Thanksgiving? Grandma is flying in "
            "from Florida and it would mean so much to everyone. Let me know about dietary "
            "restrictions for your partner. Love, Mom"
        ),
        "timestamp": "2024-03-02T09:00:00Z",
        "has_attachment": False,
        "thread_length": 3,
        "ground_truth": {
            "category": "personal",
            "priority": 3,
            "action": "reply",
            "response_required": False,
        },
    },
    {
        "id": "med_005",
        "sender": "weekly@producthunt.com",
        "subject": "This week's top products on Product Hunt",
        "body": (
            "🚀 Top launches this week: 1) AI writing tool with 10k upvotes, "
            "2) No-code database builder, 3) Design-to-code converter. "
            "See all 50+ new products. Unsubscribe | Manage preferences"
        ),
        "timestamp": "2024-03-02T07:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 1,
            "action": "archive",
            "response_required": False,
        },
    },
    {
        "id": "med_006",
        "sender": "hr@company.com",
        "subject": "Performance review deadline: Submit self-assessment by Friday",
        "body": (
            "Reminder: The annual performance review cycle closes this Friday at 5pm. "
            "Please complete your self-assessment form at hr.company.com/review. "
            "Late submissions will not be accepted. Contact HR if you have issues logging in."
        ),
        "timestamp": "2024-03-02T08:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "work",
            "priority": 4,
            "action": "flag",
            "response_required": False,
        },
    },
    {
        "id": "med_007",
        "sender": "notifications@twitter.com",
        "subject": "@techguru123 liked your tweet",
        "body": (
            "@techguru123 liked your tweet: 'Just shipped a new feature! Super excited '🚀 "
            "See more activity on Twitter. You can manage your notification preferences "
            "in settings. © Twitter Inc."
        ),
        "timestamp": "2024-03-02T11:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "social",
            "priority": 1,
            "action": "archive",
            "response_required": False,
        },
    },
    {
        "id": "med_008",
        "sender": "client@importantclient.com",
        "subject": "Contract renewal discussion — need decision this week",
        "body": (
            "Hi, our current contract expires in 10 days and we need to finalize the "
            "renewal terms. We're happy with the service but need the revised pricing "
            "proposal before Thursday. Could you schedule a 30-min call this week? "
            "— David Martinez, VP Operations"
        ),
        "timestamp": "2024-03-02T13:00:00Z",
        "has_attachment": False,
        "thread_length": 2,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
        },
    },
    {
        "id": "med_009",
        "sender": "deals@amazon.com",
        "subject": "Today's deals just for you: Up to 60% off!",
        "body": (
            "Hi there! Based on your browsing history, we think you'll love these deals: "
            "Laptop 40% off, Headphones 60% off, Coffee maker 30% off. "
            "Shop now before they expire. Unsubscribe | Privacy Notice"
        ),
        "timestamp": "2024-03-02T06:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 1,
            "action": "delete",
            "response_required": False,
        },
    },
    {
        "id": "med_010",
        "sender": "james.wilson@company.com",
        "subject": "Code review requested: Authentication module PR #472",
        "body": (
            "Hey, could you review PR #472 when you get a chance? It's the authentication "
            "module refactor we discussed. No rush — end of week is fine. Added you as "
            "reviewer on GitHub. Thanks! — James"
        ),
        "timestamp": "2024-03-02T15:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "work",
            "priority": 3,
            "action": "flag",
            "response_required": False,
        },
    },
]

MEDIUM_TASK_CONFIG = {
    "name": "priority_inbox",
    "description": "Categorize and prioritize 10 emails. Both category and priority are equally graded.",
    "difficulty": "medium",
    "max_steps": 10,
    "emails": MEDIUM_EMAILS,
    "instructions": (
        "You are an email triage assistant managing a busy professional inbox. For each email, determine:\n"
        "1. category: one of [spam, work, personal, newsletter, urgent, social]\n"
        "2. priority: 1 (lowest) to 5 (highest) — be precise, priority matters here\n"
        "3. action: one of [delete, archive, reply, forward, flag, mark_read]\n"
        "Both category AND priority are graded equally. Distinguish carefully between 'urgent' (P5), "
        "'work' (P3-4), and 'newsletter' (P1)."
    ),
    "weights": {"category": 0.4, "priority": 0.4, "action": 0.2, "response": 0.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD: 8 emails, full triage + response drafts for urgent/work
# ─────────────────────────────────────────────────────────────────────────────

HARD_EMAILS: List[Dict[str, Any]] = [
    {
        "id": "hard_001",
        "sender": "legal@biglaw.com",
        "subject": "NOTICE: Intellectual property dispute — response required within 48 hours",
        "body": (
            "Dear Sir/Madam, our client alleges that your product infringes on Patent #US9876543. "
            "Under 35 U.S.C. § 271, we formally demand that you: (1) cease all infringing activity, "
            "(2) provide an accounting of all revenues, (3) respond within 48 hours to avoid "
            "immediate injunctive relief. Failure to respond will be treated as admission. "
            "— Morrison & Foerster LLP"
        ),
        "timestamp": "2024-03-03T09:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "forward",
            "response_required": True,
            "response_keywords": ["acknowledge", "legal", "review", "counsel", "attorney", "48 hours", "patent"],
        },
    },
    {
        "id": "hard_002",
        "sender": "security-alert@company-internal.com",
        "subject": "SECURITY ALERT: Unusual login detected from unknown IP",
        "body": (
            "We detected a login to your account from IP 185.234.x.x (Russia) at 03:14 UTC. "
            "If this was you, no action needed. If not, your account may be compromised. "
            "Immediately: (1) change your password, (2) enable 2FA, (3) review recent activity. "
            "Contact security@company.com if you need assistance. — IT Security Team"
        ),
        "timestamp": "2024-03-03T03:20:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
            "response_keywords": ["not me", "suspicious", "password", "security", "unauthorized", "investigate"],
        },
    },
    {
        "id": "hard_003",
        "sender": "board@company.com",
        "subject": "Board meeting rescheduled — new date conflicts with product launch",
        "body": (
            "The Q2 board meeting has been moved from April 15 to April 8 due to board member "
            "availability. However, April 8 is our planned product launch date. We need to "
            "decide: should we postpone the board meeting again, delay the launch, or split "
            "the team? Please share your recommendation by tomorrow noon. — Chairman Roberts"
        ),
        "timestamp": "2024-03-03T10:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
            "response_keywords": ["recommendation", "launch", "board", "reschedule", "option", "propose"],
        },
    },
    {
        "id": "hard_004",
        "sender": "engineering-team@company.com",
        "subject": "Weekly engineering sync notes + action items",
        "body": (
            "Hi all, here are the notes from today's sync: (1) Auth service refactor on track "
            "for March 15, (2) Performance regression in v2.3.1 needs investigation, "
            "(3) New hire starts Monday — please review onboarding checklist. "
            "Action items assigned in Jira. Let me know if you have questions. — Tech Lead"
        ),
        "timestamp": "2024-03-03T17:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "work",
            "priority": 3,
            "action": "mark_read",
            "response_required": False,
            "response_keywords": [],
        },
    },
    {
        "id": "hard_005",
        "sender": "fake-apple@appleid-verify.com",
        "subject": "Your Apple ID has been locked — immediate verification required",
        "body": (
            "Apple Support: Your Apple ID (user@email.com) was used to sign in on a new device. "
            "If this wasn't you, verify your identity immediately to prevent data loss: "
            "http://apple-id-verify-secure.suspicious.net. "
            "Your account will be permanently deleted if not verified within 24 hours."
        ),
        "timestamp": "2024-03-03T08:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "spam",
            "priority": 1,
            "action": "delete",
            "response_required": False,
            "response_keywords": [],
        },
    },
    {
        "id": "hard_006",
        "sender": "top-customer@enterprise-client.com",
        "subject": "Re: Re: Re: Production bug causing data corruption in our environment",
        "body": (
            "This is now our FOURTH escalation on this issue. Our CTO is involved. "
            "The bug has corrupted 3 days of production data. Our SLA mandates a P1 response "
            "within 2 hours and a fix within 24 hours. We are considering invoking the penalty "
            "clause in our contract (~$200k). We need a call TODAY. — Jennifer Walsh, CTO"
        ),
        "timestamp": "2024-03-03T11:30:00Z",
        "has_attachment": True,
        "thread_length": 4,
        "ground_truth": {
            "category": "urgent",
            "priority": 5,
            "action": "reply",
            "response_required": True,
            "response_keywords": ["apologize", "call", "today", "fix", "engineer", "priority", "investigate", "escalate"],
        },
    },
    {
        "id": "hard_007",
        "sender": "digest@hackernews.com",
        "subject": "Hacker News Top Stories: LLMs, Rust, and Distributed Systems",
        "body": (
            "Today's top stories: 1) 'The end of REST APIs' (2.4k points), "
            "2) 'Rust is now the fastest language in TechEmpower benchmarks' (1.8k), "
            "3) 'How Discord stores messages at scale' (1.5k). "
            "Click to read. Unsubscribe from digest."
        ),
        "timestamp": "2024-03-03T07:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "ground_truth": {
            "category": "newsletter",
            "priority": 1,
            "action": "archive",
            "response_required": False,
            "response_keywords": [],
        },
    },
    {
        "id": "hard_008",
        "sender": "partner@strategic-vendor.com",
        "subject": "Partnership proposal: Co-marketing opportunity worth $500k",
        "body": (
            "Hi, I'm the VP of Partnerships at StrategicVendor. We'd like to propose a "
            "co-marketing initiative targeting the same enterprise segment. We're projecting "
            "$500k in combined pipeline. I'll be in your city next Tuesday and Wednesday — "
            "would love 30 minutes to walk through the proposal. Can you find a slot?"
        ),
        "timestamp": "2024-03-03T14:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "ground_truth": {
            "category": "work",
            "priority": 3,
            "action": "reply",
            "response_required": True,
            "response_keywords": ["meeting", "Tuesday", "Wednesday", "schedule", "calendar", "proposal", "interest"],
        },
    },
]

HARD_TASK_CONFIG = {
    "name": "full_triage",
    "description": (
        "Complete email triage: categorize, prioritize, choose action, AND draft "
        "a response for emails that require one. Response quality is graded."
    ),
    "difficulty": "hard",
    "max_steps": 8,
    "emails": HARD_EMAILS,
    "instructions": (
        "You are an executive assistant performing full email triage. For each email:\n"
        "1. category: one of [spam, work, personal, newsletter, urgent, social]\n"
        "2. priority: 1 (lowest) to 5 (highest)\n"
        "3. action: one of [delete, archive, reply, forward, flag, mark_read]\n"
        "4. response_draft: REQUIRED for any email where action='reply' or action='forward'.\n"
        "   Write a professional, concise response (2-4 sentences) addressing the key points.\n"
        "   For urgent emails, acknowledge urgency and commit to next steps.\n"
        "Responses are graded on relevance, professionalism, and addressing the core issue."
    ),
    "weights": {"category": 0.3, "priority": 0.25, "action": 0.2, "response": 0.25},
}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy_triage": EASY_TASK_CONFIG,
    "priority_inbox": MEDIUM_TASK_CONFIG,
    "full_triage": HARD_TASK_CONFIG,
}

ALL_TASK_NAMES = list(TASK_REGISTRY.keys())
