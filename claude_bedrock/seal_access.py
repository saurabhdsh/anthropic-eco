"""The Claude access SEAL already uses on EC2.

Source of truth: SEAL `.env.example` and `server/src/services/ai/bedrockProvider.ts`.
`BEDROCK_ENABLED=true`, `ANTHROPIC_API_KEY` empty, credentials from the instance role.
"""

from __future__ import annotations

ROLE_NAME = "WeaveEC2BedrockRole"
REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# SEAL points generation, evaluation, and critic at that same id.
JOBS = (
    ("generation", "Draft assessment questions"),
    ("evaluation", "Score written answers"),
    ("critic", "Review a generated item"),
)
