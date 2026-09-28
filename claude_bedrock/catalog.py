"""Facts the session teaches. Prices are classroom figures, not a bill."""

from __future__ import annotations

ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Dollars per million tokens. Update these from the Bedrock pricing page
# before a talk if you want the numbers to match this month's invoice.
CLASSROOM_RATES: dict[str, dict[str, float | str]] = {
    "haiku": {
        "label": "Haiku",
        "role": "Fast and inexpensive. Routing, classification, extraction, high volume.",
        "input_per_million": 1.00,
        "output_per_million": 5.00,
    },
    "sonnet": {
        "label": "Sonnet",
        "role": "The default. Coding, long documents, tools, most product features.",
        "input_per_million": 3.00,
        "output_per_million": 15.00,
    },
    "opus": {
        "label": "Opus",
        "role": "The hardest reasoning. Use it when a wrong answer is expensive.",
        "input_per_million": 15.00,
        "output_per_million": 75.00,
    },
}

SURFACES = [
    (
        "SEAL on this EC2",
        "IAM role WeaveEC2BedrockRole",
        "AnthropicBedrock messages.create. This session runs the same call.",
    ),
    (
        "Claude.ai, Desktop, Claude Code",
        "A person signs in",
        "The product sends the prompt for you",
    ),
    (
        "Claude API",
        "An API key",
        "Messages JSON to api.anthropic.com",
    ),
    (
        "Bedrock Converse",
        "An IAM role",
        "messages + inferenceConfig. AWS's uniform wrapper.",
    ),
    (
        "Bedrock InvokeModel",
        "An IAM role",
        "The raw Anthropic body, anthropic_version bedrock-2023-05-31",
    ),
    (
        "Bedrock Messages (Mantle)",
        "IAM SigV4",
        "The same Messages JSON as the Claude API, on AWS",
    ),
    (
        "MCP and the Agent SDK",
        "Whatever runtime you host",
        "Tools and data Claude can call while it works",
    ),
]

SONNET_PREFERENCES = [
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "claude-sonnet",
]

HAIKU_PREFERENCES = [
    "claude-haiku-4-5",
    "claude-3-5-haiku",
    "claude-3-haiku",
    "claude-haiku",
]


def family_of(model_id: str) -> str:
    lowered = model_id.lower()
    for name in ("haiku", "opus", "sonnet"):
        if name in lowered:
            return name
    return "sonnet"


def normalize_family(value: str) -> str:
    lowered = value.lower()
    for name in ("haiku", "sonnet", "opus"):
        if name in lowered:
            return name
    known = ", ".join(CLASSROOM_RATES)
    raise ValueError(f"family must be one of: {known}")


def family_card(family: str) -> dict:
    name = normalize_family(family)
    rate = CLASSROOM_RATES[name]
    return {
        "family": name,
        "label": rate["label"],
        "role": rate["role"],
        "input_usd_per_million_tokens": rate["input_per_million"],
        "output_usd_per_million_tokens": rate["output_per_million"],
        "price_source": "Classroom rates stored in claude_bedrock/catalog.py",
    }


def estimate_cost(family: str, input_tokens: int, output_tokens: int) -> dict:
    card = family_card(family)
    input_tokens = max(0, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    input_usd = input_tokens / 1_000_000 * float(card["input_usd_per_million_tokens"])
    output_usd = output_tokens / 1_000_000 * float(card["output_usd_per_million_tokens"])
    return {
        "family": card["family"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(input_usd + output_usd, 6),
        "price_source": card["price_source"],
    }


def anthropic_invoke_body(system: str, user_text: str, max_tokens: int) -> dict:
    """The JSON body InvokeModel expects for Claude on Bedrock."""
    return {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
    }
