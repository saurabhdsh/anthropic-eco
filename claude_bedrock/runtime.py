"""Messages API on Bedrock, the same call SEAL's BedrockProvider makes.

SEAL (TypeScript):
    new AnthropicBedrock({ awsRegion })
    client.messages.create({ model, max_tokens, system, messages })

This module is that call in Python. The EC2 role supplies the credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


RETRYABLE = {
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class Turn:
    text: str
    usage: Usage
    stop_reason: str
    message: object


@dataclass
class Meter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, usage: Usage, label: str) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.notes.append(
            f"{label}: in {usage.input_tokens} / out {usage.output_tokens} / {usage.latency_ms} ms"
        )


def client(region: str):
    from anthropic import AnthropicBedrock

    # No access key. AnthropicBedrock uses the default AWS chain, which on
    # this EC2 is WeaveEC2BedrockRole — the same chain SEAL uses.
    return AnthropicBedrock(aws_region=region)


def caller_identity(region: str) -> dict:
    import boto3

    identity = boto3.client("sts", region_name=region).get_caller_identity()
    return {"account": identity["Account"], "arn": identity["Arn"]}


def explain_error(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    code = error.get("Code") or type(exc).__name__
    message = error.get("Message") or getattr(exc, "message", None) or str(exc)
    hints = {
        "AccessDeniedException": (
            "The instance role cannot invoke this model. "
            "SEAL's role is WeaveEC2BedrockRole. Attach iam/bedrock-session-policy.json if this caller is missing bedrock:InvokeModel."
        ),
        "PermissionDeniedError": (
            "Bedrock refused this role. Confirm the instance profile is WeaveEC2BedrockRole, "
            "the same one SEAL's API uses."
        ),
        "NotFoundError": (
            "This model id is not in the region. SEAL uses "
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0 in us-east-1."
        ),
        "ValidationException": (
            "Bedrock rejected the model id. Set BEDROCK_MODEL_ID to the SEAL id: "
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0."
        ),
        "NoCredentialsError": (
            "This process has no AWS credentials. Run it on the SEAL EC2, "
            "where WeaveEC2BedrockRole is the instance profile."
        ),
    }
    hint = hints.get(code, "")
    return f"{code}: {message}" + (f"\n\n{hint}" if hint else "")


def _call(fn, **kwargs):
    last: Exception | None = None
    for attempt in range(3):
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - retry only busy-service errors
            last = exc
            if type(exc).__name__ in RETRYABLE and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last  # pragma: no cover


def _request(
    model_id: str,
    messages: list[dict],
    system: str,
    max_tokens: int,
    temperature: float,
    tools: list[dict] | None,
    tool_choice: dict | None,
) -> dict:
    payload: dict = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    return payload


def message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", [])
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            parts.append(str(block.text))
    return "\n".join(part for part in parts if part).strip()


def assistant_content(message: object) -> list[dict]:
    """Blocks safe to send back as the assistant turn."""
    blocks: list[dict] = []
    for block in getattr(message, "content", []) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return blocks


def _usage(message: object, started: float) -> Usage:
    usage = getattr(message, "usage", None)
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def complete(
    runtime,
    model_id: str,
    messages: list[dict],
    system: str,
    max_tokens: int = 400,
    temperature: float = 0.2,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
) -> Turn:
    started = time.perf_counter()
    message = _call(
        runtime.messages.create,
        **_request(model_id, messages, system, max_tokens, temperature, tools, tool_choice),
    )
    return Turn(
        text=message_text(message),
        usage=_usage(message, started),
        stop_reason=str(getattr(message, "stop_reason", "") or ""),
        message=message,
    )


def stream_text(
    runtime,
    model_id: str,
    messages: list[dict],
    system: str,
    on_text,
    max_tokens: int = 220,
    temperature: float = 0.3,
) -> tuple[Usage, str | None]:
    started = time.perf_counter()
    with runtime.messages.stream(
        **_request(model_id, messages, system, max_tokens, temperature, None, None)
    ) as stream:
        for piece in stream.text_stream:
            on_text(piece)
        final = stream.get_final_message()
    return _usage(final, started), getattr(final, "stop_reason", None)
