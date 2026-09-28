"""The agent loop on the same Messages API SEAL uses.

Claude returns a tool_use block. We run the function. We send a tool_result.
Claude writes the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from claude_bedrock.catalog import estimate_cost, family_card
from claude_bedrock.machine import snapshot
from claude_bedrock.runtime import Turn, Usage, assistant_content, complete


SYSTEM = (
    "You are live on stage, explaining Claude to engineers. "
    "For any fact about this machine, a Claude model family, or a price, call a tool. "
    "After the tools return, answer in four spoken sentences. No bullet list."
)

USER = (
    "Look at this machine. I need a live coding assistant and a separate batch "
    "classifier. Which Claude family fits each job, and what is the classroom "
    "cost of 8000 input tokens and 1000 output tokens on the assistant model?"
)


def tool_specs() -> list[dict]:
    return [
        {
            "name": "inspect_machine",
            "description": "Read non-secret facts about the computer running this demo.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "describe_model_family",
            "description": "Describe Haiku, Sonnet, or Opus and when to choose it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "family": {
                        "type": "string",
                        "description": "haiku, sonnet, or opus",
                    }
                },
                "required": ["family"],
            },
        },
        {
            "name": "estimate_cost",
            "description": (
                "Estimate classroom USD cost for a Claude family. "
                "Rates live in the demo, not in a live price API."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "family": {"type": "string"},
                    "input_tokens": {"type": "integer"},
                    "output_tokens": {"type": "integer"},
                },
                "required": ["family", "input_tokens", "output_tokens"],
            },
        },
    ]


def outline_tool() -> dict:
    return {
        "name": "emit_session_card",
        "description": "Return a short talk outline as structured data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "beats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "minute": {"type": "integer"},
                            "title": {"type": "string"},
                            "point": {"type": "string"},
                        },
                        "required": ["minute", "title", "point"],
                    },
                },
            },
            "required": ["title", "duration_minutes", "beats"],
        },
    }


@dataclass
class ToolEvent:
    name: str
    tool_input: dict
    output: dict


@dataclass
class AgentResult:
    answer: str
    events: list[ToolEvent] = field(default_factory=list)
    usages: list[Usage] = field(default_factory=list)
    stop_reason: str = ""


def dispatch(name: str, payload: dict, region: str) -> dict:
    try:
        if name == "inspect_machine":
            return snapshot(region)
        if name == "describe_model_family":
            return family_card(str(payload.get("family", "")))
        if name == "estimate_cost":
            return estimate_cost(
                str(payload.get("family", "")),
                int(payload.get("input_tokens", 0)),
                int(payload.get("output_tokens", 0)),
            )
        return {"error": f"Unknown tool {name}"}
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


def _bad_request(exc: Exception) -> bool:
    return type(exc).__name__ in {"BadRequestError", "ValidationError"} or getattr(exc, "status_code", None) == 400


def _turn(runtime, model_id: str, messages: list[dict], choice: dict) -> Turn:
    try:
        return complete(
            runtime,
            model_id,
            messages,
            SYSTEM,
            max_tokens=500,
            temperature=0.2,
            tools=tool_specs(),
            tool_choice=choice,
        )
    except Exception as exc:
        if choice == {"type": "auto"} or not _bad_request(exc):
            raise
        return complete(
            runtime,
            model_id,
            messages,
            SYSTEM,
            max_tokens=500,
            temperature=0.2,
            tools=tool_specs(),
            tool_choice={"type": "auto"},
        )


def run_agent(runtime, model_id: str, region: str, max_turns: int = 5) -> AgentResult:
    messages: list[dict] = [{"role": "user", "content": USER}]
    result = AgentResult(answer="")

    for turn in range(max_turns):
        # The first turn must call a tool, so the room sees the loop.
        choice = {"type": "any"} if turn == 0 else {"type": "auto"}
        completed = _turn(runtime, model_id, messages, choice)
        result.usages.append(completed.usage)
        result.stop_reason = completed.stop_reason
        messages.append({"role": "assistant", "content": assistant_content(completed.message)})

        tool_blocks = [
            block
            for block in getattr(completed.message, "content", []) or []
            if getattr(block, "type", None) == "tool_use"
        ]
        if completed.stop_reason != "tool_use" or not tool_blocks:
            result.answer = completed.text
            return result

        results = []
        for block in tool_blocks:
            output = dispatch(block.name, block.input or {}, region)
            result.events.append(ToolEvent(block.name, block.input or {}, output))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output),
                    "is_error": "error" in output,
                }
            )
        messages.append({"role": "user", "content": results})

    result.answer = "The tool loop hit its turn limit before Claude finished."
    return result
