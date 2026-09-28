"""The beats of the talk. Each function is one thing to show."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from claude_bedrock.agent import USER, outline_tool, run_agent
from claude_bedrock.catalog import (
    CLASSROOM_RATES,
    SURFACES,
    anthropic_invoke_body,
    estimate_cost,
    family_of,
)
from claude_bedrock.runtime import Meter, Usage, complete, explain_error, stream_text
from claude_bedrock.seal_access import MODEL_ID, ROLE_NAME
from claude_bedrock.ui import say, show_json, usage_line


HELLO_SYSTEM = "You explain Claude to engineers. Be brief, concrete, and spoken."
HELLO_USER = "In two sentences, what is Claude, and why call it through Amazon Bedrock?"


@dataclass
class Stage:
    out: Console
    region: str
    primary: str | None
    fast: str | None
    offline: bool
    runtime: object | None = None
    meter: Meter = field(default_factory=Meter)
    identity: dict | None = None

    def guard(self, label: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a live talk should continue
            self.out.print(Panel(explain_error(exc), title=label, border_style="red"))
            return None


def ecosystem(stage: Stage) -> None:
    say(
        stage.out,
        "Claude is one model family behind several front doors. "
        "This session calls it from this EC2 through AnthropicBedrock. "
        f"The instance role is {ROLE_NAME}. The model is {MODEL_ID}.",
    )
    surfaces = Table(title="Where Claude shows up", header_style="bold")
    surfaces.add_column("Surface")
    surfaces.add_column("Credential")
    surfaces.add_column("What you send")
    for row in SURFACES:
        surfaces.add_row(*row)
    stage.out.print(surfaces)

    families = Table(title="Three families, classroom USD per million tokens", header_style="bold")
    families.add_column("Family")
    families.add_column("Use it for")
    families.add_column("Input", justify="right")
    families.add_column("Output", justify="right")
    for key in ("haiku", "sonnet", "opus"):
        rate = CLASSROOM_RATES[key]
        families.add_row(
            str(rate["label"]),
            str(rate["role"]),
            f"${rate['input_per_million']:.2f}",
            f"${rate['output_per_million']:.2f}",
        )
    stage.out.print(families)
    stage.out.print(
        "[dim]Those prices are stored in claude_bedrock/catalog.py so the demo is deterministic. "
        "Check the Bedrock pricing page before you quote them.[/dim]"
    )


def hello(stage: Stage) -> None:
    say(
        stage.out,
        "One Claude request is a model, a system prompt, a messages list, and max_tokens. "
        "The Python client is AnthropicBedrock(aws_region=...). "
        "The instance role signs the call. There is no API key.",
    )
    payload = {
        "aws_region": stage.region,
        "model": stage.primary,
        "max_tokens": 200,
        "temperature": 0.2,
        "system": HELLO_SYSTEM,
        "messages": [{"role": "user", "content": HELLO_USER}],
    }
    show_json(stage.out, "messages.create", payload)
    show_json(
        stage.out,
        "InvokeModel body  ·  the raw AWS form of that request",
        anthropic_invoke_body(HELLO_SYSTEM, HELLO_USER, 200),
    )

    if stage.offline or stage.runtime is None or not stage.primary:
        stage.out.print(Panel(
            "Claude is Anthropic's model family. This EC2 reaches it with the instance role. "
            "There is no API key in this repo.",
            title="Offline sample",
            border_style="green",
        ))
        return

    def _call():
        turn = complete(
            stage.runtime,
            stage.primary,
            payload["messages"],
            HELLO_SYSTEM,
            max_tokens=200,
            temperature=0.2,
        )
        stage.meter.add(turn.usage, "hello")
        stage.out.print(Panel(turn.text, title=stage.primary, border_style="green"))
        stage.out.print(f"[dim]{usage_line(turn.usage, turn.stop_reason)}[/dim]")
        return turn

    stage.guard("Hello", _call)


def conversation(stage: Stage) -> None:
    say(
        stage.out,
        "Every call is stateless. You send the whole conversation back. "
        "The second call spends more input tokens because the first answer is now part of the prompt.",
    )
    system = "You explain Claude to engineers. Stay under twenty-five words."
    first_user = "What is a Claude message?"
    second_user = "Say that again for someone who has only built REST APIs."
    messages = [{"role": "user", "content": first_user}]

    if stage.offline or stage.runtime is None or not stage.primary:
        messages.append({
            "role": "assistant",
            "content": "A message is one turn: a role, and the content of that turn.",
        })
        messages.append({"role": "user", "content": second_user})
        show_json(stage.out, "What you resend on turn 2", {"messages": messages})
        stage.out.print(Panel(
            "A message is one turn in the list you send back: who spoke, and the text they sent.",
            title="Offline sample",
            border_style="green",
        ))
        return

    def _call():
        first = complete(stage.runtime, stage.primary, messages, system, max_tokens=80, temperature=0.2)
        stage.meter.add(first.usage, "turn 1")
        stage.out.print(Panel(first.text, title="Turn 1", border_style="green"))
        stage.out.print(f"[dim]{usage_line(first.usage, first.stop_reason)}[/dim]")

        messages.append({"role": "assistant", "content": first.text})
        messages.append({"role": "user", "content": second_user})
        show_json(stage.out, "Turn 2 sends the history, not a memory id", {"messages": messages})

        second = complete(stage.runtime, stage.primary, messages, system, max_tokens=80, temperature=0.2)
        stage.meter.add(second.usage, "turn 2")
        stage.out.print(Panel(second.text, title="Turn 2", border_style="green"))
        stage.out.print(f"[dim]{usage_line(second.usage, second.stop_reason)}[/dim]")
        stage.out.print(
            f"[bold]Input tokens {first.usage.input_tokens} → {second.usage.input_tokens}.[/bold] "
            "The growth is the history you just attached."
        )

    stage.guard("Conversation", _call)


def streaming(stage: Stage) -> None:
    say(
        stage.out,
        "Streaming is the same request. The service sends tokens as they are ready, "
        "so a demo and a product both feel alive. The usage arrives after the text.",
    )
    system = "You explain cloud security to engineers. Four short sentences."
    user = (
        "Why is an EC2 instance role a better way to call Claude than an API key committed to GitHub?"
    )
    if stage.offline or stage.runtime is None or not stage.primary:
        sample = (
            "The instance role is issued by AWS and rotated for you. "
            "A key in GitHub is a long-lived secret anyone with the repo can copy. "
            "Bedrock authorizes the role, so the application never handles a credential. "
            "Take the role away and the calls stop."
        )
        for word in sample.split(" "):
            stage.out.print(word + " ", end="")
            time.sleep(0.02)
        stage.out.print("\n[dim]offline stream[/dim]")
        return

    def _call():
        chunks: list[str] = []

        def on_text(piece: str) -> None:
            chunks.append(piece)
            stage.out.print(piece, end="")

        usage, stop = stream_text(
            stage.runtime,
            stage.primary,
            [{"role": "user", "content": user}],
            system,
            on_text,
            max_tokens=220,
            temperature=0.3,
        )
        stage.meter.add(usage, "stream")
        stage.out.print()
        stage.out.print(f"[dim]{usage_line(usage, stop)}[/dim]")

    stage.guard("Stream", _call)


def tools(stage: Stage) -> None:
    say(
        stage.out,
        "Tool use is how Claude touches this machine. The model returns a tool name and JSON. "
        "Our code runs the function. We send the result back. Claude writes the answer. "
        "That loop is an agent.",
    )
    stage.out.print(Panel(USER, title="User", border_style="white"))
    if stage.offline or stage.runtime is None or not stage.primary:
        machine = {
            "hostname": "rehearsal",
            "region": stage.region,
            "credential": f"IAM role {ROLE_NAME}. No API key in this repo.",
        }
        sonnet = estimate_cost("sonnet", 8000, 1000)
        show_json(stage.out, "tool inspect_machine", machine)
        show_json(stage.out, "tool estimate_cost  ·  sonnet", sonnet)
        stage.out.print(Panel(
            "Use Sonnet for the live assistant and Haiku for the batch classifier. "
            f"Eight thousand input tokens and one thousand output tokens on Sonnet "
            f"are about ${sonnet['total_usd']:.4f} at the classroom rates.",
            title="Offline sample",
            border_style="green",
        ))
        return

    def _call():
        result = run_agent(stage.runtime, stage.primary, stage.region)
        for usage in result.usages:
            stage.meter.add(usage, "tool turn")
        for event in result.events:
            show_json(
                stage.out,
                f"tool {event.name}",
                {"input": event.tool_input, "output": event.output},
            )
        stage.out.print(Panel(result.answer or "(no text)", title="Claude, after the tools", border_style="green"))
        stage.out.print(f"[dim]stop {result.stop_reason}  ·  {len(result.events)} tool call(s)[/dim]")

    stage.guard("Tools", _call)


def structured(stage: Stage) -> None:
    say(
        stage.out,
        "When you need JSON, pin toolChoice to one tool. The tool input is the object, ready to use.",
    )
    tool = outline_tool()
    if stage.offline or stage.runtime is None or not stage.primary:
        show_json(stage.out, "Forced tool", {"tool_choice": {"type": "tool", "name": "emit_session_card"}})
        show_json(stage.out, "Offline card", {
            "title": "Claude on Amazon Bedrock",
            "duration_minutes": 20,
            "beats": [
                {"minute": 0, "title": "The doors", "point": "Same model, several credentials."},
                {"minute": 4, "title": "One request", "point": "System, messages, max tokens."},
                {"minute": 12, "title": "Tools", "point": "The model asks, your code answers."},
            ],
        })
        return

    def _call():
        turn = complete(
            stage.runtime,
            stage.primary,
            [{
                "role": "user",
                "content": (
                    "Build a 20 minute outline titled Claude on Amazon Bedrock. "
                    "Five beats. Each point is one sentence a room can hear."
                ),
            }],
            "Call emit_session_card. Put the outline in the tool input.",
            max_tokens=700,
            temperature=0,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_session_card"},
        )
        stage.meter.add(turn.usage, "structured")
        card = None
        for block in getattr(turn.message, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                card = block.input
        if isinstance(card, str):
            card = json.loads(card)
        if isinstance(card, dict):
            table = Table(title=str(card.get("title", "Outline")), header_style="bold")
            table.add_column("Min", justify="right")
            table.add_column("Beat")
            table.add_column("Point")
            for beat in card.get("beats", []):
                table.add_row(str(beat.get("minute", "")), str(beat.get("title", "")), str(beat.get("point", "")))
            stage.out.print(table)
        else:
            stage.out.print(Panel(turn.text or "No card returned", title="Structured", border_style="yellow"))
        stage.out.print(f"[dim]{usage_line(turn.usage, turn.stop_reason)}[/dim]")

    stage.guard("Structured", _call)


def _model_table(model_id: str) -> Table:
    table = Table(title="One model id, several jobs", header_style="bold")
    table.add_column("Job")
    table.add_column("What it does")
    table.add_column("Model id")
    for name, purpose in (
        ("Chat", "A person asks a question"),
        ("Classification", "Label a batch of short texts"),
        ("Summary", "Condense a long document"),
    ):
        table.add_row(name, purpose, model_id)
    return table


def families(stage: Stage) -> None:
    say(
        stage.out,
        "Haiku, Sonnet, and Opus are the families. This session uses one Sonnet id for chat, "
        "classification, and summaries. You switch families by changing the model id. "
        f"The instance role is {ROLE_NAME}.",
    )
    stage.out.print(_model_table(stage.primary or MODEL_ID))
    prompt = "Write a one-line git commit message for adding Bedrock streaming to a Python service."
    system = "Reply with the commit message only."
    if not stage.fast and (stage.offline or stage.runtime is None or not stage.primary):
        return

    if not stage.fast:
        def _one_job():
            turn = complete(
                stage.runtime,
                stage.primary,
                [{
                    "role": "user",
                    "content": (
                        "This service uses one Sonnet model for chat, classification, and summaries. "
                        "In two sentences, when would you switch to Haiku instead?"
                    ),
                }],
                "Be brief and concrete.",
                max_tokens=160,
                temperature=0.2,
            )
            stage.meter.add(turn.usage, "one-model")
            stage.out.print(Panel(turn.text, title=stage.primary, border_style="green"))
            stage.out.print(f"[dim]{usage_line(turn.usage, turn.stop_reason)}[/dim]")

        stage.guard("One model", _one_job)
        return

    if stage.offline or stage.runtime is None or not stage.primary:
        return

    def _one(model_id: str) -> tuple[str, Usage] | None:
        turn = complete(
            stage.runtime,
            model_id,
            [{"role": "user", "content": prompt}],
            system,
            max_tokens=60,
            temperature=0.2,
        )
        stage.meter.add(turn.usage, family_of(model_id))
        return turn.text, turn.usage

    def _call():
        table = Table(title="Same prompt, two model ids", header_style="bold")
        table.add_column("Model")
        table.add_column("Latency", justify="right")
        table.add_column("Out", justify="right")
        table.add_column("Answer")
        for model_id in (stage.fast, stage.primary):
            try:
                answered = _one(model_id)
            except Exception as exc:  # noqa: BLE001
                table.add_row(model_id, "—", "—", explain_error(exc).split("\n", 1)[0])
                continue
            if answered is None:
                continue
            text, usage = answered
            table.add_row(model_id, f"{usage.latency_ms} ms", str(usage.output_tokens), text)
        stage.out.print(table)

    stage.guard("Families", _call)


ACTS = [
    ("map", "The ecosystem", ecosystem),
    ("hello", "One request", hello),
    ("conversation", "The conversation is the memory", conversation),
    ("stream", "Streaming", streaming),
    ("tools", "Tools, the agent loop", tools),
    ("structured", "Structured output", structured),
    ("families", "One model id", families),
]
