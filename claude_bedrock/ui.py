"""Terminal presentation helpers."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def console() -> Console:
    return Console()


def banner(out: Console) -> None:
    body = Text.from_markup(
        "[bold]Claude  ×  Amazon Bedrock[/bold]\n"
        "A live session from this machine.\n"
        "[dim]Instance role  ·  Messages API  ·  no API key[/dim]"
    )
    out.print(Panel(body, border_style="bright_cyan", padding=(1, 2)))


def say(out: Console, line: str) -> None:
    out.print(Panel(line, title="Say", border_style="cyan", padding=(0, 1)))


def show_json(out: Console, title: str, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, default=str)
    out.print(Panel(Syntax(rendered, "json", theme="monokai", word_wrap=True), title=title))


def pause(out: Console, auto: bool) -> None:
    if auto:
        out.print()
        return
    out.print("[dim]Press Enter for the next beat.[/dim]")
    try:
        input()
    except EOFError:
        return


def usage_line(usage, stop_reason: str | None = None) -> str:
    bits = [
        f"in {usage.input_tokens}",
        f"out {usage.output_tokens}",
        f"{usage.latency_ms} ms",
    ]
    if stop_reason:
        bits.append(f"stop {stop_reason}")
    return "  ·  ".join(bits)


def meter_table(meter) -> Table:
    table = Table(title="What this session spent", show_header=True, header_style="bold")
    table.add_column("Calls", justify="right")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_row(str(meter.calls), str(meter.input_tokens), str(meter.output_tokens))
    return table
