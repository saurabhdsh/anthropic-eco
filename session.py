#!/usr/bin/env python3
"""Run the Claude on Bedrock session.

    python session.py                 # live, pause between beats
    python session.py --auto          # live, no pauses
    python session.py --offline       # rehearse on a laptop, no AWS
    python session.py --act tools     # one beat
    python session.py --list-models   # what this account can call
"""

from __future__ import annotations

import argparse
import sys

from rich.panel import Panel
from rich.table import Table

from claude_bedrock.envfile import load_dotenv, setting
from claude_bedrock.lessons import ACTS, Stage
from claude_bedrock.runtime import Meter, caller_identity, client, explain_error
from claude_bedrock.seal_access import MODEL_ID, REGION, ROLE_NAME
from claude_bedrock.ui import banner, meter_table, pause


def resolve_region() -> str:
    region = setting("AWS_REGION") or setting("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        import boto3

        return boto3.session.Session().region_name or REGION
    except Exception:
        return REGION


def parse_args() -> argparse.Namespace:
    names = [name for name, _, _ in ACTS]
    parser = argparse.ArgumentParser(description="Claude on Amazon Bedrock, live from this machine.")
    parser.add_argument("--offline", action="store_true", help="Narrate and print payloads. Do not call AWS.")
    parser.add_argument("--auto", action="store_true", help="Run every beat without waiting for Enter.")
    parser.add_argument("--act", choices=names, help="Run a single beat.")
    parser.add_argument("--list-models", action="store_true", help="Print Claude ids visible in this region.")
    return parser.parse_args()


def show_models(out, region: str) -> int:
    from claude_bedrock.discover import resolve_models

    try:
        found = resolve_models(region, chosen_model(), setting("CLAUDE_FAST_MODEL"))
    except Exception as exc:  # noqa: BLE001
        out.print(Panel(explain_error(exc), title="List models", border_style="red"))
        return 1

    profiles = found.get("profiles") or []
    models = found.get("models") or []
    table = Table(title=f"Claude in {region}", header_style="bold")
    table.add_column("Kind")
    table.add_column("Id")
    for profile_id in profiles:
        table.add_row("inference profile", profile_id)
    for model_id in models:
        table.add_row("foundation model", model_id)
    if profiles or models:
        out.print(table)
    else:
        out.print(Panel("No Claude models were returned for this region.", border_style="yellow"))
    out.print(
        f"[bold]Session would use[/bold]  primary={found.get('primary')}  "
        f"fast={found.get('fast')}  source={found.get('source')}"
    )
    return 0 if found.get("primary") else 1


def chosen_model() -> str:
    return setting("BEDROCK_MODEL_ID") or setting("CLAUDE_MODEL") or MODEL_ID


def build_stage(out, region: str, offline: bool) -> tuple[Stage, int]:
    primary = chosen_model()
    fast = setting("CLAUDE_FAST_MODEL")
    if offline:
        out.print(Panel(
            f"Offline rehearsal in {region}.\n"
            f"Model {primary}\n"
            f"Role {ROLE_NAME}\n"
            "No AWS calls. The sample answers are local.",
            title="Stage check",
            border_style="yellow",
        ))
        return Stage(out, region, primary, fast, True, None, Meter(), None), 0

    try:
        identity = caller_identity(region)
    except Exception as exc:  # noqa: BLE001
        out.print(Panel(explain_error(exc), title="Stage check", border_style="red"))
        return Stage(out, region, None, None, False), 1

    on_expected_role = ROLE_NAME in identity["arn"]
    check = Table(title="Stage check", header_style="bold")
    check.add_column("Piece")
    check.add_column("Value")
    check.add_row("Account", identity["account"])
    check.add_row("Caller", identity["arn"])
    check.add_row(
        "Instance role",
        ROLE_NAME if on_expected_role else f"{ROLE_NAME}  (this caller is a different principal)",
    )
    check.add_row("Region", region)
    check.add_row("Model", primary)
    check.add_row("Fast model", fast or "not set — this session uses one Sonnet id")
    check.add_row("Credential", "Instance role. No API key in this repo.")
    out.print(check)
    return Stage(out, region, primary, fast, False, client(region), Meter(), identity), 0


def main() -> int:
    load_dotenv()
    args = parse_args()
    from claude_bedrock.ui import console as make_console

    out = make_console()
    region = resolve_region()
    banner(out)

    if args.list_models:
        return show_models(out, region)

    stage, status = build_stage(out, region, args.offline)
    if status != 0:
        return status

    selected = [act for act in ACTS if args.act in (None, act[0])]
    for index, (name, title, fn) in enumerate(selected, start=1):
        out.print(f"\n[bold bright_cyan]{index}  {title}[/bold bright_cyan]  [dim]{name}[/dim]")
        out.rule(style="bright_cyan")
        fn(stage)
        if index != len(selected):
            pause(out, args.auto)

    out.print()
    if stage.meter.calls:
        out.print(meter_table(stage.meter))
        out.print(Panel("\n".join(stage.meter.notes), title="Calls", border_style="dim"))
    else:
        out.print("[dim]No live calls on this run.[/dim]")
    out.print("[dim]Open examples/simple_message.py when someone asks for the smallest version.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
