"""Run only the tool-use loop. The annotated version is claude_bedrock/agent.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_bedrock.agent import run_agent
from claude_bedrock.envfile import load_dotenv, setting
from claude_bedrock.runtime import client
from claude_bedrock.seal_access import MODEL_ID, REGION


def main() -> None:
    load_dotenv()
    region = setting("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or REGION
    model_id = setting("BEDROCK_MODEL_ID") or setting("CLAUDE_MODEL") or MODEL_ID
    result = run_agent(client(region), model_id, region)
    for event in result.events:
        print(f"\n== {event.name} ==")
        print("input ", event.tool_input)
        print("output", event.output)
    print("\n== answer ==")
    print(result.answer)


if __name__ == "__main__":
    main()
