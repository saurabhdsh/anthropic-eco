"""Python twin of SEAL's Bedrock provider.

SEAL server/src/services/ai/bedrockProvider.ts:

    this.client = new AnthropicBedrock({ awsRegion: env.AWS_REGION });
    const response = await client.messages.create({
      model,            // us.anthropic.claude-sonnet-4-5-20250929-v1:0
      max_tokens,
      temperature,
      system: input.system,
      messages: [{ role: "user", content: input.user }],
    });

Run on the SEAL EC2, from the repo root:

    python examples/seal_messages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_bedrock.envfile import load_dotenv, setting
from claude_bedrock.runtime import client, complete
from claude_bedrock.seal_access import MODEL_ID, REGION

SYSTEM = "You explain Claude to engineers. Be brief and concrete."
USER = (
    "In two sentences, what is Claude, and why does SEAL call it "
    "through Amazon Bedrock with an EC2 role?"
)


def main() -> None:
    load_dotenv()
    region = setting("AWS_REGION") or REGION
    model_id = setting("BEDROCK_MODEL_ID") or setting("CLAUDE_MODEL") or MODEL_ID
    turn = complete(
        client(region),
        model_id,
        [{"role": "user", "content": USER}],
        SYSTEM,
        max_tokens=200,
        temperature=0.2,
    )
    print(turn.text)
    print(
        f"\n{model_id}  in={turn.usage.input_tokens} "
        f"out={turn.usage.output_tokens}  {turn.usage.latency_ms} ms"
    )


if __name__ == "__main__":
    main()
