"""One MCP server, one Claude call.

MCP is how a program advertises tools. Claude asks for a tool.
This file's client runs that tool on the server, then Claude answers.

Run on the EC2 host:

    python examples/mcp_demo.py

The instance role is the credential. There is no API key.
"""

import asyncio

from anthropic import AnthropicBedrock
from fastmcp import Client, FastMCP

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

RATES = {
    "haiku": "Haiku: $1 per million input tokens, $5 per million output tokens.",
    "sonnet": "Sonnet: $3 per million input tokens, $15 per million output tokens.",
    "opus": "Opus: $15 per million input tokens, $75 per million output tokens.",
}
FAMILIES = {
    "haiku": "Haiku is the fast, inexpensive family for classification and high volume.",
    "sonnet": "Sonnet is the default family for coding, documents, and most features.",
    "opus": "Opus is the family for the hardest reasoning.",
}

# The MCP server. These functions stay here. Claude does not import this file.
server = FastMCP("classroom")


@server.tool
def lookup_rate(family: str) -> str:
    """Return the classroom USD price for haiku, sonnet, or opus."""
    return RATES.get(family.lower(), "Unknown family. Choose haiku, sonnet, or opus.")


@server.tool
def describe_family(family: str) -> str:
    """Say when to use haiku, sonnet, or opus."""
    return FAMILIES.get(family.lower(), "Unknown family. Choose haiku, sonnet, or opus.")


def claude_tools(tools) -> list[dict]:
    """MCP tool descriptions, rewritten in the shape Claude expects."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


async def main() -> None:
    question = input("Type a question: ").strip()
    if not question:
        question = "What is the classroom output price of Sonnet, and when should I use Haiku?"

    # Client(server) speaks MCP in this process. Claude Desktop would use stdio instead.
    async with Client(server) as mcp:
        tools = await mcp.list_tools()
        print("\n--- 1. MCP server advertises tools ---")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        bedrock = AnthropicBedrock(aws_region=REGION)
        messages = [{"role": "user", "content": question}]
        system = (
            "You answer engineers. For a price or a model family, call a tool. "
            "Then answer in two short sentences."
        )

        # First turn must call a tool, so the room sees MCP run.
        response = bedrock.messages.create(
            model=MODEL_ID,
            max_tokens=400,
            system=system,
            tools=claude_tools(tools),
            tool_choice={"type": "any"},
            messages=messages,
        )

        print("\n--- 2. Claude asks the MCP server ---")
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"tool {block.name}  input {block.input}")
            result = await mcp.call_tool(block.name, dict(block.input))
            print(f"mcp result {result.data}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result.data),
                    "is_error": bool(result.is_error),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    if block.type == "tool_use"
                    else {"type": "text", "text": block.text}
                    for block in response.content
                    if block.type in {"tool_use", "text"}
                ],
            }
        )
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            follow = bedrock.messages.create(
                model=MODEL_ID,
                max_tokens=300,
                system=system,
                tools=claude_tools(tools),
                messages=messages,
            )
            answer = "\n".join(block.text for block in follow.content if block.type == "text")
        else:
            answer = "\n".join(block.text for block in response.content if block.type == "text")

    print("\n--- 3. Claude answers ---")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
