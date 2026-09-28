"""One typed message to Claude on Bedrock.

Run on the EC2 host:

    python examples/simple_message.py

The instance role is the credential. There is no API key.
"""

from anthropic import AnthropicBedrock

# Bedrock in us-east-1. The instance role is the credential.
REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# 1. System: instructions that stay outside the chat.
system = "You explain Claude to engineers. Answer in two short sentences."

# 2. The person types one message. Press Enter to keep the sample sentence.
typed = input("Type a message: ").strip()
if not typed:
    typed = "In one sentence, what is a Claude message?"

# 3. Messages: a list of turns. This call has one turn, from the user.
messages = [
    {
        "role": "user",
        "content": typed,
    }
]

print("\n--- request structure ---")
print("model   ", MODEL_ID)
print("system  ", system)
print("messages", messages)

# 4. The client signs the call with the EC2 role.
client = AnthropicBedrock(aws_region=REGION)

response = client.messages.create(
    model=MODEL_ID,
    max_tokens=200,
    system=system,
    messages=messages,
)

print("\n--- response structure ---")
print("stop_reason   ", response.stop_reason)
print("input_tokens  ", response.usage.input_tokens)
print("output_tokens ", response.usage.output_tokens)
print("content       ", response.content[0].text)
