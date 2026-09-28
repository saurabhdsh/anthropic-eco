"""Retrieval, then generation. One question, a few local passages, one Claude call.

Run on the EC2 host:

    python examples/rag.py

The instance role is the credential. There is no API key.
"""

from anthropic import AnthropicBedrock

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# The private notes. Claude does not see this list until we retrieve from it.
PASSAGES = [
    "Haiku is the fast, inexpensive Claude family. Use it as a high-volume classifier, and for routing.",
    "Sonnet is the default Claude family. Use it for coding, long documents, and most product features.",
    "Opus is the Claude family for the hardest reasoning. It costs more than Haiku and Sonnet.",
    "On Amazon Bedrock, the EC2 instance role signs the call. The application holds no API key.",
    "A Claude message is one turn. It has a role, user or assistant, and the content of that turn.",
    "RAG retrieves a few relevant passages first, then asks the model to answer from those passages.",
]

STOP = {"a", "an", "the", "is", "for", "and", "to", "of", "on", "it", "in", "i", "should"}


def retrieve(question: str, limit: int = 2) -> list[str]:
    """Rank passages by how many question words they contain."""
    words = [word for word in question.lower().split() if word not in STOP]
    ranked = []
    for passage in PASSAGES:
        text = passage.lower()
        score = sum(1 for word in words if word in text)
        ranked.append((score, passage))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [passage for score, passage in ranked[:limit] if score > 0] or PASSAGES[:limit]


def main() -> None:
    # 1. The person types one question. Press Enter to keep the sample.
    question = input("Type a question: ").strip()
    if not question:
        question = "Which Claude family should I use for a high-volume classifier?"

    # 2. Retrieve. This step does not call Claude.
    passages = retrieve(question)
    print("\n--- 2. retrieve ---")
    for number, passage in enumerate(passages, start=1):
        print(f"[{number}] {passage}")

    # 3. Augment. The retrieved passages become part of the user message.
    notes = "\n".join(f"[{number}] {passage}" for number, passage in enumerate(passages, start=1))
    user_message = (
        "Answer using only the passages below. "
        "If they do not contain the answer, say so.\n\n"
        f"Passages:\n{notes}\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": user_message}]

    print("\n--- 3. augment ---\n")
    print(user_message)

    # 4. Generate. One Claude call on Bedrock.
    client = AnthropicBedrock(aws_region=REGION)
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=200,
        system="You answer from the passages you are given. Two short sentences.",
        messages=messages,
    )

    print("\n--- 4. generate ---")
    print(response.content[0].text)
    print(
        f"\n{MODEL_ID}  in={response.usage.input_tokens} "
        f"out={response.usage.output_tokens}"
    )


if __name__ == "__main__":
    main()
