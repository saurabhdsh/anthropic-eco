# Claude on Amazon Bedrock

A live Python session for the Anthropic ecosystem. It uses the Claude access SEAL already runs on EC2: instance role `WeaveEC2BedrockRole`, region `us-east-1`, model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. `ANTHROPIC_API_KEY` stays empty. The terminal is the slide deck.

## What the room will see

| Beat | Command | Idea |
| --- | --- | --- |
| The ecosystem | `map` | Claude.ai, the Claude API, Bedrock Converse, InvokeModel, Mantle, MCP |
| One request | `hello` | The Messages call SEAL's `BedrockProvider` makes |
| Memory | `conversation` | You resend history. Input tokens grow. |
| Streaming | `stream` | Same request, tokens as they are ready |
| Tools | `tools` | Claude calls Python on this machine, then answers |
| Structured output | `structured` | A pinned tool is a JSON schema |
| One model | `families` | SEAL uses that Sonnet id for generation, evaluation, and critic |

Classroom prices live in `claude_bedrock/catalog.py`. Update them from the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) if you will quote a real invoice.

## 20 minute run of show

1. **The doors (3 min).** Claude is one family. The credential changes: an account, an API key, or an IAM role. This EC2 uses the role.
2. **One request (3 min).** Open `examples/seal_messages.py`. It is the Python twin of `server/src/services/ai/bedrockProvider.ts`: `AnthropicBedrock({ awsRegion })`, then `messages.create`.
3. **Stateless (3 min).** Turn 2 includes turn 1. Point at the input token count.
4. **Streaming (2 min).** The usage line arrives after the words.
5. **Tools (5 min).** This is the agent loop. The model picks `inspect_machine`, `describe_model_family`, and `estimate_cost`. Your code runs them.
6. **Schema (2 min).** `toolChoice` pinned to `emit_session_card`.
7. **One model (2 min).** Generation, evaluation, and critic in SEAL all point at the same inference profile.

## Put it on EC2 from GitHub

On your Mac, create an empty GitHub repo, then from this folder:

```bash
git init
git add .
git commit -m "Add the Claude on Bedrock session"
git branch -M main
git remote add origin git@github.com:YOU/anthropic-eco.git
git push -u origin main
```

Use the same EC2 as SEAL. The instance profile is already `WeaveEC2BedrockRole`, and Claude Sonnet 4.5 is already enabled in `us-east-1`. Run the session on the host so it inherits that role. `iam/bedrock-session-policy.json` is the invoke permission that role needs, for reference.

```bash
git clone git@github.com:YOU/anthropic-eco.git
cd anthropic-eco
bash scripts/setup.sh
source .venv/bin/activate
```

`.env.example` already matches SEAL: `AWS_REGION=us-east-1` and `BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0`.

```bash
python session.py --list-models
python session.py
```

The live calls use that model id. `--list-models` only prints what else the account can see.

Rehearse on the Mac with no AWS account:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python session.py --offline --auto
```

## Useful commands

```bash
python session.py --act tools          # one beat, live
python session.py --auto               # live, no Enter between beats
python examples/seal_messages.py      # the SEAL BedrockProvider, in Python
python examples/tool_use.py            # the agent loop alone
python -m unittest discover -s tests -v
```

## How the code maps to the talk

```text
session.py                     the runner
claude_bedrock/lessons.py      the seven beats
claude_bedrock/agent.py        the tool loop to project
claude_bedrock/seal_access.py  role, region, and the model id copied from SEAL
claude_bedrock/catalog.py      families, rates, InvokeModel JSON
examples/seal_messages.py      smallest live call, same shape as BedrockProvider
iam/bedrock-session-policy.json
```

SEAL reaches Claude like this:

- `BEDROCK_ENABLED=true` wins over `AI_PROVIDER=anthropic`.
- `ANTHROPIC_API_KEY` is empty.
- `BedrockProvider` constructs `AnthropicBedrock({ awsRegion: "us-east-1" })`.
- `messages.create` uses `us.anthropic.claude-sonnet-4-5-20250929-v1:0` for generation, evaluation, and critic.
- Cost in SEAL is the Sonnet classroom rate, $3 / $15 per million tokens.

This session sends that same `messages.create`. The hello beat also prints the InvokeModel body (`anthropic_version: bedrock-2023-05-31`) so the room can see the raw AWS form.

## If a live call fails

| What you see | What to do |
| --- | --- |
| `NoCredentialsError` | Run on the SEAL EC2. The profile is `WeaveEC2BedrockRole`. |
| `AccessDeniedException` / `PermissionDeniedError` | This caller is not that role. SEAL's API uses the instance profile. |
| model not found | Keep `BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0` and `AWS_REGION=us-east-1`. |
