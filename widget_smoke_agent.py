#!/usr/bin/env python3
"""
TokenPulse Mini — simple widget smoke agent (AWS Bedrock via AWS CLI profile).

What it does:
  1. Calls Bedrock with a tiny prompt (real tokens)
  2. Posts those tokens to TokenPulse local capture (widget updates NOW)
  3. Prints exactly how much Today Tokens should increase

Cost Explorer / Sync may still show 0 for today — that is normal (CE lag).

Prereqs (Bedrock Mac):
  - TokenPulse Mini running with Capture ON (http://127.0.0.1:8787)
  - AWS CLI configured: `aws sts get-caller-identity` works
  - pip install boto3

Usage:
  cd tests/regression
  python3 widget_smoke_agent.py
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ── Config (AWS CLI profile — leave keys empty) ───────────────────────────────
AWS_PROFILE = "default"
AWS_REGION = "us-east-1"

# Model to invoke (must be enabled in Bedrock Model access for this region)
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# Fallbacks if MODEL_ID isn't available
MODEL_FALLBACKS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-micro-v1:0",
]

PROMPT = "Say only the word: ok"
MAX_TOKENS = 8

CAPTURE_URL = "http://127.0.0.1:8787/v1/capture"
RUN_ID = "widget-smoke"


def session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def pick_model(available: list[str]) -> str:
    # Prefer configured MODEL_ID even if not in list_foundation_models
    # (inference profiles like us.anthropic.* often omit from that list).
    if MODEL_ID:
        return MODEL_ID
    for m in MODEL_FALLBACKS:
        if m in available:
            return m
    return available[0] if available else MODEL_ID


def list_models(sess: boto3.Session) -> list[str]:
    try:
        resp = sess.client("bedrock").list_foundation_models()
    except (ClientError, BotoCoreError):
        return []
    out: list[str] = []
    for summary in resp.get("modelSummaries", []):
        mid = summary.get("modelId")
        if mid:
            out.append(mid)
    return out


def invoke(sess: boto3.Session, model_id: str) -> dict[str, Any]:
    runtime = sess.client("bedrock-runtime")
    last_err: Exception | None = None
    candidates = [model_id] + [m for m in MODEL_FALLBACKS if m != model_id]
    for candidate in candidates:
        try:
            resp = runtime.converse(
                modelId=candidate,
                messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS},
            )
            usage = resp.get("usage") or {}
            reply = ""
            for block in (resp.get("output") or {}).get("message", {}).get("content", []):
                if "text" in block:
                    reply = block["text"].strip()
                    break
            prompt_t = int(usage.get("inputTokens") or 0)
            completion_t = int(usage.get("outputTokens") or 0)
            total = int(usage.get("totalTokens") or prompt_t + completion_t)
            return {
                "model": candidate,
                "reply": reply,
                "prompt_tokens": prompt_t,
                "completion_tokens": completion_t,
                "total_tokens": total,
            }
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"All models failed; last error: {last_err}")


def post_capture(result: dict[str, Any], iteration: int = 1) -> str:
    payload = {
        "provider": "AWS Bedrock",
        "model": result["model"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "run_id": RUN_ID,
        "iteration": iteration,
    }
    req = urllib.request.Request(
        CAPTURE_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode()


def main() -> int:
    print("\n══ TokenPulse Widget Smoke Agent (Bedrock + AWS CLI) ══\n")
    print(f"  Profile : {AWS_PROFILE}")
    print(f"  Region  : {AWS_REGION}")
    print(f"  Prompt  : {PROMPT!r}")
    print(f"  Capture : {CAPTURE_URL}\n")

    # 1) Health
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/health", timeout=3) as r:
            print(f"  ✓ TokenPulse capture: {r.read().decode()}")
    except Exception as exc:
        print(f"  ✗ TokenPulse capture not reachable: {exc}")
        print("    → Open TokenPulse v0.1.4+ and turn Capture ON")
        return 1

    # 2) AWS CLI identity
    sess = session()
    try:
        identity = sess.client("sts").get_caller_identity()
        print(f"  ✓ AWS CLI OK: {identity.get('Arn')} ({identity.get('Account')})")
    except Exception as exc:
        print(f"  ✗ AWS CLI credentials failed: {exc}")
        print("    → Fix with: aws configure   then retry")
        return 1

    # 3) Invoke Bedrock
    models = list_models(sess)
    model = pick_model(models)
    print(f"\n  → Invoking Bedrock model: {model}")
    try:
        result = invoke(sess, model)
    except Exception as exc:
        print(f"  ✗ Bedrock invoke failed: {exc}")
        print("    → Enable model access in AWS Console → Bedrock → Model access")
        return 1

    print(f"  ✓ Reply: {result['reply']!r}")
    print(
        f"  ✓ Tokens from Bedrock API: "
        f"{result['total_tokens']} "
        f"(input {result['prompt_tokens']} + output {result['completion_tokens']})"
    )

    # 4) Push to TokenPulse
    try:
        capture_body = post_capture(result, iteration=1)
        print(f"  ✓ Posted to capture: {capture_body}")
    except Exception as exc:
        print(f"  ✗ Capture post failed: {exc}")
        return 1

    delta = result["total_tokens"]
    # Rough estimated cost — Sonnet-class rates (~$3 / $15 per 1M); Nova fallbacks cheaper
    model_l = str(result["model"]).lower()
    if "nova-micro" in model_l:
        in_rate, out_rate = 0.035, 0.14
    elif "nova" in model_l:
        in_rate, out_rate = 0.80, 3.20
    elif "haiku" in model_l:
        in_rate, out_rate = 0.25, 1.25
    else:
        # claude sonnet / default bedrock premium
        in_rate, out_rate = 3.0, 15.0
    est_cost = (result["prompt_tokens"] / 1_000_000) * in_rate + (
        result["completion_tokens"] / 1_000_000
    ) * out_rate
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  WHAT TO CHECK ON THE WIDGET")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  AWS Bedrock widget → Today Tokens should increase by about:  +{delta}")
    print(f"    (exactly {result['prompt_tokens']} input + {result['completion_tokens']} output)")
    print(f"  Today Cost should increase by about:  +${est_cost:.8f}")
    print(f"    (rates used for estimate: ${in_rate}/${out_rate} per 1M · model={result['model']})")
    print("    Use TokenPulse v0.1.6+ so micro-costs don’t show as $0.0000")
    print("  Burn Rate/hr and Est. Monthly move with that cost.")
    print()
    print("  Usage History → new row:")
    print("    Provider : AWS Bedrock")
    print("    Source   : local")
    print(f"    Project  : run:{RUN_ID}/iter:1")
    print(f"    Tokens   : {delta}")
    print()
    print("  Cost Explorer / Sync “today” may still be 0 — NORMAL (CE lag).")
    print("  Local capture = live meter · Sync = delayed billing check.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
