"""Ask Bedrock which Claude model ids this account can name."""

from __future__ import annotations

from claude_bedrock.catalog import HAIKU_PREFERENCES, SONNET_PREFERENCES
from claude_bedrock.select import pick_profile


def _pages(client, method: str, item_key: str):
    token = None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = getattr(client, method)(**kwargs)
        yield from page.get(item_key, [])
        token = page.get("nextToken")
        if not token:
            break


def claude_targets(region: str) -> dict[str, list[str]]:
    import boto3

    control = boto3.client("bedrock", region_name=region)
    profiles: list[str] = []
    for item in _pages(control, "list_inference_profiles", "inferenceProfileSummaries"):
        profile_id = item.get("inferenceProfileId", "")
        status = item.get("status", "ACTIVE")
        if status not in {"ACTIVE", None, ""}:
            continue
        if "anthropic.claude" in profile_id or "claude" in profile_id:
            profiles.append(profile_id)

    models: list[str] = []
    response = control.list_foundation_models(byProvider="Anthropic")
    for item in response.get("modelSummaries", []):
        model_id = item.get("modelId", "")
        if "claude" in model_id:
            models.append(model_id)
    return {"profiles": profiles, "models": models}


def resolve_models(region: str, primary: str | None, fast: str | None) -> dict[str, str | None]:
    """Prefer an explicit id, then an inference profile, then a foundation model."""
    if primary and fast:
        return {"primary": primary, "fast": fast, "source": "environment"}

    discovered = claude_targets(region)
    # Inference profiles are the ids Converse accepts in most regions.
    pool = discovered["profiles"] or discovered["models"]
    chosen_primary = primary or pick_profile(pool, SONNET_PREFERENCES, region)
    chosen_fast = fast or pick_profile(pool, HAIKU_PREFERENCES, region)
    source = "inference profile" if discovered["profiles"] else "foundation model"
    if not discovered["profiles"] and not discovered["models"]:
        source = "none found"
    return {
        "primary": chosen_primary,
        "fast": chosen_fast,
        "source": source,
        "profiles": discovered["profiles"],
        "models": discovered["models"],
    }
