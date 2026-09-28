"""Choose a Claude inference profile id for this region."""

from __future__ import annotations


def geo_prefix(region: str) -> str:
    if region.startswith("eu-"):
        return "eu."
    if region.startswith("us-") or region.startswith("ca-") or region.startswith("sa-"):
        return "us."
    if region.startswith(("ap-", "me-", "af-", "il-")):
        return "apac."
    return "global."


def _rank(profile_id: str, region: str) -> tuple[int, str]:
    prefix = geo_prefix(region)
    if profile_id.startswith(prefix):
        return (0, profile_id)
    if profile_id.startswith("global."):
        return (1, profile_id)
    return (2, profile_id)


def pick_profile(profile_ids: list[str], preferences: list[str], region: str) -> str | None:
    """Return the best id that contains the earliest matching preference."""
    for needle in preferences:
        matches = [profile_id for profile_id in profile_ids if needle in profile_id]
        if matches:
            return sorted(matches, key=lambda profile_id: _rank(profile_id, region))[0]
    return None
