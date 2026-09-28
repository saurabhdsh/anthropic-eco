"""Pure tests. They do not call AWS."""

from __future__ import annotations

import unittest

from claude_bedrock.agent import dispatch
from claude_bedrock.catalog import anthropic_invoke_body, estimate_cost, family_of
from claude_bedrock.seal_access import MODEL_ID, REGION, ROLE_NAME
from claude_bedrock.select import geo_prefix, pick_profile


PROFILES = [
    "us.anthropic.claude-haiku-4-5",
    "eu.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-opus-4-7",
]


class SelectTests(unittest.TestCase):
    def test_geo_prefix(self):
        self.assertEqual(geo_prefix("us-east-1"), "us.")
        self.assertEqual(geo_prefix("eu-west-1"), "eu.")
        self.assertEqual(geo_prefix("ap-south-1"), "apac.")

    def test_prefers_the_region_profile(self):
        chosen = pick_profile(
            PROFILES,
            ["claude-sonnet-4-6", "claude-sonnet"],
            "us-east-1",
        )
        self.assertEqual(chosen, "us.anthropic.claude-sonnet-4-6")

    def test_europe_picks_the_eu_profile(self):
        chosen = pick_profile(PROFILES, ["claude-sonnet-4-6"], "eu-west-1")
        self.assertEqual(chosen, "eu.anthropic.claude-sonnet-4-6")

    def test_falls_through_preferences(self):
        chosen = pick_profile(PROFILES, ["claude-haiku-4-5", "claude-haiku"], "us-east-1")
        self.assertEqual(chosen, "us.anthropic.claude-haiku-4-5")

    def test_missing_profile_returns_none(self):
        self.assertIsNone(pick_profile(PROFILES, ["claude-mythos"], "us-east-1"))


class SealAccessTests(unittest.TestCase):
    def test_matches_the_ec2_setup(self):
        self.assertEqual(MODEL_ID, "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        self.assertEqual(REGION, "us-east-1")
        self.assertEqual(ROLE_NAME, "WeaveEC2BedrockRole")


class CatalogTests(unittest.TestCase):
    def test_family_from_model_id(self):
        self.assertEqual(family_of("us.anthropic.claude-haiku-4-5"), "haiku")
        self.assertEqual(family_of("anthropic.claude-opus-4-7"), "opus")
        self.assertEqual(family_of("anthropic.claude-sonnet-4-6"), "sonnet")

    def test_cost_is_per_million(self):
        cost = estimate_cost("sonnet", 8000, 1000)
        # 8000/1e6 * 3 + 1000/1e6 * 15 = 0.024 + 0.015
        self.assertAlmostEqual(cost["total_usd"], 0.039, places=6)

    def test_invoke_body_uses_anthropic_content_blocks(self):
        body = anthropic_invoke_body("be brief", "hello", 100)
        self.assertEqual(body["anthropic_version"], "bedrock-2023-05-31")
        self.assertEqual(body["messages"][0]["content"][0]["type"], "text")


class ToolTests(unittest.TestCase):
    def test_dispatch_cost_and_unknown_tool(self):
        cost = dispatch(
            "estimate_cost",
            {"family": "Haiku", "input_tokens": 1000, "output_tokens": 100},
            "us-east-1",
        )
        self.assertEqual(cost["family"], "haiku")
        self.assertIn("total_usd", cost)
        unknown = dispatch("weather", {}, "us-east-1")
        self.assertIn("error", unknown)

    def test_machine_snapshot_has_no_secret_fields(self):
        snap = dispatch("inspect_machine", {}, "us-east-1")
        self.assertIn("hostname", snap)
        self.assertNotIn("secret", snap)
        self.assertNotIn("credentials", snap)


if __name__ == "__main__":
    unittest.main()
