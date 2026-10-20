"""Live checks against the real API. Every call is a dry run, so nothing is charged.

Run with LAYRCAKE_API_KEY set: `uv run pytest -m live tests/integration_tests`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_layrcake import LayrcakeFindLeads, LayrcakeSession, LayrcakeToolkit, LayrcakeVerifyEmail

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LAYRCAKE_API_KEY"), reason="LAYRCAKE_API_KEY is not set"),
]


def _session() -> LayrcakeSession:
    return LayrcakeSession(dry_run=True, budget_credits=50)


class TestFindLeadsIntegration(ToolsIntegrationTests):
    tool_constructor = LayrcakeFindLeads
    tool_invoke_params_example = {
        "audience": "Heads of sales at UK software companies",
        "titles": ["Head of Sales"],
        "locations": ["United Kingdom"],
    }

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"session": _session()}


class TestVerifyEmailIntegration(ToolsIntegrationTests):
    tool_constructor = LayrcakeVerifyEmail
    tool_invoke_params_example = {"email": "founders@layrcake.dev"}

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"session": _session()}


def test_every_paid_tool_returns_a_free_plan() -> None:
    tools = {tool.name: tool for tool in LayrcakeToolkit(dry_run=True, budget_credits=50).get_tools()}
    calls: dict[str, dict[str, Any]] = {
        "find_leads": {"audience": "Heads of sales at UK software companies", "titles": ["Head of Sales"]},
        "enrich": {"full_name": "Ada Lovelace", "company_domain": "layrcake.dev"},
        "verify_email": {"email": "founders@layrcake.dev"},
        "check_companies": {"question": "Do they sell an API for AI agents?", "company_domains": ["layrcake.dev"]},
        "research_company": {"company_domain": "layrcake.dev"},
        "create_campaign": {
            "name": "Dry run",
            "leads": [{"email": "founders@layrcake.dev"}],
            "steps": [{"subject": "Hello", "body": "Hi {{first_name}}"}],
        },
    }
    for name, args in calls.items():
        output = json.loads(tools[name].invoke(args))
        assert output["dry_run"] is True, (name, output)
        print(name, output)
    assert tools["find_leads"].session.spent == 0  # type: ignore[attr-defined]
