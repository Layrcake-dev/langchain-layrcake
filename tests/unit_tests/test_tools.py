"""LangChain-specific behaviour: the standard tool tests, the toolkit and ToolCall handling."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests
from layrcake import Layrcake
from pydantic import BaseModel

from langchain_layrcake import (
    LayrcakeCheckCompanies,
    LayrcakeCreateCampaign,
    LayrcakeEnrich,
    LayrcakeFindLeads,
    LayrcakeGetMe,
    LayrcakeLaunchCampaign,
    LayrcakeResearchCompany,
    LayrcakeSession,
    LayrcakeToolkit,
    LayrcakeVerifyEmail,
)
from tests.unit_tests.conftest import APPROVAL_URL, CAMPAIGN_ID, MockLayrcake

_SESSION = LayrcakeSession(api_key="lc_test")
_ENV: tuple[dict[str, str], dict[str, Any], dict[str, Any]] = ({"LAYRCAKE_API_KEY": "lc_test"}, {}, {})


class _Standard(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"session": _SESSION}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return _ENV


class TestFindLeadsStandard(_Standard):
    tool_constructor = LayrcakeFindLeads
    tool_invoke_params_example = {"audience": "Heads of sales at UK software companies", "titles": ["Head of Sales"]}


class TestEnrichStandard(_Standard):
    tool_constructor = LayrcakeEnrich
    tool_invoke_params_example = {"full_name": "Ada Lovelace", "company_domain": "acme.com"}


class TestVerifyEmailStandard(_Standard):
    tool_constructor = LayrcakeVerifyEmail
    tool_invoke_params_example = {"email": "ada@acme.com"}


class TestCheckCompaniesStandard(_Standard):
    tool_constructor = LayrcakeCheckCompanies
    tool_invoke_params_example = {"question": "Do they sell to hospitals?", "company_domains": ["acme.com"]}


class TestResearchCompanyStandard(_Standard):
    tool_constructor = LayrcakeResearchCompany
    tool_invoke_params_example = {"company_domain": "acme.com"}


class TestCreateCampaignStandard(_Standard):
    tool_constructor = LayrcakeCreateCampaign
    tool_invoke_params_example = {
        "name": "Q4",
        "leads": [{"email": "ada@acme.com"}],
        "steps": [{"subject": "Hi", "body": "Hi {{first_name}}"}],
    }


class TestLaunchCampaignStandard(_Standard):
    tool_constructor = LayrcakeLaunchCampaign
    tool_invoke_params_example = {"campaign_id": CAMPAIGN_ID}


class TestGetMeStandard(_Standard):
    tool_constructor = LayrcakeGetMe
    tool_invoke_params_example: dict[str, Any] = {}


@pytest.fixture
def tools(client: Layrcake) -> dict[str, BaseTool]:
    toolkit = LayrcakeToolkit(client=client, budget_credits=100, run_id="run-test")
    return {tool.name: tool for tool in toolkit.get_tools()}


def test_toolkit_gives_eight_tools_sharing_one_run(tools: dict[str, BaseTool]) -> None:
    assert list(tools) == [
        "find_leads",
        "enrich",
        "verify_email",
        "check_companies",
        "research_company",
        "create_campaign",
        "launch_campaign",
        "get_me",
    ]
    sessions = {id(tool.session) for tool in tools.values()}  # type: ignore[attr-defined]
    assert len(sessions) == 1


def test_a_full_run_through_tool_calls_stops_at_approval(tools: dict[str, BaseTool], api: MockLayrcake) -> None:
    def tool_call(name: str, args: dict[str, Any]) -> Any:
        message = tools[name].invoke({"type": "tool_call", "id": "c1", "name": name, "args": args})
        assert isinstance(message, ToolMessage)
        assert isinstance(message.content, str)
        try:
            return json.loads(message.content)
        except ValueError:
            return message.content

    found = tool_call("find_leads", {"audience": "Sales leaders", "titles": ["Head of Sales"]})
    lead = found["leads"][0]
    email = tool_call("enrich", {"full_name": lead["full_name"], "company_domain": lead["company_domain"]})
    assert tool_call("verify_email", {"email": email["work_email"]})["keep"] is True
    draft = tool_call(
        "create_campaign",
        {"name": "Q4", "leads": [{"email": email["work_email"]}], "steps": [{"body": "Hi {{first_name}}"}]},
    )
    preview = tool_call("launch_campaign", {"campaign_id": draft["campaign_id"]})
    assert preview["approval_url"] == APPROVAL_URL
    assert api.paths() == [
        "/v1/find_leads",
        "/v1/enrich",
        "/v1/verify_email",
        "/v1/campaigns",
        "/v1/launch_campaign",
    ]


def test_invalid_arguments_come_back_as_text(tools: dict[str, BaseTool], api: MockLayrcake) -> None:
    output = tools["check_companies"].invoke({"question": "short", "company_domains": []})
    assert output.startswith("Refused: invalid arguments:")
    assert "question" in output
    assert api.requests == []


def test_launch_schema_has_no_confirmation_token(tools: dict[str, BaseTool]) -> None:
    schema = tools["launch_campaign"].tool_call_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    assert list(schema.model_json_schema()["properties"]) == ["campaign_id"]


def test_toolkit_dry_run_reaches_the_tools(client: Layrcake, api: MockLayrcake) -> None:
    tools = LayrcakeToolkit(client=client, dry_run=True).get_tools()
    output = json.loads(tools[2].invoke({"email": "ada@acme.com"}))
    assert output["dry_run"] is True
    assert api.requests[0][1]["dry_run"] is True
