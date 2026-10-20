"""The run rules every Layrcake tool shares. The same tests run in all three framework packages."""

from __future__ import annotations

import hashlib
import inspect
import json
import typing
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from layrcake import types as sdk_types

from langchain_layrcake import _core as core
from tests.unit_tests.conftest import APPROVAL_URL, CAMPAIGN_ID, OPERATION_ID, MockLayrcake

LeadArgs = core.Lead
STEPS = [core.Step(subject="Hi", body="Hi {{first_name}}")]


def call(session: core.LayrcakeSession, tool: str, /, **args: Any) -> Any:
    spec = next(s for s in core.TOOL_SPECS if s.name == tool)
    output = session.run(spec.method, spec.args(**args))
    try:
        return json.loads(output)
    except ValueError:
        return output  # a refusal, as the model reads it


def test_paid_calls_carry_the_budget_left_and_an_idempotency_key(
    session: core.LayrcakeSession, api: MockLayrcake
) -> None:
    found = call(session, "find_leads", audience="Sales leaders", titles=["Head of Sales"])
    assert found["leads"][0]["full_name"] == "Ada Lovelace"
    assert "work_email" not in found["leads"][0]
    path, body, key = api.requests[0]
    assert (path, body["budget_credits"], body["dry_run"], body["limit"]) == ("/v1/find_leads", 100.0, False, 10)
    assert key is not None
    assert key.startswith("run-test-find_leads-")

    call(session, "enrich", full_name="Ada Lovelace", company_domain="acme.com")
    assert api.requests[1][1]["want"] == ["work_email"]
    assert api.requests[1][1]["budget_credits"] == 97.0
    assert session.spent == Decimal(8)


def test_the_same_call_sends_the_same_key(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    call(session, "verify_email", email="ada@acme.com")
    call(session, "verify_email", email="ada@acme.com")
    assert api.requests[0][2] == api.requests[1][2]


def test_find_leads_needs_a_filter(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    output = call(session, "find_leads", audience="Anyone at all")
    assert output.startswith("Refused: Add at least one of titles")
    assert api.requests == []


def test_campaigns_only_take_verified_emails(session: core.LayrcakeSession) -> None:
    call(session, "verify_email", email="ada@acme.com")
    bob = call(session, "verify_email", email="bob@acme.com")
    assert bob["keep"] is False
    refused = call(session, "create_campaign", name="Q4", leads=[LeadArgs(email="bob@acme.com")], steps=STEPS)
    assert "bob@acme.com" in refused
    created = call(session, "create_campaign", name="Q4", leads=[LeadArgs(email="Ada@acme.com")], steps=STEPS)
    assert created["campaign_id"] == CAMPAIGN_ID
    assert session.spent == Decimal("2.5")


def test_require_verified_can_be_turned_off(client: Any) -> None:
    session = core.LayrcakeSession(client, require_verified=False)
    created = call(session, "create_campaign", name="Q4", leads=[LeadArgs(email="x@acme.com")], steps=STEPS)
    assert created["state"] == "draft"


def test_launch_stops_at_the_approval_link_and_hides_the_token(
    session: core.LayrcakeSession, api: MockLayrcake
) -> None:
    preview = call(session, "launch_campaign", campaign_id=CAMPAIGN_ID)
    assert preview["approval_url"] == APPROVAL_URL
    assert preview["status"] == "needs_confirmation"
    assert "tok_secret" not in json.dumps(preview)
    assert "confirmation_token" not in api.requests[0][1]
    assert session.approval_url == APPROVAL_URL
    assert "waiting for approval" in call(session, "verify_email", email="x@acme.com")


def test_launch_cannot_take_a_confirmation_token() -> None:
    assert "confirmation_token" not in core.LaunchCampaignArgs.model_fields
    assert list(inspect.signature(core.LayrcakeSession.launch_campaign).parameters) == ["self", "args"]


def test_paid_calls_stop_when_the_budget_is_spent(client: Any) -> None:
    session = core.LayrcakeSession(client, budget_credits=5)
    call(session, "enrich", full_name="Ada Lovelace", company_domain="acme.com")
    assert "budget of 5 credits is spent" in call(session, "enrich", full_name="Bob Smith", company_domain="acme.com")


def test_dry_run_reaches_every_call_and_charges_nothing(client: Any, api: MockLayrcake) -> None:
    session = core.LayrcakeSession(client, dry_run=True)
    plan = call(session, "find_leads", audience="Sales leaders", titles=["Head of Sales"])
    assert plan == {"dry_run": True, "would_cost_up_to": "3", "would_exceed": [], "warnings": []}
    call(session, "check_companies", question="Do they sell to hospitals?", company_domains=["acme.com"])
    call(session, "research_company", company_domain="acme.com")
    call(session, "create_campaign", name="Q4", leads=[LeadArgs(email="x@acme.com")], steps=STEPS)
    call(session, "launch_campaign", campaign_id=CAMPAIGN_ID)
    assert all(body["dry_run"] is True for _, body, _ in api.requests)
    assert not any(path.startswith("/v1/operations/") for path in api.paths())
    assert session.spent == 0
    assert session.approval_url is None


def test_check_companies_returns_answers_with_evidence(session: core.LayrcakeSession) -> None:
    result = call(session, "check_companies", question="Do they let users book demos?", company_domains=["acme.com"])
    assert result["companies"] == [
        {
            "company_domain": "acme.com",
            "answer": "yes",
            "excerpt": "Book a demo inside the app",
            "source_url": "https://acme.com/product",
        }
    ]
    assert result["credits_charged"] == "3.5"


def test_research_waits_for_the_operation_and_counts_its_hold(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    result = call(session, "research_company", company_domain="acme.com", audience="We sell payroll software")
    assert result["state"] == "succeeded"
    assert result["result"]["facts"][0]["fact"] == "Raised a Series B"
    assert api.requests[0][1]["audience"] == "We sell payroll software"
    assert session.spent == Decimal(9)


def test_research_that_outlasts_the_wait_reports_its_operation(client: Any, api: MockLayrcake) -> None:
    session = core.LayrcakeSession(client, research_timeout=0.01)
    still_running = dict(api.answers[f"/v1/operations/{OPERATION_ID}"]({}), state="running", result=None)
    api.answers[f"/v1/operations/{OPERATION_ID}"] = lambda body: still_running
    result = call(session, "research_company", company_domain="acme.com")
    assert (result["operation_id"], result["state"]) == (OPERATION_ID, "running")
    assert "Do not start it again" in result["note"]


def test_api_errors_come_back_as_text(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    api.answers["/v1/verify_email"] = lambda body: httpx.Response(
        402,
        json={"error": {"code": "insufficient_credits", "message": "Top up to continue."}},
        headers={"x-request-id": "req_9"},
    )
    output = call(session, "verify_email", email="ada@acme.com")
    assert output.startswith("Refused: insufficient_credits: Top up to continue.")


def test_get_me_is_free_and_shows_the_run(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    me = call(session, "get_me")
    assert me == {
        "workspace": "Test workspace",
        "credits_available": "250",
        "run_budget": "100",
        "run_spent": "0",
        "dry_run": False,
    }
    assert api.requests[0][2] is None


def test_budget_must_be_positive(client: Any) -> None:
    with pytest.raises(ValueError, match="budget_credits"):
        core.LayrcakeSession(client, budget_credits=0)


def _sdk_fields(typed_dict: Any) -> set[str]:
    return set(typing.get_type_hints(typed_dict))


@pytest.mark.parametrize(
    ("model", "sdk_type"),
    [
        (core.FindLeadsArgs, sdk_types.FindLeadsParams),
        (core.Headcount, sdk_types.FindLeadsParamsCompanyHeadcount),
        (core.VerifyEmailArgs, sdk_types.VerifyEmailParams),
        (core.CheckCompaniesArgs, sdk_types.CheckCompaniesParams),
        (core.ResearchCompanyArgs, sdk_types.ResearchCompanyParams),
        (core.CreateCampaignArgs, sdk_types.CreateCampaignParams),
        (core.Lead, sdk_types.CreateCampaignParamsLeadsItem),
        (core.Step, sdk_types.CreateCampaignParamsStepsItem),
        (core.LaunchCampaignArgs, sdk_types.LaunchCampaignParams),
    ],
)
def test_argument_models_follow_the_sdk_contract(model: Any, sdk_type: Any) -> None:
    assert set(model.model_fields) <= _sdk_fields(sdk_type)


def test_enrich_arguments_follow_the_sdk_contract() -> None:
    assert set(core.EnrichArgs.model_fields) <= _sdk_fields(sdk_types.EnrichParamsPerson)


def test_every_tool_says_what_it_costs() -> None:
    for spec in core.TOOL_SPECS:
        assert "credit" in spec.description.lower() or "free" in spec.description.lower(), spec.name
        assert "—" not in spec.description
    launch = next(s for s in core.TOOL_SPECS if s.name == "launch_campaign")
    assert "approve" in launch.description


def test_raw_arguments_are_validated_into_readable_text(session: core.LayrcakeSession, api: MockLayrcake) -> None:
    output = session.call("check_companies", {"question": "short", "company_domains": []})
    assert output.startswith("Refused: invalid arguments: question:")
    assert "company_domains" in output
    assert api.requests == []
    found = json.loads(session.call("find_leads", {"audience": "Sales leaders", "titles": ["Head of Sales"]}))
    assert found["leads"][0]["company_domain"] == "acme.com"


CORE_SHA256 = "87e29e7dcc761d358e1787006ee0baf47ed4fd61f7fc2d8e32db0b517c826268"


def test_core_is_the_shared_copy() -> None:
    """_core.py is the same file in langchain-layrcake, llama-index-tools-layrcake and crewai-layrcake."""
    digest = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    assert digest == CORE_SHA256, "_core.py changed: copy it to all three packages, bump CORE_VERSION, update this hash"
