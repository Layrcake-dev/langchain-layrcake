"""Framework-neutral Layrcake tools: the run rules, the argument models and one function per verb.

This file is the same in langchain-layrcake, llama-index-tools-layrcake and
crewai-layrcake. Each package only adapts these functions to its framework, so
the rules below hold everywhere. Every API call goes through the official
`layrcake` SDK; there is no second HTTP client.

The rules (from https://layrcake.dev/agents.md):

- Every paid call carries the run budget that is left as `budget_credits`, and
  no paid call starts once the budget is spent.
- `dry_run` reaches every call. A dry run returns the most a call could cost
  and charges nothing.
- Every call sends an idempotency key derived from the run and its arguments,
  so a repeated call returns the first answer instead of charging twice.
- Only addresses verified in this run go into a campaign (unless turned off).
- `launch_campaign` never takes a confirmation token. It returns a preview and
  an approval link, and a person approves in the Layrcake console. Nothing in
  this package can approve a send.
- Errors come back as readable text, so the model can fix the call or stop.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from layrcake import Layrcake, LayrcakeError, LayrcakeResponse, OperationTimeoutError, approval_required
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from layrcake.types import (
        CheckCompaniesResultData,
        CreateCampaignParamsLeadsItem,
        CreateCampaignParamsStepsItem,
        CreateCampaignResultData,
        EnrichParamsPerson,
        FindLeadsParamsCompanyHeadcount,
        FindLeadsResultData,
        GetMeResult,
        LaunchCampaignResultData,
        ResearchCompanyResultData,
        VerifyEmailResultData,
    )

CORE_VERSION = "1"
"""Bumped whenever this file changes, in all three packages at once."""

SENDABLE = frozenset({"valid", "catch_all_valid"})
"""verify_email answers that are safe to send to."""

DEFAULT_BUDGET_CREDITS = Decimal(50)
_MAX_CALL_BUDGET = {"research_company": Decimal(1000)}  # the API's ceiling per call; 2000 elsewhere
_CENT = Decimal("0.01")


class Refused(Exception):
    """A call this package will not make. The message is shown to the model."""


# Argument models. Field names and limits follow the SDK's request types
# (layrcake.types.*Params); tests check every field against them.


class Headcount(BaseModel):
    """A company size range."""

    min: Optional[int] = Field(default=None, ge=1, description="Fewest employees.")
    max: Optional[int] = Field(default=None, ge=1, description="Most employees.")


class FindLeadsArgs(BaseModel):
    """Arguments for find_leads."""

    audience: str = Field(min_length=3, description="Who you want, in plain words.")
    titles: Optional[list[str]] = Field(
        default=None, max_length=50, description='Job titles, for example ["Head of Sales", "VP Sales"].'
    )
    industries: Optional[list[str]] = Field(
        default=None, max_length=50, description='LinkedIn industry names, for example ["Software Development"].'
    )
    locations: Optional[list[str]] = Field(
        default=None, max_length=50, description='Places, for example ["London, United Kingdom"].'
    )
    company_domains: Optional[list[str]] = Field(
        default=None, max_length=200, description="Only people at these company domains."
    )
    company_headcount: Optional[Headcount] = Field(default=None, description="Company size range.")
    limit: int = Field(default=10, ge=1, le=25, description="How many people, 1 to 25.")


class EnrichArgs(BaseModel):
    """Arguments for enrich."""

    full_name: str = Field(min_length=1, description="The person's full name.")
    company_domain: str = Field(min_length=1, description="Their company's website domain, for example acme.com.")


class VerifyEmailArgs(BaseModel):
    """Arguments for verify_email."""

    email: str = Field(min_length=3, max_length=320, description="The address to check.")


class CheckCompaniesArgs(BaseModel):
    """Arguments for check_companies."""

    question: str = Field(
        min_length=10,
        max_length=300,
        description="One yes or no question about each company's own offering, for example "
        '"Do they let their users book appointments inside their product?"',
    )
    company_domains: list[str] = Field(
        min_length=1, max_length=20, description='Company website domains, up to 20, for example ["acme.com"].'
    )


class ResearchCompanyArgs(BaseModel):
    """Arguments for research_company."""

    company_domain: str = Field(min_length=1, max_length=2048, description="Company website domain, e.g. acme.com.")
    audience: Optional[str] = Field(
        default=None, description="Optional goal, for example what you sell and to whom, to rank the facts."
    )


class Lead(BaseModel):
    """One campaign recipient."""

    email: str = Field(description="A work email verified in this run.")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_name: Optional[str] = None
    location: Optional[str] = Field(default=None, max_length=200)


class Step(BaseModel):
    """One email in the sequence."""

    subject: Optional[str] = Field(default=None, max_length=300, description="Subject line.")
    body: str = Field(min_length=1, description="Plain text. {{first_name}} is replaced per lead.")
    wait_days: int = Field(default=0, ge=0, le=60, description="Days after the previous step. Ignored on step one.")


class CreateCampaignArgs(BaseModel):
    """Arguments for create_campaign."""

    name: str = Field(min_length=1, max_length=120, description="Campaign name.")
    leads: list[Lead] = Field(min_length=1, max_length=5000, description="Recipients with verified emails.")
    steps: list[Step] = Field(min_length=1, max_length=3, description="One to three emails.")


class LaunchCampaignArgs(BaseModel):
    """Arguments for launch_campaign. There is deliberately no confirmation_token."""

    campaign_id: str = Field(min_length=1, description="The draft's campaign_id from create_campaign.")


class NoArgs(BaseModel):
    """get_me takes no arguments."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its name, what the model is told, its arguments and the session method that runs it."""

    name: str
    description: str
    args: type[BaseModel]
    method: str


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "find_leads",
        "Find people who match an audience. Returns names, titles and companies, without emails. "
        "Needs at least one of titles, industries, locations or company_domains. "
        "Costs 0.25 credits per person found, at least 3 credits when anyone is found; finding nobody is free.",
        FindLeadsArgs,
        "find_leads",
    ),
    ToolSpec(
        "enrich",
        "Find one person's work email from their full name and company domain. "
        "Costs 5 credits when an email is found; free when not.",
        EnrichArgs,
        "enrich",
    ),
    ToolSpec(
        "verify_email",
        "Check that an email address can receive mail. Keep valid and catch_all_valid; drop everything else. "
        "Costs 1.25 credits per check.",
        VerifyEmailArgs,
        "verify_email",
    ),
    ToolSpec(
        "check_companies",
        "Answer one yes or no question about each company in a list, from each company's own website, "
        "with the quote that shows it. Costs 3.5 credits per company answered; unclear answers and "
        "unreadable sites are free.",
        CheckCompaniesArgs,
        "check_companies",
    ),
    ToolSpec(
        "research_company",
        "Research one company from public sources and return dated facts with their source links. "
        "Costs 9 credits per completed research; finding nothing is free. Can take a few minutes.",
        ResearchCompanyArgs,
        "research_company",
    ),
    ToolSpec(
        "create_campaign",
        "Save a draft email campaign for people whose emails were verified in this run, with one to three "
        "emails. A draft never sends. Free.",
        CreateCampaignArgs,
        "create_campaign",
    ),
    ToolSpec(
        "launch_campaign",
        "Ask a person to approve a draft campaign. This sends nothing: it returns a preview and an approval "
        "link, and no email goes out until a person approves it in the Layrcake console. Free. "
        "Call it once, last, then stop and report the approval link.",
        LaunchCampaignArgs,
        "launch_campaign",
    ),
    ToolSpec(
        "get_me",
        "Show the workspace, the credits available and this run's budget. Free.",
        NoArgs,
        "get_me",
    ),
)


SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def _json(**values: Any) -> str:
    return json.dumps(values, default=str)


def _plan(result: Any) -> Optional[str]:
    """Return the dry-run plan as text: the most the call could cost. Nothing is charged."""
    if isinstance(result, dict) and result.get("dry_run") is True and "plan" in result:
        plan = result["plan"]
        return _json(
            dry_run=True,
            would_cost_up_to=plan.get("credits_estimate"),
            would_exceed=plan.get("would_exceed", []),
            warnings=plan.get("warnings", []),
        )
    return None


def _without_tokens(value: Any) -> Any:
    """Drop confirmation tokens from what the model reads. Only a person approves a send."""
    if isinstance(value, dict):
        return {k: _without_tokens(v) for k, v in value.items() if k != "confirmation_token"}
    if isinstance(value, list):
        return [_without_tokens(v) for v in value]
    return value


def invalid_arguments(error: ValidationError) -> str:
    """Describe bad tool arguments so the model can fix the call."""
    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in error.errors())
    return f"Refused: invalid arguments: {problems}"


def _describe(error: Exception) -> str:
    if isinstance(error, LayrcakeError):
        text = f"Refused: {error.code}: {error.message}" if error.code else f"Refused: {error.message}"
        if error.request_id:
            text += f" (request {error.request_id})"
        return text
    return f"Refused: {error}"


class LayrcakeSession:
    """The state one agent run shares across its Layrcake tools.

    Create one per run. It holds the run budget, the dry_run flag, the emails
    verified so far and, once launch_campaign has asked for approval, the
    approval link, after which no further paid call runs.

    Args:
        client: A `layrcake.Layrcake` client. Built from `api_key` or the
            LAYRCAKE_API_KEY environment variable when omitted.
        api_key: Used only when `client` is omitted.
        budget_credits: The most this run may spend, in credits. Default 50.
        dry_run: Return every call's plan instead of running it. Charges nothing.
        run_id: Prefix for idempotency keys. Reuse it to resume a run without paying twice.
        require_verified: Refuse campaign leads not verified in this run. Default True.
        research_timeout: Seconds research_company waits for its result. Default 300.

    """

    def __init__(
        self,
        client: Optional[Layrcake] = None,
        *,
        api_key: Optional[str] = None,
        budget_credits: Union[float, int, str, Decimal] = DEFAULT_BUDGET_CREDITS,
        dry_run: bool = False,
        run_id: Optional[str] = None,
        require_verified: bool = True,
        research_timeout: float = 300.0,
    ) -> None:
        self.client = client if client is not None else Layrcake(api_key=api_key)
        self.budget = Decimal(str(budget_credits))
        if self.budget <= 0:
            raise ValueError("budget_credits must be above 0")
        self.dry_run = dry_run
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.require_verified = require_verified
        self.research_timeout = research_timeout
        self.spent = Decimal(0)
        self.verified: set[str] = set()
        self.approval_url: Optional[str] = None
        self._lock = threading.Lock()  # one paid call at a time keeps the budget exact

    def left(self) -> Decimal:
        """Credits left in this run's budget."""
        return self.budget - self.spent

    def key(self, verb: str, args: Any) -> str:
        """Return the idempotency key: same call, same key, so a repeat never charges twice."""
        digest = hashlib.sha256(json.dumps([args, self.dry_run], sort_keys=True, default=str).encode()).hexdigest()
        return f"{self.run_id}-{verb}-{digest[:24]}"

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> str:
        """Validate raw tool arguments, run the tool and return what the model reads. Never raises."""
        spec = SPECS[name]
        try:
            args = spec.args.model_validate(dict(arguments or {}))
        except ValidationError as error:
            return invalid_arguments(error)
        return self.run(spec.method, args)

    def run(self, name: str, args: BaseModel) -> str:
        """Run one tool by name and return what the model reads. Never raises for API or rule errors."""
        method: Callable[[Any], str] = getattr(self, name)
        try:
            return method(args)
        except (LayrcakeError, Refused) as error:
            return _describe(error)

    def _paid(self, verb: str, args: dict[str, Any], call: Callable[..., LayrcakeResponse[Any]]) -> Any:
        """Refuse past the budget, cap the call at what is left, and count what it cost."""
        with self._lock:
            if self.approval_url:
                raise Refused("The campaign is waiting for approval. Stop and report the approval link.")
            if self.left() < _CENT:
                raise Refused(f"The run budget of {self.budget} credits is spent. Stop and report.")
            cap = min(self.left(), _MAX_CALL_BUDGET.get(verb, Decimal(2000)))
            response = call(
                **args,
                budget_credits=float(cap.quantize(_CENT, rounding=ROUND_DOWN)),
                dry_run=self.dry_run,
                idempotency_key=self.key(verb, args),
            )
            charged = response.credits_charged or Decimal(0)
            if verb == "research_company" and isinstance(response.result, dict):
                # Research is charged when it finishes. Count its hold now so the budget stays a ceiling.
                charged = max(charged, Decimal(str(response.result.get("credits_reserved") or 0)))
            self.spent += charged
            return response

    def _spend(self, response: LayrcakeResponse[Any]) -> dict[str, Any]:
        return {"credits_charged": response.credits_charged, "budget_left": self.left()}

    def find_leads(self, args: FindLeadsArgs) -> str:
        """Run find_leads."""
        if not (args.titles or args.industries or args.locations or args.company_domains):
            raise Refused("Add at least one of titles, industries, locations or company_domains.")
        payload: dict[str, Any] = {"audience": args.audience, "limit": args.limit}
        for name in ("titles", "industries", "locations", "company_domains"):
            if value := getattr(args, name):
                payload[name] = value
        if args.company_headcount:
            headcount: FindLeadsParamsCompanyHeadcount = cast(
                "FindLeadsParamsCompanyHeadcount", args.company_headcount.model_dump(exclude_none=True)
            )
            payload["company_headcount"] = headcount
        response = self._paid("find_leads", payload, self.client.find_leads)
        if plan := _plan(response.result):
            return plan
        data = cast("FindLeadsResultData", response.result)
        keep = ("full_name", "title", "company_name", "company_domain", "location", "linkedin_url")
        leads = [{k: lead.get(k) for k in keep} for lead in data["leads"]]
        return _json(leads=leads, **self._spend(response))

    def enrich(self, args: EnrichArgs) -> str:
        """Run enrich for a work email."""
        person: EnrichParamsPerson = {"full_name": args.full_name, "company_domain": args.company_domain}
        response = self._paid("enrich", {"person": person, "want": ["work_email"]}, self.client.enrich)
        if plan := _plan(response.result):
            return plan
        contact = response.result.get("contact") or {}
        return _json(work_email=contact.get("work_email"), **self._spend(response))

    def verify_email(self, args: VerifyEmailArgs) -> str:
        """Run verify_email and remember sendable addresses."""
        response = self._paid("verify_email", {"email": args.email}, self.client.verify_email)
        if plan := _plan(response.result):
            return plan
        data = cast("VerifyEmailResultData", response.result)
        keep = data["status"] in SENDABLE
        if keep:
            self.verified.add(args.email.strip().lower())
        return _json(email=args.email, status=data["status"], keep=keep, **self._spend(response))

    def check_companies(self, args: CheckCompaniesArgs) -> str:
        """Run check_companies."""
        payload = {"question": args.question, "company_domains": args.company_domains}
        response = self._paid("check_companies", payload, self.client.check_companies)
        if plan := _plan(response.result):
            return plan
        data = cast("CheckCompaniesResultData", response.result)
        keep = ("company_domain", "answer", "excerpt", "source_url")
        companies = [{k: row.get(k) for k in keep} for row in data["companies"]]
        return _json(question=data["question"], companies=companies, **self._spend(response))

    def research_company(self, args: ResearchCompanyArgs) -> str:
        """Start research_company and wait for its result."""
        payload: dict[str, Any] = {"company_domain": args.company_domain}
        if args.audience:
            payload["audience"] = args.audience
        response = self._paid("research_company", payload, self.client.research_company)
        if plan := _plan(response.result):
            return plan
        started = cast("ResearchCompanyResultData", response.result)
        try:
            operation = self.client.wait_for_operation(started["operation_id"], timeout=self.research_timeout)
        except OperationTimeoutError:
            return _json(
                operation_id=started["operation_id"],
                state="running",
                note=f"Still running after {self.research_timeout:g} seconds. Do not start it again; "
                "the result will appear in the Layrcake console.",
                credits_reserved=started["credits_reserved"],
                budget_left=self.left(),
            )
        return _json(
            operation_id=started["operation_id"],
            state=operation["state"],
            result=operation.get("result"),
            error=operation.get("error"),
            credits_reserved=started["credits_reserved"],
            budget_left=self.left(),
        )

    def create_campaign(self, args: CreateCampaignArgs) -> str:
        """Save a draft campaign. Drafts never send."""
        if self.require_verified and not self.dry_run:
            unverified = [lead.email for lead in args.leads if lead.email.strip().lower() not in self.verified]
            if unverified:
                raise Refused(f"Verify these first with verify_email, or drop them: {', '.join(unverified)}")
        leads = [cast("CreateCampaignParamsLeadsItem", lead.model_dump(exclude_none=True)) for lead in args.leads]
        steps = [cast("CreateCampaignParamsStepsItem", step.model_dump(exclude_none=True)) for step in args.steps]
        key = self.key("create_campaign", [args.name, leads, steps])
        response = self.client.create_campaign(
            name=args.name, leads=leads, steps=steps, dry_run=self.dry_run, idempotency_key=key
        )
        if plan := _plan(response.result):
            return plan
        draft = cast("CreateCampaignResultData", response.result)
        return _json(
            campaign_id=draft["campaign_id"],
            state=draft["state"],
            leads_accepted=draft["leads_accepted"],
            leads_suppressed=draft["leads_suppressed"],
        )

    def launch_campaign(self, args: LaunchCampaignArgs) -> str:
        """Ask a person to approve the draft. Sends nothing; there is no way to pass a confirmation token."""
        response = self.client.launch_campaign(
            campaign_id=args.campaign_id,
            dry_run=self.dry_run,
            idempotency_key=self.key("launch_campaign", args.campaign_id),
        )
        if plan := _plan(response.result):
            return plan
        data = cast("LaunchCampaignResultData", response.result)
        approval = approval_required(response)
        if approval is not None:
            self.approval_url = approval.approval_url or "(approval requested in the Layrcake console)"
        return _json(
            status=data["status"],
            campaign_id=data["campaign_id"],
            approval_url=approval.approval_url if approval else None,
            next_step="A person must approve this in the Layrcake console before anything sends. Stop here."
            if approval
            else None,
            preview=_without_tokens({k: v for k, v in data.items() if k not in ("status", "campaign_id")}),
        )

    def get_me(self, args: Optional[BaseModel] = None) -> str:
        """Show the workspace, its credits and this run's budget. Free."""
        me: GetMeResult = self.client.get_me().result
        credits = me.get("credits")
        return _json(
            workspace=me["workspace"].get("name"),
            credits_available=credits["available"] if credits else None,
            run_budget=self.budget,
            run_spent=self.spent,
            dry_run=self.dry_run,
        )
