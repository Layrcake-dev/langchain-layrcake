"""A Layrcake client whose HTTP answers come from a table instead of the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any, Optional

import httpx
import pytest
from layrcake import Layrcake

Answer = Callable[[dict[str, Any]], Any]
CAMPAIGN_ID = "4f1c2b8e-0b7a-4c1e-9a55-2f6f1d2c3b4a"
OPERATION_ID = "9b0e6a52-2d4f-4f0a-8f3e-0c1d2e3f4a5b"
APPROVAL_URL = "https://app.layrcake.dev/approvals/a1"


def meta(credits: str) -> dict[str, str]:
    return {"request_id": "req_1", "credits_charged": credits, "credits_available": "250"}


LEAD = {
    "fingerprint": "f1",
    "full_name": "Ada Lovelace",
    "title": "Head of Sales",
    "company_name": "Acme",
    "company_domain": "acme.com",
    "work_email": None,
    "linkedin_url": None,
    "location": "London",
}


def _plan(verb: str, estimate: str) -> dict[str, Any]:
    return {
        "dry_run": True,
        "advisory": True,
        "evaluated_at": "2026-10-20T00:00:00Z",
        "plan": {"verb": verb, "credits_estimate": estimate, "would_exceed": [], "warnings": []},
    }


def _answer(verb: str, estimate: str, real: Answer) -> Answer:
    def answer(body: dict[str, Any]) -> Any:
        if body.get("dry_run"):
            return {"result": _plan(verb, estimate), "layrcake": meta("0")}
        return real(body)

    return answer


ANSWERS: dict[str, Answer] = {
    "/v1/find_leads": _answer(
        "find_leads", "3", lambda body: {"result": {"leads": [LEAD], "total_found": 1}, "layrcake": meta("3")}
    ),
    "/v1/enrich": _answer(
        "enrich",
        "5",
        lambda body: {"result": {"contact": {"work_email": "ada@acme.com"}}, "layrcake": meta("5.000000")},
    ),
    "/v1/verify_email": _answer(
        "verify_email",
        "1.25",
        lambda body: {
            "result": {"email": body["email"], "status": "valid" if body["email"] == "ada@acme.com" else "invalid"},
            "layrcake": meta("1.250000"),
        },
    ),
    "/v1/check_companies": _answer(
        "check_companies",
        "3.5",
        lambda body: {
            "result": {
                "question": body["question"],
                "answered": 1,
                "companies": [
                    {
                        "company_domain": "acme.com",
                        "status": "checked",
                        "answer": "yes",
                        "excerpt": "Book a demo inside the app",
                        "source_url": "https://acme.com/product",
                        "pages_read": ["https://acme.com/product"],
                        "charged": True,
                    }
                ],
            },
            "layrcake": meta("3.5"),
        },
    ),
    "/v1/research_company": _answer(
        "research_company",
        "9",
        lambda body: {
            "result": {"operation_id": OPERATION_ID, "state": "queued", "cached": False, "credits_reserved": "9"},
            "layrcake": meta("0"),
        },
    ),
    f"/v1/operations/{OPERATION_ID}": lambda body: {
        "operation_id": OPERATION_ID,
        "kind": "research_company",
        "state": "succeeded",
        "progress": {},
        "result": {"facts": [{"fact": "Raised a Series B", "source_url": "https://acme.com/news"}]},
        "error": None,
        "created_at": "2026-10-20T00:00:00Z",
        "completed_at": "2026-10-20T00:00:05Z",
    },
    "/v1/campaigns": _answer(
        "create_campaign",
        "0",
        lambda body: {
            "result": {
                "campaign_id": CAMPAIGN_ID,
                "name": body["name"],
                "state": "draft",
                "leads_accepted": len(body["leads"]),
                "leads_suppressed": 0,
                "created_at": "2026-10-20T00:00:00Z",
            }
        },
    ),
    "/v1/launch_campaign": _answer(
        "launch_campaign",
        "0",
        lambda body: {
            "result": {
                "status": "needs_confirmation",
                "campaign_id": body["campaign_id"],
                "confirmation_token": "tok_secret",
                "approval_url": APPROVAL_URL,
                "approval": {"confirmation_token": "tok_secret", "execution": "server"},
                "leads_active": 1,
            }
        },
    ),
    "/v1/me": lambda body: {
        "key": {"id": "k1"},
        "workspace": {"id": "w1", "name": "Test workspace"},
        "guard": {},
        "credits": {"available": "250", "reserved": "0"},
    },
}


class MockLayrcake:
    """Records every request and answers from ANSWERS."""

    def __init__(self, answers: dict[str, Answer]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, dict[str, Any], Optional[str]]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append((request.url.path, body, request.headers.get("Idempotency-Key")))
        answer = self.answers.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": request.url.path}})
        result = answer(body)
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, json=result)

    def paths(self) -> list[str]:
        return [path for path, _, _ in self.requests]


@pytest.fixture
def api() -> MockLayrcake:
    return MockLayrcake(dict(ANSWERS))


@pytest.fixture
def client(api: MockLayrcake) -> Iterator[Layrcake]:
    http = httpx.Client(transport=httpx.MockTransport(api.handle))
    with Layrcake(api_key="lc_test", http_client=http, max_retries=0) as layrcake:
        yield layrcake
    http.close()
