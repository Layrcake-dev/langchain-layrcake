# langchain-layrcake

LangChain tools for [Layrcake](https://layrcake.dev), go-to-market infrastructure for AI agents. Your agent can find leads, find and verify work emails, check and research companies, and draft email campaigns that a person approves before anything sends.

The tools wrap the official [`layrcake`](https://pypi.org/project/layrcake/) Python SDK. They work with any LangChain or LangGraph agent that calls tools.

## Install

```sh
pip install -U langchain-layrcake
```

Get an API key at [layrcake.dev/connect](https://layrcake.dev/connect) and set `LAYRCAKE_API_KEY`. New workspaces start with 300 free credits.

## Quickstart

This example also needs LangChain and a model provider: `pip install -U "langchain[anthropic]"`.

```python
from langchain.agents import create_agent
from langchain_layrcake import LayrcakeToolkit

toolkit = LayrcakeToolkit(budget_credits=50)  # reads LAYRCAKE_API_KEY
agent = create_agent(model="anthropic:claude-sonnet-5-5", tools=toolkit.get_tools())

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find 5 heads of sales at UK software companies, verify their work emails, "
                    "draft a two-step campaign and ask for approval."
                ),
            }
        ]
    }
)
print(result["messages"][-1].content)
```

The agent stops at an approval link. Nothing is sent until a person approves it in the Layrcake console.

Try it free first: `LayrcakeToolkit(dry_run=True)` makes every call return its cost instead of data and charges nothing.

### A test run

On 20 October 2026 we ran this quickstart with the model `openrouter:anthropic/claude-sonnet-5.5` (through `langchain-openrouter`), `budget_credits=25` and this prompt: "Find 3 heads of sales at software companies in London with 20 to 200 staff, verify their work emails, and draft a short campaign about cutting time spent on lead research."

The agent called `get_me`, `find_leads`, `enrich` three times and `verify_email` twice. One person had no work email, so it called `find_leads`, `enrich` and `verify_email` once more for a replacement, then `create_campaign` and `launch_campaign`. It found 6 people, kept 3 verified emails (1 catch-all) and spent 24.75 of its 25 credits. It stopped at the approval link, and nothing was sent. The run took 67 seconds. The draft emails were signed "[Your name]", so edit the sign-off before you approve.

## Tools

| Tool | What it does | Cost |
| --- | --- | --- |
| `find_leads` | Finds people who match an audience: names, titles, companies. No emails. | 0.25 credits per person found, at least 3 when anyone is found. Finding nobody is free. |
| `enrich` | Finds one person's work email from their name and company domain. | 5 credits when found. Free when not. |
| `verify_email` | Checks that an address can receive mail. | 1.25 credits per check. |
| `check_companies` | Answers one yes or no question per company from its own website, with the quote. | 3.5 credits per company answered. Unclear answers are free. |
| `research_company` | Researches one company from public sources and returns dated facts with links. | 9 credits per completed research. |
| `create_campaign` | Saves a draft email campaign for verified people. Drafts never send. | Free. |
| `launch_campaign` | Asks a person to approve the draft. Sends nothing. Returns a preview and an approval link. | Free. |
| `get_me` | Shows the workspace, the credits available and the run budget. | Free. |

1 credit is about $0.01. New workspaces start with 300 free credits. Current prices are at [docs.layrcake.dev/pricing.html](https://docs.layrcake.dev/pricing.html).


Use the whole toolkit, or single tools. Tools built from the same `LayrcakeSession` share one run:

```python
from langchain_layrcake import LayrcakeFindLeads, LayrcakeSession, LayrcakeVerifyEmail

session = LayrcakeSession(budget_credits=20, dry_run=True)
tools = [LayrcakeFindLeads(session=session), LayrcakeVerifyEmail(session=session)]

print(tools[1].invoke({"email": "ada@example.com"}))
# {"dry_run": true, "would_cost_up_to": "1.25", "would_exceed": [], "warnings": []}
```

## How approval works

Nothing reaches a real person until a person on your side says yes.

1. `create_campaign` saves a draft. A draft never sends.
2. `launch_campaign` asks for approval. It returns `status: "needs_confirmation"`, a preview of what would send and an `approval_url`.
3. A signed-in person opens the link and approves in the Layrcake console. The Layrcake API enforces this step.
4. Only then does the campaign start sending.

The `launch_campaign` tool has no `confirmation_token` argument, and the confirmation token is removed from what the model reads, so an agent cannot approve its own send. After `launch_campaign` asks for approval, the other paid tools in the same run refuse to run, and the agent should stop and report the link.

## Run rules

The tools of one run share a `LayrcakeSession` from `layrcake.agent_tools`, the one implementation of these rules in the Layrcake Python SDK. It applies the rules from [layrcake.dev/agents.md](https://layrcake.dev/agents.md) in code:

- **Budget.** Every paid call sends what is left of the run budget as `budget_credits`, so the API refuses a call that could cost more. Once the budget is spent, paid tools refuse to run. The default budget is 50 credits.
- **Dry run.** With `dry_run=True`, every call returns its plan instead: the most it could cost and any limit it would trip. Nothing is looked up or charged.
- **Idempotency.** Every call sends an idempotency key derived from the run and its arguments. A repeated call returns the first answer instead of charging again. Pass the same `run_id` to resume a run safely.
- **Verified only.** `create_campaign` refuses an address that did not pass `verify_email` in this run (`valid` or `catch_all_valid`). Turn this off with `require_verified=False` if you verify elsewhere.
- **Errors as text.** API errors and refusals come back to the model as text that starts with `Refused:` and says why, such as a spent budget or missing credits, so it can fix the call or stop.

Every call goes through the official [`layrcake`](https://pypi.org/project/layrcake/) Python SDK, which handles authentication, retries and typed errors.


## Configuration

| Argument | Default | Meaning |
| --- | --- | --- |
| `budget_credits` | `50` | The most this run may spend, in credits. |
| `dry_run` | `False` | Return each call's plan instead of running it. Free. |
| `api_key` | `LAYRCAKE_API_KEY` | Your Layrcake API key. |
| `client` | `None` | An existing `layrcake.Layrcake` client, for custom timeouts or proxies. |
| `run_id` | random | Prefix for idempotency keys. Reuse it to resume a run. |
| `require_verified` | `True` | Refuse campaign leads not verified in this run. |

Use one toolkit per agent run. Before a live `launch_campaign`, connect sending mailboxes in the [Layrcake console](https://app.layrcake.dev); without them the call returns an error that says what is missing.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest tests/unit_tests --disable-socket --allow-unix-socket
LAYRCAKE_API_KEY=... uv run pytest -m live tests/integration_tests   # dry runs only, charges nothing
```

Unit tests run against a mocked API and include LangChain's standard tool tests from `langchain-tests`. The run rules are tested in the SDK.

Until `layrcake` 0.2.0 is on PyPI, `[tool.uv.sources]` in `pyproject.toml` installs the SDK from a local checkout of [layrcake-python](https://github.com/Layrcake-dev/layrcake-python) at `../../layrcake-python`. Delete that table and run `uv lock` before publishing.

## Links

- Layrcake docs: [docs.layrcake.dev](https://docs.layrcake.dev)
- For agents: [layrcake.dev/agents.md](https://layrcake.dev/agents.md)
- MCP server: `https://mcp.layrcake.dev/mcp`
- Python SDK: [github.com/Layrcake-dev/layrcake-python](https://github.com/Layrcake-dev/layrcake-python)

## License

MIT
