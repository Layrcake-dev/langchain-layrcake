# Listing langchain-layrcake in the LangChain docs

Researched 2026-10-20 from the langchain-ai/docs repo (`src/oss/contributing/integrations-langchain.mdx`, `publish-langchain.mdx`, `.github/ISSUE_TEMPLATE/06-integration-submission.yml`).

## The path

1. LangChain does not accept new integrations as PRs to any `langchain-ai` repo. The package lives in our own repo (`Layrcake-dev/langchain-layrcake`) and is published to PyPI first.
2. The listing is **an issue, not a PR**: the "Integration listing" form at
   https://github.com/langchain-ai/docs/issues/new?template=06-integration-submission.yml
3. A maintainer applies the `integration-run` label. Automation then opens a PR that adds a row under `python.tools` in `scripts/data/integration_external_docs.yaml` and tags us for review. We review that PR; we do not open one by hand (automation closes manual listing PRs).
4. A hosted docs page (`src/oss/python/integrations/tools/<name>.mdx`, from `tools/TEMPLATE.mdx`) is only for packages with 50,000+ monthly downloads or ones maintainers mark as featured. Until then the row links to our `docs_url`.
5. Rules: English only; one factual description sentence (about 160 characters, no marketing); package must already be on PyPI (a required checkbox); AI-generated content must follow their "acceptable uses of LLMs" policy; do not tag maintainers repeatedly.

Prerequisites before filing: the GitHub repo is public, `langchain-layrcake` 0.1.0 is on PyPI, and the README renders (it is the `docs_url`).

## Draft issue (fill the form fields exactly)

**Title:** `[integration]: langchain-layrcake`

| Field | Value |
| --- | --- |
| Display or class name | `LayrcakeToolkit` |
| Language | Python |
| Component type | tools |
| PyPI package name | `langchain-layrcake` |
| npm package name | (blank) |
| Docs URL | `https://github.com/Layrcake-dev/langchain-layrcake#readme` |
| Source repository | `Layrcake-dev/langchain-layrcake` |
| Capability flags | (blank; they apply to chat models and vector stores) |

**Short provider description** (148 characters):

> Tools to find B2B leads, find and verify work emails, check and research companies, and draft email campaigns that a person approves before sending.

**Confirmations:** tick "The package is already published on PyPI and/or npm."

## Expected YAML row (what the automation should add)

```yaml
  tools:
  - name: LayrcakeToolkit
    pypi: langchain-layrcake
    docs_url: https://github.com/Layrcake-dev/langchain-layrcake#readme
```

Check the automated PR against this before approving it.

## Optional later: hosted guide

If downloads pass 50,000 a month or maintainers feature it, a hosted page follows `src/oss/python/integrations/tools/TEMPLATE.mdx`: Overview (details table with class, package, serializable, JS support, badges), Setup (credentials with `getpass`, install with pip and uv), Instantiation, Invocation (direct and as a ToolCall), Within an agent (`create_agent`), API reference. The README already has each piece; the dry-run examples are safe to run in their docs CI because they charge nothing.

## Who does what

- Founder or Layrcake-dev org account: make the repo public, publish to PyPI (trusted publishing from GitHub Actions is preferable to a token), file the issue.
- Claude: review the automated PR when it opens.
