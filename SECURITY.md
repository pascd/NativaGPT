# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in NativaGPT, please **do not** open a public GitHub issue. Instead, report it privately to `<CHANGE_ME: maintainer security contact email>`, using the structured template at [`docs/reporting_template.md`](docs/reporting_template.md).

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce it (a minimal example is very helpful).
- Any suggested fix or mitigation, if you have one.

We'll acknowledge your report and aim to provide a status update within a reasonable timeframe.

## Handling API keys and credentials

NativaGPT talks to an LLM over a configurable OpenAI-compatible endpoint (see `config/config_default.json`'s `llm_config`). API keys are read from an environment variable (default name `LLM_API_KEY`, see `.env.example`) and are **never** meant to be committed to the repository.

If you accidentally commit a real API key or other credential:

1. Revoke/rotate the key with your provider immediately.
2. Remove it from git history (not just the latest commit) before pushing.
3. Consider it compromised even after removal from history.

`.env` is already covered by `.gitignore` - keep it that way.

## Supported Versions

This project does not yet have a formal release/support cycle; security fixes are applied to the `master` branch.
