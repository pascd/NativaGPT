# Contributing to NativaGPT

Thanks for considering a contribution! This document covers how to set up a development environment, the conventions the codebase follows, and how to get a change merged.

## Development setup

```bash
git clone https://github.com/pascd/NativaGPT.git
cd NativaGPT
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install pytest ruff black   # dev tools (also listed under [dependency-groups].dev in pyproject.toml)
cp .env.example .env
```

You'll also want a local LLM backend to test against - the quickest is Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
./NativaGPT/bash/launch_llm_ollama.sh
```

## Running checks

```bash
pytest tests/          # smoke tests: package import, config loading, LLM request building
ruff check .           # lint
black --check .        # formatting check
```

CI (`.github/workflows/ci.yml`) runs the same three commands on every push/PR.

## Manual end-to-end check

The automated tests are deliberately lightweight (no real network calls). To verify a change actually works end-to-end:

```bash
# Terminal 1
./NativaGPT/bash/launch_llm_ollama.sh

# Terminal 2
python3 NativaGPT/scripts/start_nativa.py
```

Send a text prompt and confirm you get a real, non-error response back.

## Coding conventions

- **Docstrings**: Google-style (`Args:` / `Returns:` / `Raises:`) on every public class, method, and function. A one-line module docstring at the top of every file.
- **Formatting/linting**: `black` (default settings) and `ruff` (see `[tool.ruff]` in `pyproject.toml`).
- **Config paths**: never hardcode an absolute path in `config/config_default.json` or a bash script. Use the `${REPO_ROOT}` placeholder (resolved by `ConfigManager`) for paths inside this repo, or a `${VAR:-default}` shell pattern for paths outside it (see the scripts under `NativaGPT/bash/`).
- **The `LLMPromptHandler` contract**: `send_to_llm(prompt, images=None, system_instruction=None)` and `send_output_to_llm(out)` are treated as a stable internal API by `NativaMCPWrapper`, `nativa.py`, and `MCPClient`. If you need to change their signature or return shape, update all call sites in the same change.

## Submitting a change

1. Create a branch off `master` (don't push directly to `master`).
2. Keep commits focused; write a clear commit message.
3. Make sure `pytest`, `ruff check`, and `black --check` all pass.
4. Open a pull request describing what changed and why.

## Reporting bugs / requesting features

Please open a GitHub issue with steps to reproduce (for bugs) or a clear use case (for feature requests). For security vulnerabilities, see [`SECURITY.md`](SECURITY.md) instead - please don't open a public issue.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you're expected to uphold it.
