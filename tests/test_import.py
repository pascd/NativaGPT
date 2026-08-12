"""Smoke test: the package must import cleanly with no leftover dead references.

``NativaGPT.lib.__init__`` eagerly imports most of the package's public
classes, so a single successful `import NativaGPT` here is a strong signal
that no module raises at import time (e.g. a stray reference to a removed
name like ``channel_id`` or ``_handle_ndjson_responses`` after a refactor).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_imports():
    """`import NativaGPT` should succeed without raising."""
    import NativaGPT  # noqa: F401


def test_llm_prompt_handler_is_openai_compatible_only():
    """LLMPromptHandler must expose the generic OpenAI-compatible client shape."""
    from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler

    assert "base_url" in LLMPromptHandler.__slots__
    assert "channel_id" not in LLMPromptHandler.__slots__
    assert "endpoint" not in LLMPromptHandler.__slots__


def _scan_repo_for_pattern(pattern):
    """Return repo-relative paths of code/config/script files matching ``pattern``.

    Scoped to `.py`/`.json`/`.sh` files (not Markdown docs, which may
    legitimately document removed features by name in CHANGELOG.md) and
    excludes venvs and this test suite itself (whose assertion strings
    necessarily mention the terms they're checking for the absence of).
    """
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in {".py", ".json", ".sh"}:
            continue
        if ".venv" in path.parts or "venv" in path.parts or "node_modules" in path.parts:
            continue
        if "tests" in path.relative_to(REPO_ROOT).parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    return offenders


def test_no_leftover_webgpthandler_or_proprietary_references():
    """No source/config file should mention WebGPTHandler, channel_id, or the old proprietary endpoint."""
    pattern = re.compile(r"webgpthandler|channel_id|api\.iaedu\.pt", re.IGNORECASE)
    offenders = _scan_repo_for_pattern(pattern)
    assert (
        not offenders
    ), f"Found leftover WebGPTHandler/proprietary-endpoint references in: {offenders}"


def test_no_leftover_stt_tts_references():
    """No source/config file should mention the removed STT/TTS subsystems."""
    pattern = re.compile(
        r"speech_to_text|text_to_speech|use_stt|use_tts|stt_config|tts_config", re.IGNORECASE
    )
    offenders = _scan_repo_for_pattern(pattern)
    assert not offenders, f"Found leftover STT/TTS references in: {offenders}"
