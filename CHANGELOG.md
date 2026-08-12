# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Dates are omitted for historical entries reconstructed from commit history; entries are grouped by theme rather than by exact date.

## [Unreleased]

Open-source release preparation.

### Removed (this change)
- The speech-to-text (STT) and text-to-speech (TTS) subsystems have been removed entirely: `NativaGPT/lib/speech_to_text/`, `NativaGPT/lib/text_to_speech/`, the `launch_stt_restAPI.sh`/`launch_tts_xtts.sh`/`launch_all_api.sh` bash launchers, the `stt_config`/`tts_config` config blocks, voice mode in `nativa.py` (wake-word detection, active listening, the STT/TTS lazy handlers, the `/voice` meta command), and the STT/TTS-only dependencies (`sounddevice`, `pydub`, `soundfile`, `numpy`, `scipy`, `Flask-RESTful`). NativaGPT is now a text/API-only assistant; voice I/O may return as a separate, pluggable integration in the future.

### Added
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, this `CHANGELOG.md`.
- `docs/CONFIGURATION.md` and `docs/ARCHITECTURE.md`.
- `.env.example` documenting the `LLM_API_KEY` environment variable.
- Lightweight smoke tests under `tests/` (package import, config loading, LLM request building) and a GitHub Actions CI workflow.
- Google-style (`Args`/`Returns`/`Raises`) docstrings across the codebase, replacing ad-hoc "vX.Y - feature list" class docstrings.
- `${REPO_ROOT}` placeholder support in `ConfigManager`, so config files no longer need contributor-specific absolute paths.

### Changed
- **`LLMPromptHandler` rewritten to speak a generic, OpenAI-compatible Chat Completions API** (`base_url` / `model` / `api_key_env` / `stream` config) instead of a private, proprietary NDJSON streaming endpoint. Works unmodified against OpenAI, Ollama, LM Studio, and KoboldCpp. Supports both SSE streaming and non-streaming responses, and OpenAI-style multimodal (`image_url`) content blocks for vision prompts.
- `config/config_default.json`'s `llm_config` block replaced accordingly; developer-specific absolute paths elsewhere in the file replaced with `${REPO_ROOT}`-relative or explicit `<CHANGE_ME>` placeholders.
- `NativaGPT/bash/launch_kobold_restAPI.sh`, `launch_llm_lms.sh`, `launch_tts_xtts.sh` now read their paths from overridable environment variables instead of hardcoding one contributor's home directory.
- `README.md` rewritten: fixed the non-existent `requirements.txt` reference, fixed a stale installation step, added Configuration/Contributing sections.

### Removed
- All remaining textual references to "WebGPTHandler" (a browser-automation ChatGPT-web-UI driver that was never vendored in this repo, only referenced by long-deleted test files): a stale README install step, two dead config keys (`use_webgpthandler`, `webgpthandler_platform`), and a stale comment in `llm_response_handler.py`.
- The conflicting CC BY-NC-ND 4.0 license footer from `README.md` - this project is, and has always been, MIT licensed (see `LICENSE`).

## [0.1.0]

Initial development history (squashed/summarized from git log).

### Added
- NativaGPT REST API (`NativaGPT/scripts/nativa_restAPI.py`) exposing chat/history/tools/status endpoints.
- `NativaMCPWrapper`: a simplified, synchronous wrapper around the MCP client with conversation memory management and usage examples.
- MCP server launchers and a generic server-generation script for native, ROS, and Turtlesim function sets.

### Changed
- `LLMPromptHandler` gained system-prompt override support and optimized (parallelized) image handling.
- `MCPClient` initialization integrated `LLMPromptHandler` directly for performance.
- `ColoredLogger` gained support for custom output streams (e.g. routing errors to stderr).
- RAG similarity check gained JSON-to-TOON conversion and logging.
- Several rounds of MCP server, `NativaMCPWrapper`, and general code-structure refactors for readability, maintainability, and performance.

### Removed
- Obsolete test files (including the last references to the external `WebGPTHandler` package) and unused topic configuration from default settings.
- Auto-generated launcher scripts for native/ROS/Turtlesim functions (superseded by the generic server-generation script).
