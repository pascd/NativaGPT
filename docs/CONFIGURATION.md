# Configuration Reference

NativaGPT is configured through a single JSON file, `config/config_default.json`, loaded by `NativaGPT.lib.config_manager.ConfigManager`. You can point any entry-point script (`start_nativa.py`, `nativa_restAPI.py`, `NativaMCPWrapper(config_path=...)`) at a different config file if you want an alternate profile.

## The `${REPO_ROOT}` placeholder

Any string value in the config file may contain the literal token `${REPO_ROOT}`. `ConfigManager` replaces it with the absolute path to the repository root before the config is used, so paths that live inside this repo (MCP server scripts, the tool-function database) don't need to hardcode a contributor's home directory. For paths that point *outside* the repo, use an explicit `<CHANGE_ME: ...>` placeholder instead and edit it by hand.

## `nativa_gpt`

General application behavior.

| Field | Meaning |
|---|---|
| `database_folder` | Directory scanned (recursively) for `*.json` tool/function definitions exposed to the LLM. Defaults to `${REPO_ROOT}/config/functions`. |
| `embedding_model` | Ollama embedding model used by `RAGSimilarityCheck` for tool retrieval (unrelated to chat completion - always goes through the local `ollama` Python package, not `llm_config`). |
| `analysis_history_limit` | Cap on conversation history length. |

## `llm_config` - the LLM backend

`LLMPromptHandler` speaks a generic **OpenAI-compatible Chat Completions API** (`POST {base_url}/chat/completions`). This works, unmodified, against real OpenAI and against several local servers.

| Field | Default | Meaning |
|---|---|---|
| `base_url` | `http://localhost:11434/v1` | Root URL of the OpenAI-compatible API (no trailing slash). |
| `model` | `deepseek-r1:latest` | Model name for text-only requests. |
| `vision_model` | `llava:latest` | Model name used when a request includes images. |
| `api_key_env` | `LLM_API_KEY` | **Name** of the environment variable holding the API key (never the key itself - see `.env.example`). Leave the env var unset for backends that don't require auth. |
| `temperature` | `0.1` | Sampling temperature. |
| `max_tokens` | `2000` | Max tokens per completion. |
| `stream` | `true` | Use SSE streaming (`true`) or a single blocking response (`false`). |
| `timeout` | `60` | Request timeout, in seconds. |
| `model_config.setup_prompt` | - | System prompt sent with every request (unless overridden per-call). |

### Worked examples

**Local Ollama (default)** - run `./NativaGPT/bash/launch_llm_ollama.sh`, then use the config as shipped. No `.env` entry needed.

**OpenAI**:
```json
"llm_config": {
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "vision_model": "gpt-4o-mini",
  "api_key_env": "LLM_API_KEY"
}
```
and in `.env`: `LLM_API_KEY=sk-...`.

**LM Studio** (run `./NativaGPT/bash/launch_llm_lms.sh` first, serves on port 8000):
```json
"llm_config": {
  "base_url": "http://localhost:8000/v1",
  "model": "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF"
}
```

**KoboldCpp** (run `./NativaGPT/bash/launch_kobold_restAPI.sh` first):
```json
"llm_config": {
  "base_url": "http://localhost:5001/v1",
  "model": "koboldcpp"
}
```

## `mcp`

| Field | Meaning |
|---|---|
| `enabled` | Turn MCP tool integration on/off. |
| `mcp_servers.<name>.host` | Path to an MCP server script launched over stdio. Ships with `weather_mcp` (example server) and `ros1_mcp` (ROS1 bridge, only useful if ROS1 is installed on the host). |

Additional MCP servers can be registered at runtime via `NativaMCPWrapper.add_mcp_server()`/`add_function_json()` - see `NativaMCPWrapper.usage_examples()`.

## `logging`

| Field | Meaning |
|---|---|
| `level` | Standard logging level string. |
| `log_directory` | Defaults to `${REPO_ROOT}/logs` (git-ignored). |
| `log_rotation` | Enable log file rotation. |

## Local backend launcher scripts (`NativaGPT/bash/`)

These are optional convenience scripts for starting local LLM backends; none of them are invoked automatically by the Python code.

| Script | Starts | Overridable env vars |
|---|---|---|
| `launch_llm_ollama.sh` | Ollama server + `deepseek-r1:latest` | - |
| `launch_llm_lms.sh` | LM Studio CLI server | `LMS_APPIMAGE_DIR`, `LMS_MODEL_NAME` |
| `launch_kobold_restAPI.sh` | KoboldCpp server | `KOBOLD_REPO_PATH`, `KOBOLD_CONFIG` |
