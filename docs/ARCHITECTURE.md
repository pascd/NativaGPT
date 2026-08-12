# Architecture Overview

## Module map

```
NativaGPT/                          top-level Python package (import root)
├── lib/                            core library code
│   ├── coloring_logger.py          ColoredLogger / get_logger - shared console logger
│   ├── config_manager.py           ConfigManager - loads config_default.json + tool database
│   ├── command_execution.py        CommandExecution - subprocess/ROS command execution engine
│   ├── rag_similarity_check.py     RAGSimilarityCheck - embedding-based retrieval over the tool DB
│   ├── handlers/
│   │   ├── llm_prompt_handler.py   LLMPromptHandler - the OpenAI-compatible LLM client
│   │   ├── llm_response_handler.py LLMResponseHandler - parses LLM replies into text + JSON commands
│   │   ├── json_response_handler.py JsonResponseHandler - matches/normalizes/executes tool JSON
│   │   ├── topic_reader_handler.py TopicReaderHandler - reads ROS/MQTT/file "topics" for context
│   │   └── text_translator.py      standalone HF translation script (manual use only)
│   └── mcp/
│       ├── mcp_client.py           MCPClient - multi-server MCP client + tool-call orchestration
│       ├── mcp_server.py           example weather MCP server
│       ├── mcp_server_generic.py   builds a FastMCP server dynamically from a functions JSON file
│       └── mcp_server_ros.py       ROS-specific MCP server (topic capture, CLI bridge)
├── scripts/                        runnable entry points
│   ├── nativa.py                   main interactive text app - NativaGPT class
│   ├── nativa_mcp_wrapper.py       NativaMCPWrapper - sync wrapper w/ conversation memory
│   ├── nativa_restAPI.py           Flask REST API exposing chat/history/tools/status
│   └── start_nativa.py             orchestrator: launches dependent services, then nativa.py
└── bash/                           optional launcher scripts for local LLM backends

config/
├── config_default.json             the single runtime config file (see docs/CONFIGURATION.md)
└── functions/*.json                tool/command definitions exposed to the LLM
```

## Request lifecycle

```
start_nativa.py
      │  (or nativa_restAPI.py's /chat endpoint)
      ▼
nativa.py (NativaGPT)  ──or──  nativa_mcp_wrapper.py (NativaMCPWrapper)
      │
      │  builds a prompt (user input + conversation history + tool list + RAG context)
      ▼
LLMPromptHandler.send_to_llm(prompt, images=None, system_instruction=None)
      │
      │  POST {base_url}/chat/completions  (OpenAI-compatible; streaming or not)
      ▼
  LLM backend (Ollama / LM Studio / KoboldCpp / OpenAI / ...)
      │
      │  reply text, possibly containing a JSON "command" block
      ▼
LLMResponseHandler / JsonResponseHandler
      │
      │  if a tool/command was requested:
      ▼
CommandExecution (shell/ROS)  ──or──  MCPClient._execute_mcp_tool (MCP tool call)
      │
      │  result fed back via LLMPromptHandler.send_output_to_llm(...)
      ▼
final natural-language response returned to the caller
```

## Key design points

- **The LLM client is provider-agnostic.** `LLMPromptHandler` only assumes an OpenAI-compatible Chat Completions endpoint; swapping backends is a config change (`llm_config.base_url`/`model`), not a code change. See `docs/CONFIGURATION.md`.
- **`send_to_llm()` / `send_output_to_llm()` are a stable internal contract.** `NativaMCPWrapper`, `nativa.py`, and `MCPClient` all depend on their exact signature and return shape (`text_content`/`json_strings`/`success`). Changes to `LLMPromptHandler` must preserve this contract or update every call site.
- **Tool execution has two paths**: local shell/ROS commands go through `CommandExecution`; MCP-server-provided tools go through `MCPClient`/`NativaMCPWrapper`'s MCP session management. Both ultimately get summarized back through the LLM.
- **RAG is retrieval-only, and separate from chat.** `RAGSimilarityCheck` uses local Ollama embeddings (`nativa_gpt.embedding_model`) purely to rank which tool definitions are relevant to a query; it never calls the chat LLM.
