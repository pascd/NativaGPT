"""Small interactive smoke-test script for NativaMCPWrapper.

Not a pytest test suite (see tests/ at the repo root for that) - this is a
manual REPL-style script for quickly checking that the wrapper can be
initialized and can answer a prompt end-to-end against whatever LLM backend
is configured in config/config_default.json.
"""

from pathlib import Path

from NativaGPT.scripts.nativa_mcp_wrapper import NativaMCPWrapper

# Resolve the default config path relative to this file instead of a
# hardcoded, developer-specific absolute path.
DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "config" / "config_default.json"
)


def main():
    """Start an interactive prompt loop against a NativaMCPWrapper instance."""
    wrapper = NativaMCPWrapper(config_path=DEFAULT_CONFIG_PATH)

    while True:
        result = wrapper.ask(input("Prompt:"))

        print(result["tools_called"])
        print(result["response"])


if __name__ == "__main__":
    main()
