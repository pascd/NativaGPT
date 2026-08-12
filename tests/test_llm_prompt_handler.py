"""Smoke tests for LLMPromptHandler's OpenAI-compatible request building.

These tests never make a real network call - HTTP is mocked via
unittest.mock so they run offline and in CI.
"""

from unittest.mock import MagicMock, patch

from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler


def make_handler(**llm_overrides):
    """Build an LLMPromptHandler from a minimal config dict for testing."""
    config = {"llm_config": {"base_url": "http://localhost:11434/v1", **llm_overrides}}
    return LLMPromptHandler(config=config)


def test_defaults_point_at_local_ollama():
    """With no llm_config at all, defaults must be safe/local, never a private endpoint."""
    handler = LLMPromptHandler(config={})
    assert handler.base_url == "http://localhost:11434/v1"
    assert handler.api_key == "" or isinstance(handler.api_key, str)


def test_build_messages_text_only():
    """A plain text prompt becomes a single user message with string content."""
    handler = make_handler()
    messages, model = handler._build_messages("hello there", None, None)

    assert model == handler.model
    assert messages[-1]["role"] == "user"
    assert isinstance(messages[-1]["content"], str)
    assert "hello there" in messages[-1]["content"]


def test_build_messages_with_system_override():
    """A system_instruction override becomes a system message, replacing setup_prompt."""
    handler = make_handler(model_config={"setup_prompt": "default system prompt"})
    messages, _ = handler._build_messages("describe this", None, "custom override")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "custom override"


def test_build_messages_with_image_uses_vision_model_and_content_blocks(tmp_path):
    """Attaching an image switches to the vision model and builds OpenAI image_url blocks."""
    handler = make_handler(vision_model="llava:latest")

    # A tiny valid-enough fake image file (content doesn't need to be a real
    # image for this test - only the encoding path is exercised).
    img_path = tmp_path / "fixture.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    messages, model = handler._build_messages("describe", [str(img_path)], None)

    assert model == "llava:latest"
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    image_blocks = [b for b in content if b["type"] == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/")


def test_send_to_llm_streaming_parses_sse(monkeypatch):
    """send_to_llm parses SSE delta chunks and returns the concatenated content."""
    handler = make_handler(stream=True)

    sse_lines = [
        'data: {"model": "deepseek-r1:latest", "choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"model": "deepseek-r1:latest", "choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = sse_lines
    mock_response.raise_for_status.return_value = None

    with patch.object(handler.session, "post", return_value=mock_response) as mock_post:
        result = handler.send_to_llm("hi")

    assert mock_post.called
    assert result["success"] is True
    assert result["text_content"] == "Hello world"


def test_send_to_llm_nonstreaming_parses_openai_shape(monkeypatch):
    """send_to_llm (non-streaming) reads choices[0].message.content."""
    handler = make_handler(stream=False)

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "model": "deepseek-r1:latest",
        "choices": [{"message": {"content": "a plain reply"}}],
    }

    with patch.object(handler.session, "post", return_value=mock_response) as mock_post:
        result = handler.send_to_llm("hi")

    assert mock_post.called
    assert result["success"] is True
    assert result["text_content"] == "a plain reply"


def test_headers_include_bearer_token_only_when_api_key_set(monkeypatch):
    """No Authorization header is sent when no API key is configured (e.g. local Ollama)."""
    handler = make_handler()
    handler.api_key = ""
    assert "Authorization" not in handler._headers()

    handler.api_key = "sk-test"
    assert handler._headers()["Authorization"] == "Bearer sk-test"
