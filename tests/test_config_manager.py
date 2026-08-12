"""Smoke tests for ConfigManager and the ${REPO_ROOT} placeholder mechanism."""

from pathlib import Path

from NativaGPT.lib.config_manager import ConfigManager, REPO_ROOT

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config" / "config_default.json")


def test_config_loads():
    """The shipped default config file must load without error."""
    manager = ConfigManager(CONFIG_PATH)
    config = manager.get()
    assert isinstance(config, dict)
    assert "llm_config" in config
    assert "nativa_gpt" in config


def test_llm_config_has_generic_openai_compatible_shape():
    """llm_config must expose the generic client fields, not the old proprietary ones."""
    manager = ConfigManager(CONFIG_PATH)
    llm_config = manager.get()["llm_config"]

    for field in ("base_url", "model", "api_key_env", "stream"):
        assert field in llm_config, f"llm_config is missing '{field}'"

    assert "endpoint" not in llm_config
    assert "channel_id" not in llm_config


def test_repo_root_placeholder_is_resolved():
    """No literal "${REPO_ROOT}" token should survive into the loaded config."""
    manager = ConfigManager(CONFIG_PATH)
    config = manager.get()

    database_folder = config["nativa_gpt"]["database_folder"]
    assert "${REPO_ROOT}" not in database_folder
    assert database_folder == str(REPO_ROOT / "config" / "functions")


def test_webgpthandler_keys_are_gone():
    """The dead WebGPTHandler config keys must not be present anymore."""
    manager = ConfigManager(CONFIG_PATH)
    nativa_gpt = manager.get()["nativa_gpt"]

    assert "use_webgpthandler" not in nativa_gpt
    assert "webgpthandler_platform" not in nativa_gpt


def test_stt_tts_keys_are_gone():
    """The speech-to-text/text-to-speech subsystems have been removed; their
    config keys must not be present anymore."""
    manager = ConfigManager(CONFIG_PATH)
    config = manager.get()
    nativa_gpt = config["nativa_gpt"]

    assert "use_tts" not in nativa_gpt
    assert "use_stt" not in nativa_gpt
    assert "audio_key" not in nativa_gpt
    assert "text_key" not in nativa_gpt
    assert "stt_config" not in config
    assert "tts_config" not in config
