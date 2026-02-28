"""Tests for configuration loading."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from tg_purge.config import Config, load_config


class TestConfig:
    """Test Config dataclass."""

    def test_default_session_path(self):
        config = Config()
        assert ".tg_purge" in config.session_path
        assert "session" in config.session_path

    def test_custom_session_path(self):
        config = Config(session_path="/custom/path")
        assert config.session_path == "/custom/path"

    def test_validate_credentials_missing_both(self):
        config = Config(api_id=None, api_hash=None)
        with pytest.raises(SystemExit):
            config.validate_credentials()

    def test_validate_credentials_missing_id(self):
        config = Config(api_id=None, api_hash="hash123")
        with pytest.raises(SystemExit):
            config.validate_credentials()

    def test_validate_credentials_missing_hash(self):
        config = Config(api_id="123", api_hash=None)
        with pytest.raises(SystemExit):
            config.validate_credentials()

    def test_validate_credentials_success(self):
        config = Config(api_id="123", api_hash="hash123")
        config.validate_credentials()  # Should not raise

    def test_resolve_channel_cli_arg(self):
        config = Config(default_channel="@default")
        result = config.resolve_channel("@cli_channel")
        assert result == "@cli_channel"

    def test_resolve_channel_config_default(self):
        config = Config(default_channel="@default")
        result = config.resolve_channel(None)
        assert result == "@default"

    def test_resolve_channel_missing(self):
        config = Config(default_channel=None)
        with pytest.raises(SystemExit):
            config.resolve_channel(None)


class TestLoadConfig:
    """Test config loading from env vars and TOML."""

    def test_env_vars(self):
        env = {
            "TG_PURGE_API_ID": "12345",
            "TG_PURGE_API_HASH": "abcdef",
            "TG_PURGE_SESSION": "/tmp/test_session",
            "TG_PURGE_CHANNEL": "@test_channel",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config(config_path="/nonexistent/path.toml")

        assert config.api_id == "12345"
        assert config.api_hash == "abcdef"
        assert config.session_path == "/tmp/test_session"
        assert config.default_channel == "@test_channel"

    def test_defaults_when_nothing_set(self):
        env_keys = ["TG_PURGE_API_ID", "TG_PURGE_API_HASH", "TG_PURGE_SESSION", "TG_PURGE_CHANNEL"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, clean_env, clear=True):
            config = load_config(config_path="/nonexistent/path.toml")

        assert config.api_id is None
        assert config.api_hash is None
        assert ".tg_purge" in config.session_path
        assert config.default_channel is None
        assert config.delay == 1.5
        assert config.threshold == 2

    def test_toml_config(self):
        toml_content = b"""
[telegram]
api_id = "99999"
api_hash = "toml_hash"

[purge]
default_channel = "@toml_channel"
delay = 2.5
threshold = 4
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            env_keys = ["TG_PURGE_API_ID", "TG_PURGE_API_HASH", "TG_PURGE_SESSION", "TG_PURGE_CHANNEL"]
            clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
            with patch.dict(os.environ, clean_env, clear=True):
                config = load_config(config_path=toml_path)

            assert config.api_id == "99999"
            assert config.api_hash == "toml_hash"
            assert config.default_channel == "@toml_channel"
            assert config.delay == 2.5
            assert config.threshold == 4
        finally:
            os.unlink(toml_path)

    def test_env_vars_override_toml(self):
        toml_content = b"""
[telegram]
api_id = "99999"
api_hash = "toml_hash"

[purge]
default_channel = "@toml_channel"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            env = {"TG_PURGE_API_ID": "env_id", "TG_PURGE_API_HASH": "env_hash"}
            with patch.dict(os.environ, env, clear=False):
                config = load_config(config_path=toml_path)

            assert config.api_id == "env_id"  # env wins
            assert config.api_hash == "env_hash"  # env wins
            assert config.default_channel == "@toml_channel"  # TOML still used
        finally:
            os.unlink(toml_path)

    def test_invalid_toml_warning(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            f.write("this is not valid toml {{{{")
            toml_path = f.name

        try:
            env_keys = ["TG_PURGE_API_ID", "TG_PURGE_API_HASH", "TG_PURGE_SESSION", "TG_PURGE_CHANNEL"]
            clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
            with patch.dict(os.environ, clean_env, clear=True):
                config = load_config(config_path=toml_path)
            assert config.api_id is None  # Falls back to defaults
            captured = capsys.readouterr()
            assert "Warning" in captured.err or "Failed" in captured.err
        finally:
            os.unlink(toml_path)

    def test_missing_toml_no_error(self):
        """Missing TOML file should not raise, just use defaults."""
        env_keys = ["TG_PURGE_API_ID", "TG_PURGE_API_HASH", "TG_PURGE_SESSION", "TG_PURGE_CHANNEL"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, clean_env, clear=True):
            config = load_config(config_path="/definitely/not/a/real/file.toml")
        assert config.api_id is None

    def test_default_channel_from_config_enables_commands(self):
        """When default_channel is set, resolve_channel succeeds without CLI arg."""
        toml_content = b"""
[purge]
default_channel = "@configured_channel"
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            toml_path = f.name

        try:
            env_keys = ["TG_PURGE_API_ID", "TG_PURGE_API_HASH", "TG_PURGE_SESSION", "TG_PURGE_CHANNEL"]
            clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
            with patch.dict(os.environ, clean_env, clear=True):
                config = load_config(config_path=toml_path)

            channel = config.resolve_channel(None)
            assert channel == "@configured_channel"
        finally:
            os.unlink(toml_path)
