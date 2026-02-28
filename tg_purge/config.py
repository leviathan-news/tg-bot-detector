"""
Configuration loading for tg-purge.

Reads from environment variables and optionally from a TOML config file.
No .env file parsing — use your shell or a tool like direnv.

Environment variables:
    TG_PURGE_API_ID     — Telegram API ID (from my.telegram.org)
    TG_PURGE_API_HASH   — Telegram API hash (from my.telegram.org)
    TG_PURGE_SESSION    — Session file path (default: ~/.tg_purge/session)
    TG_PURGE_CHANNEL    — Default channel to analyze
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Runtime configuration for tg-purge."""
    api_id: Optional[str] = None
    api_hash: Optional[str] = None
    session_path: str = ""
    default_channel: Optional[str] = None
    delay: float = 1.5
    threshold: int = 2

    def __post_init__(self):
        if not self.session_path:
            self.session_path = str(Path.home() / ".tg_purge" / "session")

    def validate_credentials(self):
        """Raise SystemExit if API credentials are missing."""
        missing = []
        if not self.api_id:
            missing.append("TG_PURGE_API_ID")
        if not self.api_hash:
            missing.append("TG_PURGE_API_HASH")
        if missing:
            print(
                f"Error: Missing required credentials: {', '.join(missing)}\n"
                f"Set them as environment variables or in config.toml.\n"
                f"Get your credentials at https://my.telegram.org",
                file=sys.stderr,
            )
            sys.exit(1)

    def resolve_channel(self, cli_channel=None):
        """Return the channel to use, preferring CLI arg over config default.

        Args:
            cli_channel: Channel passed via --channel CLI argument.

        Returns:
            The resolved channel string.

        Raises:
            SystemExit: If no channel is available from any source.
        """
        channel = cli_channel or self.default_channel
        if not channel:
            print(
                "Error: --channel is required. "
                "Set default_channel in config.toml to avoid passing it each time.",
                file=sys.stderr,
            )
            sys.exit(1)
        return channel


def _load_toml(path):
    """Load a TOML file, returning a dict. Returns empty dict on failure."""
    path = Path(path)
    if not path.exists():
        return {}

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomli as tomllib
            except ImportError:
                print(
                    "Warning: Python <3.11 requires 'tomli' for TOML config support. "
                    "Install with: pip install tomli",
                    file=sys.stderr,
                )
                return {}

        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse {path}: {e}", file=sys.stderr)
        return {}


def load_config(config_path=None):
    """Load configuration from environment variables and optional TOML file.

    Priority (highest to lowest):
        1. Environment variables
        2. TOML config file
        3. Built-in defaults

    Args:
        config_path: Path to TOML config file. If None, looks for
                     config.toml in the current directory.

    Returns:
        A Config instance.
    """
    # Load TOML if available
    toml_path = config_path or "config.toml"
    toml_data = _load_toml(toml_path)
    tg_section = toml_data.get("telegram", {})
    purge_section = toml_data.get("purge", {})

    # Build config with env vars taking priority over TOML
    config = Config(
        api_id=os.environ.get("TG_PURGE_API_ID") or tg_section.get("api_id"),
        api_hash=os.environ.get("TG_PURGE_API_HASH") or tg_section.get("api_hash"),
        session_path=(
            os.environ.get("TG_PURGE_SESSION")
            or tg_section.get("session_path")
            or ""
        ),
        default_channel=(
            os.environ.get("TG_PURGE_CHANNEL")
            or purge_section.get("default_channel")
        ),
        delay=float(purge_section.get("delay", 1.5)),
        threshold=int(purge_section.get("threshold", 2)),
    )

    return config
