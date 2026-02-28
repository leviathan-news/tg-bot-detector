"""
Telethon client setup and session management.

Handles interactive authentication on first run and channel resolution.
Session files are stored at ~/.tg_purge/session by default.

SECURITY WARNING: Telethon session files are equivalent to login credentials.
Anyone with access to your session file can act as your Telegram account.
Never share session files, commit them to git, or store them unencrypted
on shared systems. Consider enabling 2FA on your Telegram account.
"""

import stat
import sys
from pathlib import Path

from telethon import TelegramClient


_FIRST_RUN_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║  SECURITY WARNING: Session files are equivalent to login credentials ║
║                                                                      ║
║  The session file at {path}
║  grants full access to your Telegram account. NEVER:                 ║
║    • Commit it to git (already in .gitignore)                        ║
║    • Share it with anyone                                            ║
║    • Store it on shared/unencrypted systems                          ║
║                                                                      ║
║  Consider enabling 2FA on your Telegram account for extra safety.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


async def create_client(config):
    """Create and start a Telethon client.

    On first run, prompts for phone number and verification code
    interactively. Subsequent runs use the saved session.

    Args:
        config: A Config instance with api_id, api_hash, session_path.

    Returns:
        A connected TelegramClient.
    """
    config.validate_credentials()

    session_path = Path(config.session_path)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    # Restrict directory to owner-only access (700). Session files contain
    # Telegram authentication tokens equivalent to login credentials.
    try:
        session_path.parent.chmod(stat.S_IRWXU)
    except OSError:
        pass  # Best-effort — may fail on some filesystems (e.g., FAT32, network mounts)

    is_first_run = not session_path.with_suffix(".session").exists()

    client = TelegramClient(
        str(session_path),
        int(config.api_id),
        config.api_hash,
    )

    await client.start()

    # Restrict session file to owner read/write only (600) after creation.
    session_file = session_path.with_suffix(".session")
    if session_file.exists():
        try:
            session_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Best-effort

    if is_first_run:
        padded_path = str(session_path).ljust(50)
        print(_FIRST_RUN_WARNING.format(path=padded_path), file=sys.stderr)

    me = await client.get_me()
    print(f"Connected as: {me.first_name}", file=sys.stderr)

    return client


async def resolve_channel(client, channel_identifier):
    """Resolve a channel username or ID to a Telethon entity.

    Args:
        client: A connected TelegramClient.
        channel_identifier: Channel username (e.g., "@foo") or numeric ID.

    Returns:
        The resolved Telethon Channel entity.
    """
    # Handle numeric IDs
    try:
        channel_identifier = int(channel_identifier)
    except (ValueError, TypeError):
        pass

    entity = await client.get_entity(channel_identifier)
    sub_count = getattr(entity, "participants_count", None)
    print(f"Channel: {entity.title}", file=sys.stderr)
    if sub_count:
        print(f"Subscribers: {sub_count:,}", file=sys.stderr)
    return entity
