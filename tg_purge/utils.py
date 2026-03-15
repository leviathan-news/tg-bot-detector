"""
Shared utility functions for tg-bot-detector.

Small, stateless helpers used across multiple modules. Keeping them here
avoids duplication without creating a heavy dependency.
"""


def channel_slug(channel: str) -> str:
    """Derive a filesystem-safe slug from a channel identifier.

    Strips a leading '@' and replaces every character that is not alphanumeric
    or an underscore with '_'.  Truncates to 64 characters to keep filesystem
    paths sensible.

    Args:
        channel: Channel identifier string, e.g. "@leviathan_news" or "12345".

    Returns:
        A safe slug string, e.g. "leviathan_news".
        Empty string when channel is None or blank.
    """
    if not channel:
        return ""
    slug = channel.lstrip("@")
    # Replace any character that is not alphanumeric or '_' with '_'.
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
    return slug[:64]
