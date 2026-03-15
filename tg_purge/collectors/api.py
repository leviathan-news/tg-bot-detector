"""
API collector: wraps enumerate_subscribers() into a CollectorResult.

This is the primary collector for most CLI commands. It uses the search-based
prefix-expansion strategy from tg_purge.enumeration and returns all discovered
users, participants, join dates, and per-query statistics.
"""

from .base import CollectorResult
from ..enumeration import enumerate_subscribers


async def collect(
    client,
    channel,
    strategy: str = "full",
    delay: float = 1.5,
    progress_callback=None,
    max_depth: int = 3,
) -> CollectorResult:
    """Enumerate subscribers using the search-based API strategy.

    Delegates to enumerate_subscribers() from tg_purge.enumeration and wraps
    the raw dict result in a CollectorResult for uniform downstream handling.

    Args:
        client: Connected Telethon TelegramClient.
        channel: Resolved Telethon channel entity.
        strategy: "full" (69 seed queries) or "minimal" (22 seed queries).
        delay: Seconds to sleep between consecutive API calls.
        progress_callback: Optional callable(query_num, total_queries,
            total_found) called after each completed query. Useful for
            displaying progress in CLI commands.
        max_depth: Maximum recursive prefix-expansion depth (0 disables).
            Default 3 matches the original squid-bot#77 spec.

    Returns:
        CollectorResult with:
            source="api"
            users: dict[int, User]
            participants: dict[int, ChannelParticipant]
            join_dates: dict[int, datetime]
            metadata["query_stats"]: list of (query, result_count, new_count)
    """
    # enumerate_subscribers returns a plain dict with keys:
    #   users, participants, join_dates, query_stats
    result = await enumerate_subscribers(
        client,
        channel,
        strategy=strategy,
        delay=delay,
        progress_callback=progress_callback,
        max_depth=max_depth,
    )

    return CollectorResult(
        source="api",
        users=result["users"],
        participants=result["participants"],
        join_dates=result["join_dates"],
        metadata={"query_stats": result["query_stats"]},
    )
