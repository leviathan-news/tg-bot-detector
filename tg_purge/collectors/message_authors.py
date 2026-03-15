"""
Message authors collector: discovers users from channel message history.

Uses GetHistoryRequest to iterate through recent channel messages in batches
of 100. Any user referenced in the history.users list is added to the result.

This collector complements the search-based API collector because it reaches
users who may not appear in name-search queries (e.g., users with names
consisting entirely of non-Latin, non-Cyrillic, non-Arabic characters, or
users whose names happen to never match a seed query prefix).

Limitation: only users who have commented or forwarded messages to the channel
appear in history.users. In broadcast channels with comments disabled, this
may return an empty or very small set.
"""

import asyncio
import sys

from .base import CollectorResult

# Telethon types are imported at the top level. This module requires Telethon
# to be installed — unlike scoring.py, there is no need to support import
# without Telethon here.
from telethon.tl.functions.messages import GetHistoryRequest


async def collect(
    client,
    channel,
    limit: int = 1000,
    delay: float = 1.0,
) -> CollectorResult:
    """Collect users who appear as authors in the channel's message history.

    Iterates messages in batches of 100 (Telegram's practical page size for
    history requests) until `limit` total messages have been scanned or the
    channel history is exhausted.

    Args:
        client: Connected Telethon TelegramClient.
        channel: Resolved Telethon channel entity.
        limit: Maximum number of messages to scan. Default 1000.
        delay: Seconds to sleep between consecutive API calls. Default 1.0.

    Returns:
        CollectorResult with:
            source="message_authors"
            users: dict[int, User] — deduplicated users found in history
            metadata["messages_scanned"]: int — total messages fetched
    """
    users: dict = {}
    messages_scanned = 0
    offset_id = 0          # Pagination cursor: start from the newest message.
    batch_size = 100       # Telegram's practical page size for history.

    while messages_scanned < limit:
        # Calculate how many messages to request in this batch.
        remaining = limit - messages_scanned
        fetch_count = min(batch_size, remaining)

        try:
            history = await client(GetHistoryRequest(
                peer=channel,
                offset_id=offset_id,      # Fetch messages older than this ID.
                offset_date=None,
                add_offset=0,
                limit=fetch_count,
                max_id=0,
                min_id=0,
                hash=0,
            ))
        except Exception as e:
            # Non-fatal: log error and stop further pagination to avoid
            # spamming the API on a persistent error.
            print(
                f"  message_authors collector: error fetching history — {e}",
                file=sys.stderr,
            )
            break

        # GetHistoryRequest populates history.users with all users referenced
        # in the fetched batch (authors, forwarders, reply targets, etc.).
        for user in history.users:
            if user.id not in users:
                users[user.id] = user

        batch_len = len(history.messages)
        messages_scanned += batch_len

        # If we received fewer messages than requested, we've hit the start of
        # the channel history — no further pages to fetch.
        if batch_len < fetch_count:
            break

        # Advance the pagination cursor to the ID of the oldest message in
        # this batch so the next request fetches older messages.
        if history.messages:
            offset_id = history.messages[-1].id

        await asyncio.sleep(delay)

    return CollectorResult(
        source="message_authors",
        users=users,
        metadata={"messages_scanned": messages_scanned},
    )
