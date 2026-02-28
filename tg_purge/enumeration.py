"""
Subscriber enumeration via Telethon / MTProto.

Telegram's GetParticipantsRequest has inherent limitations:
  - Each query returns at most 200 results
  - ChannelParticipantsRecent returns ~200 most recently active users
  - ChannelParticipantsSearch searches by name/username prefix
  - Server-side enumeration ceiling is ~10K participants total
  - No truly random sampling is possible — all strategies have selection bias

Two search query sets are provided:
  - MINIMAL_QUERIES (22 queries): Fast sampling, good for quick assessments
  - FULL_QUERIES (67 queries): Broader coverage across Latin, Cyrillic, Arabic,
    CJK, and numeric name patterns

Neither set achieves full coverage. Results should be treated as samples,
not census data.
"""

import asyncio

from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import (
    ChannelParticipant,
    ChannelParticipantsBots,
    ChannelParticipantsRecent,
    ChannelParticipantsSearch,
)


# Quick sampling: common letters + a few uncommon combos + non-Latin + numbers
MINIMAL_QUERIES = [
    # Common Latin single letters — large result sets, bot-heavy
    "a", "m", "s", "d", "j", "k",
    # Uncommon Latin combos — smaller, different population
    "zq", "xw", "qj",
    # Cyrillic
    "\u0430", "\u043c", "\u0434", "\u0438",  # а, м, д, и
    # Arabic
    "\u0645", "\u0639",  # م, ع
    # CJK
    "\u674e", "\u738b",  # 李, 王
    # Numbers (bot farms often use numeric names)
    "0", "1", "7", "9",
    # Empty string — Telegram sometimes returns a different set
    "",
]

# Full coverage: all Latin + Cyrillic + Arabic + digits + empty
FULL_QUERIES = [
    # Latin alphabet — single letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Cyrillic
    "\u0430", "\u0431", "\u0432", "\u0433", "\u0434", "\u0435",
    "\u0436", "\u0437", "\u0438", "\u043a", "\u043b", "\u043c",
    "\u043d", "\u043e", "\u043f", "\u0440", "\u0441", "\u0442",
    "\u0443", "\u0444", "\u0445", "\u0446", "\u0447", "\u0448",
    # Arabic
    "\u0645", "\u0639", "\u0627", "\u0628", "\u062a", "\u0646",
    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Empty string
    "",
]


async def fetch_bots(client, channel, limit=200):
    """Fetch self-identified Bot API accounts subscribed to the channel.

    Args:
        client: Connected TelegramClient.
        channel: Resolved channel entity.
        limit: Max results (default 200, Telegram's hard cap).

    Returns:
        List of Telethon User objects.
    """
    result = await client(GetParticipantsRequest(
        channel=channel,
        filter=ChannelParticipantsBots(),
        offset=0,
        limit=limit,
        hash=0,
    ))
    return result.users


async def fetch_recent(client, channel, limit=200):
    """Fetch most recently active subscribers.

    Note: "recent" is determined by Telegram's internal activity tracking,
    not by join date. This skews toward active users and underrepresents
    dormant/bot accounts.

    Args:
        client: Connected TelegramClient.
        channel: Resolved channel entity.
        limit: Max results (default 200, Telegram's hard cap).

    Returns:
        Tuple of (users: list, participants: list) where participants
        contain join date metadata.
    """
    result = await client(GetParticipantsRequest(
        channel=channel,
        filter=ChannelParticipantsRecent(),
        offset=0,
        limit=limit,
        hash=0,
    ))
    return result.users, result.participants


async def fetch_by_search(client, channel, query, limit=200):
    """Fetch subscribers matching a search query.

    Searches by name and username prefix. Returns at most 200 results
    even if more match — there is no pagination for this endpoint.

    Args:
        client: Connected TelegramClient.
        channel: Resolved channel entity.
        query: Search string (name/username prefix).
        limit: Max results (default 200, Telegram's hard cap).

    Returns:
        Tuple of (users: list, participants: list) where participants
        contain join date metadata.
    """
    result = await client(GetParticipantsRequest(
        channel=channel,
        filter=ChannelParticipantsSearch(query),
        offset=0,
        limit=limit,
        hash=0,
    ))
    return result.users, result.participants


async def enumerate_subscribers(client, channel, strategy="full", delay=1.5,
                                progress_callback=None):
    """Enumerate channel subscribers using search-based sampling.

    Args:
        client: Connected TelegramClient.
        channel: Resolved channel entity.
        strategy: "minimal" (22 queries) or "full" (67 queries).
        delay: Seconds between API calls (rate limiting).
        progress_callback: Optional callable(query_num, total_queries, total_found).

    Returns:
        Dict with keys:
            - users: dict of user_id -> User object
            - participants: dict of user_id -> ChannelParticipant
            - join_dates: dict of user_id -> datetime (where available)
            - query_stats: list of (query, result_count, new_count)
    """
    queries = FULL_QUERIES if strategy == "full" else MINIMAL_QUERIES

    all_users = {}       # user_id -> User
    all_participants = {}  # user_id -> ChannelParticipant
    join_dates = {}      # user_id -> datetime
    query_stats = []

    for i, query in enumerate(queries):
        display_q = repr(query) if query else '""'
        try:
            users, participants = await fetch_by_search(client, channel, query)

            new_count = 0
            for user in users:
                if user.id not in all_users:
                    all_users[user.id] = user
                    new_count += 1

            for p in participants:
                if p.user_id not in all_participants:
                    all_participants[p.user_id] = p
                    if isinstance(p, ChannelParticipant) and hasattr(p, 'date') and p.date:
                        join_dates[p.user_id] = p.date

            query_stats.append((display_q, len(users), new_count))

            if progress_callback:
                progress_callback(i + 1, len(queries), len(all_users))

        except Exception as e:
            query_stats.append((display_q, 0, 0))
            if progress_callback:
                progress_callback(i + 1, len(queries), len(all_users))

        await asyncio.sleep(delay)

    return {
        "users": all_users,
        "participants": all_participants,
        "join_dates": join_dates,
        "query_stats": query_stats,
    }
