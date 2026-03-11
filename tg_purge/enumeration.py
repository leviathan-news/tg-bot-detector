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
  - FULL_QUERIES (69 queries): Broader coverage across Latin, Cyrillic, Arabic,
    CJK, and numeric name patterns

When a query returns exactly 200 results (the API cap), recursive prefix
expansion drills deeper by appending characters to the query (e.g., "a" -> "aa",
"ab", ...). This is controlled by the max_depth parameter (default 3).

Neither set achieves full coverage. Results should be treated as samples,
not census data.
"""

import asyncio
import sys

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
    # CJK (common surnames — covers Chinese/Japanese/Korean bot names)
    "\u674e", "\u738b",  # 李, 王
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


# Characters used to expand queries that hit the 200-result cap.
# Lowercase Latin + digits covers the most common name prefixes and
# numeric bot-farm naming patterns (e.g., "User38291").
EXPANSION_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789")

# Telegram's hard cap per GetParticipantsRequest query.
RESULT_CAP = 200


class YieldTracker:
    """Tracks per-prefix query yields for adaptive expansion decisions.

    Records how many results each prefix returned. Prefixes that hit
    the RESULT_CAP are candidates for expansion; those that didn't
    are not worth expanding further.
    """

    def __init__(self):
        # Maps prefix string -> result count recorded for that prefix.
        self._yields = {}

    def record(self, prefix, count):
        """Record the result count for a prefix query.

        Overwrites any previously recorded count for the same prefix,
        so the most recent query result always wins.
        """
        self._yields[prefix] = count

    def should_expand(self, prefix):
        """Return True if prefix hit the result cap and should be expanded.

        A prefix is a candidate for expansion when its recorded result count
        equals RESULT_CAP, indicating the API truncated the result set and
        there may be additional matching subscribers beyond the cap.
        Returns False for any prefix that was never recorded.
        """
        return self._yields.get(prefix, 0) >= RESULT_CAP


async def enumerate_subscribers(client, channel, strategy="full", delay=1.5,
                                progress_callback=None, max_depth=3):
    """Enumerate channel subscribers using search-based sampling.

    When a query returns exactly RESULT_CAP (200) results, it likely has more
    matches that were truncated. If max_depth > 0, the query is recursively
    expanded by appending each character in EXPANSION_CHARS (e.g., "a" -> "aa",
    "ab", ...) up to max_depth levels. This splits large result sets into
    smaller buckets that fit within the 200-result cap.

    Args:
        client: Connected TelegramClient.
        channel: Resolved channel entity.
        strategy: "minimal" (22 queries) or "full" (69 queries).
        delay: Seconds between API calls (rate limiting).
        progress_callback: Optional callable(query_num, total_queries, total_found).
            During recursive expansion, total_queries increases dynamically.
        max_depth: Maximum recursion depth for prefix expansion (0 disables).
            Default 3 matches the original squid-bot#77 spec (a -> aa -> aaa).

    Returns:
        Dict with keys:
            - users: dict of user_id -> User object
            - participants: dict of user_id -> ChannelParticipant
            - join_dates: dict of user_id -> datetime (where available)
            - query_stats: list of (query, result_count, new_count)
    """
    base_queries = FULL_QUERIES if strategy == "full" else MINIMAL_QUERIES

    all_users = {}       # user_id -> User
    all_participants = {}  # user_id -> ChannelParticipant
    join_dates = {}      # user_id -> datetime
    query_stats = []

    # Work queue: (query_string, current_depth). Base queries start at depth 0.
    # Using deque for O(1) popleft instead of O(n) list.pop(0).
    from collections import deque
    work_queue = deque((q, 0) for q in base_queries)
    total_planned = len(work_queue)
    completed = 0

    while work_queue:
        query, depth = work_queue.popleft()
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

            # Recursive expansion: if we hit the cap and haven't exceeded
            # max_depth, generate sub-queries by appending expansion chars.
            if len(users) >= RESULT_CAP and depth < max_depth:
                sub_queries = [(query + ch, depth + 1) for ch in EXPANSION_CHARS]
                work_queue.extend(sub_queries)
                total_planned += len(sub_queries)

        except Exception as e:
            print(f"  Query {display_q}: error — {e}", file=sys.stderr)
            query_stats.append((display_q, 0, 0))

        completed += 1
        if progress_callback:
            progress_callback(completed, total_planned, len(all_users))

        await asyncio.sleep(delay)

    return {
        "users": all_users,
        "participants": all_participants,
        "join_dates": join_dates,
        "query_stats": query_stats,
    }
