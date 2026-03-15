"""
Base dataclass for collector results.

All collector modules (api, message_authors, admin_log) return a CollectorResult.
The merge() static method combines multiple results into one, deduplicating by
user_id using a first-seen-wins strategy: the first result in the list whose
value for a given key is used if that key has already been seen.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class CollectorResult:
    """Result from a single collector.

    Attributes:
        source: Human-readable name of the collector that produced this result
            (e.g., "api", "message_authors", "admin_log", "merged").
        users: Mapping of user_id (int) -> Telethon User object (or mock in tests).
        participants: Mapping of user_id (int) -> ChannelParticipant object.
        join_dates: Mapping of user_id (int) -> datetime of when the user joined.
        metadata: Arbitrary collector-specific metadata (e.g., query_stats,
            messages_scanned, events_scanned).
    """

    source: str = ""
    users: Dict[int, Any] = field(default_factory=dict)
    participants: Dict[int, Any] = field(default_factory=dict)
    join_dates: Dict[int, datetime] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def merge(results: List["CollectorResult"]) -> "CollectorResult":
        """Merge multiple CollectorResults, deduplicating by user_id.

        Iterates over the provided results in order. For each mapping (users,
        participants, join_dates), a key is added to the merged result only if
        it has not already been seen — first-seen wins. This preserves the
        ordering guarantee: the first result in the list is treated as the
        most authoritative source for any overlapping key.

        Metadata is not merged — it varies per-collector and is left empty in
        the merged result.

        Args:
            results: Ordered list of CollectorResult objects to merge.

        Returns:
            A new CollectorResult with source="merged" and deduplicated data
            from all input results.
        """
        merged = CollectorResult(source="merged")

        for r in results:
            # Merge users: only add entries not yet seen in the merged result.
            for uid, user in r.users.items():
                if uid not in merged.users:
                    merged.users[uid] = user

            # Merge participants with the same deduplication logic.
            for uid, p in r.participants.items():
                if uid not in merged.participants:
                    merged.participants[uid] = p

            # Merge join dates — first-seen date wins, later ones are ignored.
            for uid, jd in r.join_dates.items():
                if uid not in merged.join_dates:
                    merged.join_dates[uid] = jd

        return merged
