"""
Tests for the tg_purge.collectors pluggable data collection framework.

These tests only exercise CollectorResult (base.py) and do not require
a live Telegram connection. All Telethon API calls are kept in the
collector modules (api.py, message_authors.py, admin_log.py) and are
not tested here — they would require integration test harnesses.
"""

import pytest
from datetime import datetime, timezone

from tg_purge.collectors.base import CollectorResult


class TestCollectorResult:
    """Unit tests for CollectorResult dataclass and its merge() static method."""

    def test_empty_result(self):
        """CollectorResult created with only source has empty collection fields."""
        r = CollectorResult(source="test")
        # All three collection dicts must be empty — not None, not shared objects.
        assert r.users == {}
        assert r.participants == {}
        assert r.join_dates == {}
        assert r.metadata == {}

    def test_source_set(self):
        """source field is stored as provided."""
        r = CollectorResult(source="api")
        assert r.source == "api"

        r2 = CollectorResult(source="message_authors")
        assert r2.source == "message_authors"

    def test_merge_deduplicates(self):
        """merge() deduplicates users that appear in more than one result."""
        # Two results share user_id=1 — merged result should contain it once.
        r1 = CollectorResult(source="a", users={1: "user_a1", 2: "user_a2"})
        r2 = CollectorResult(source="b", users={1: "user_b1", 3: "user_b3"})

        merged = CollectorResult.merge([r1, r2])

        assert len(merged.users) == 3
        # All three unique IDs must be present.
        assert 1 in merged.users
        assert 2 in merged.users
        assert 3 in merged.users

    def test_merge_first_seen_wins(self):
        """When two results have the same user_id, the first-seen value is kept."""
        r1 = CollectorResult(source="first", users={1: "first_value"})
        r2 = CollectorResult(source="second", users={1: "second_value"})

        merged = CollectorResult.merge([r1, r2])

        # r1 is first in the list — its value must survive.
        assert merged.users[1] == "first_value"

    def test_merge_empty_list(self):
        """merge([]) returns a CollectorResult with empty collections."""
        merged = CollectorResult.merge([])

        assert merged.source == "merged"
        assert merged.users == {}
        assert merged.participants == {}
        assert merged.join_dates == {}

    def test_merge_single_result(self):
        """merge([r]) with a single result propagates all data unchanged."""
        jd = datetime(2023, 6, 1, tzinfo=timezone.utc)
        r = CollectorResult(
            source="solo",
            users={10: "u10", 20: "u20"},
            participants={10: "p10"},
            join_dates={10: jd},
        )

        merged = CollectorResult.merge([r])

        assert merged.users == {10: "u10", 20: "u20"}
        assert merged.participants == {10: "p10"}
        assert merged.join_dates == {10: jd}

    def test_merge_preserves_join_dates(self):
        """join_dates from both results are merged; first-seen wins on collision."""
        jd1 = datetime(2023, 1, 1, tzinfo=timezone.utc)
        jd2 = datetime(2023, 6, 1, tzinfo=timezone.utc)
        jd3 = datetime(2024, 1, 1, tzinfo=timezone.utc)

        r1 = CollectorResult(source="a", join_dates={1: jd1, 2: jd2})
        r2 = CollectorResult(source="b", join_dates={2: jd3, 3: jd3})

        merged = CollectorResult.merge([r1, r2])

        # Key 1 only in r1 — must be present.
        assert merged.join_dates[1] == jd1
        # Key 2 in both — first-seen (r1) must win.
        assert merged.join_dates[2] == jd2
        # Key 3 only in r2 — must be present.
        assert merged.join_dates[3] == jd3
        assert len(merged.join_dates) == 3

    def test_merge_participants_deduplication(self):
        """merge() deduplicates participants the same way it does users."""
        r1 = CollectorResult(source="a", participants={1: "part_a", 2: "part_a2"})
        r2 = CollectorResult(source="b", participants={1: "part_b", 3: "part_b3"})

        merged = CollectorResult.merge([r1, r2])

        assert len(merged.participants) == 3
        # First-seen wins for participants too.
        assert merged.participants[1] == "part_a"
        assert merged.participants[2] == "part_a2"
        assert merged.participants[3] == "part_b3"

    def test_merge_source_is_merged(self):
        """The merged result always has source='merged' regardless of inputs."""
        r1 = CollectorResult(source="api")
        r2 = CollectorResult(source="admin_log")

        merged = CollectorResult.merge([r1, r2])

        assert merged.source == "merged"

    def test_default_source_is_empty_string(self):
        """CollectorResult() with no arguments has source='' and empty dicts."""
        r = CollectorResult()
        assert r.source == ""
        assert r.users == {}

    def test_independent_default_dicts(self):
        """Each CollectorResult instance has its own dict objects (no shared state)."""
        r1 = CollectorResult()
        r2 = CollectorResult()

        # Mutating r1's dict must not affect r2.
        r1.users[999] = "leaked"

        assert 999 not in r2.users
