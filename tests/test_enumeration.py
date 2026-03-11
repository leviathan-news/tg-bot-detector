"""Tests for enumeration query sets and recursive prefix expansion."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tg_purge.enumeration import (
    MINIMAL_QUERIES, FULL_QUERIES, EXPANSION_CHARS, RESULT_CAP,
    YieldTracker, enumerate_subscribers,
)


class TestQuerySets:
    """Verify query set coverage properties."""

    def test_full_queries_is_superset_of_minimal_single_chars(self):
        """Every single-char query in MINIMAL should appear in FULL."""
        minimal_single = {q for q in MINIMAL_QUERIES if len(q) == 1}
        full_single = {q for q in FULL_QUERIES if len(q) == 1}
        missing = minimal_single - full_single
        assert missing == set(), f"FULL_QUERIES missing single-char queries from MINIMAL: {missing}"

    def test_full_contains_cjk(self):
        """FULL_QUERIES should contain CJK characters."""
        cjk_in_full = [q for q in FULL_QUERIES if any("\u4e00" <= c <= "\u9fff" for c in q)]
        assert len(cjk_in_full) >= 2, f"Expected >= 2 CJK queries, got {len(cjk_in_full)}"

    def test_full_has_more_than_minimal(self):
        assert len(FULL_QUERIES) > len(MINIMAL_QUERIES)

    def test_both_include_empty_string(self):
        assert "" in MINIMAL_QUERIES
        assert "" in FULL_QUERIES

    def test_no_duplicate_queries_in_minimal(self):
        assert len(MINIMAL_QUERIES) == len(set(MINIMAL_QUERIES))

    def test_no_duplicate_queries_in_full(self):
        assert len(FULL_QUERIES) == len(set(FULL_QUERIES))


def _make_users(count, id_start=0):
    """Create a list of mock user objects with sequential IDs."""
    users = []
    for i in range(count):
        u = MagicMock()
        u.id = id_start + i
        users.append(u)
    return users


def _make_participants(count, id_start=0):
    """Create a list of mock participant objects (no join date for simplicity)."""
    parts = []
    for i in range(count):
        p = MagicMock()
        p.user_id = id_start + i
        # Make isinstance check fail for ChannelParticipant so join_dates
        # extraction is skipped in tests (avoids import issues).
        type(p).__name__ = "MockParticipant"
        parts.append(p)
    return parts


class TestRecursiveExpansion:
    """Test recursive prefix expansion when queries hit the 200-result cap."""

    def test_no_expansion_under_cap(self):
        """Queries returning < RESULT_CAP should not trigger expansion."""
        call_queries = []

        async def mock_fetch(client, channel, query, limit=200):
            call_queries.append(query)
            # Return 50 users (well under cap) — no expansion expected
            users = _make_users(50, id_start=len(call_queries) * 1000)
            return users, []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        # Use a tiny custom query set for test speed
        test_queries = ["a", "b"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=3,
            ))

        # Only the 2 base queries should have been called — no expansion
        assert call_queries == ["a", "b"]

    def test_expansion_when_hitting_cap(self):
        """Queries returning exactly RESULT_CAP should trigger sub-queries."""
        call_queries = []

        async def mock_fetch(client, channel, query, limit=200):
            call_queries.append(query)
            if query == "a":
                # Hit the cap — should trigger expansion
                users = _make_users(RESULT_CAP, id_start=0)
            else:
                # Sub-queries return small sets — no further expansion
                users = _make_users(5, id_start=len(call_queries) * 1000)
            return users, []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        test_queries = ["a"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=1,
            ))

        # "a" hit cap -> expanded to "a" + each EXPANSION_CHAR
        assert call_queries[0] == "a"
        expected_subs = {"a" + ch for ch in EXPANSION_CHARS}
        actual_subs = set(call_queries[1:])
        assert actual_subs == expected_subs

    def test_max_depth_zero_disables_expansion(self):
        """max_depth=0 should prevent any recursive expansion."""
        call_queries = []

        async def mock_fetch(client, channel, query, limit=200):
            call_queries.append(query)
            # Always hit cap — but expansion should be disabled
            return _make_users(RESULT_CAP, id_start=len(call_queries) * 1000), []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        test_queries = ["a", "b"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=0,
            ))

        # Only base queries — no expansion despite hitting cap
        assert call_queries == ["a", "b"]

    def test_multi_level_expansion(self):
        """Expansion should recurse up to max_depth levels."""
        call_queries = []

        async def mock_fetch(client, channel, query, limit=200):
            call_queries.append(query)
            if len(query) <= 2:
                # Depth 0 ("x") and depth 1 ("xa", "xb", ...) both hit cap
                return _make_users(RESULT_CAP, id_start=len(call_queries) * 1000), []
            else:
                # Depth 2 ("xaa", "xab", ...) returns small set
                return _make_users(3, id_start=len(call_queries) * 1000), []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        # Single base query to keep test manageable. Use only 2 expansion chars.
        test_queries = ["x"]
        test_expansion = ["a", "b"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries), \
             patch("tg_purge.enumeration.EXPANSION_CHARS", test_expansion):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=2,
            ))

        # Level 0: "x" (hits cap)
        # Level 1: "xa" (hits cap), "xb" (hits cap)
        # Level 2: "xaa", "xab", "xba", "xbb" (under cap, stop)
        assert "x" in call_queries
        assert "xa" in call_queries
        assert "xb" in call_queries
        assert "xaa" in call_queries
        assert "xab" in call_queries
        assert "xba" in call_queries
        assert "xbb" in call_queries
        # Total: 1 + 2 + 4 = 7 queries
        assert len(call_queries) == 7

    def test_deduplicates_users_across_queries(self):
        """Users found in multiple queries should only appear once."""
        async def mock_fetch(client, channel, query, limit=200):
            # Every query returns the same 5 users
            return _make_users(5, id_start=0), []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        test_queries = ["a", "b", "c"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=0,
            ))

        # Only 5 unique users despite 3 queries returning the same set
        assert len(result["users"]) == 5

    def test_error_handling_continues(self):
        """Errors on individual queries should not stop enumeration."""
        call_count = [0]

        async def mock_fetch(client, channel, query, limit=200):
            call_count[0] += 1
            if query == "b":
                raise Exception("network error")
            return _make_users(5, id_start=call_count[0] * 1000), []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        test_queries = ["a", "b", "c"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=0,
            ))

        # All 3 queries attempted despite error on "b"
        assert call_count[0] == 3
        # Users from "a" and "c" still collected
        assert len(result["users"]) > 0
        # Error query recorded in stats with 0 results
        error_stats = [s for s in result["query_stats"] if s[1] == 0]
        assert len(error_stats) == 1

class TestYieldTracker:
    """Unit tests for YieldTracker adaptive expansion helper."""

    def test_record_and_should_expand_at_cap(self):
        """A prefix recorded at exactly RESULT_CAP should be eligible for expansion."""
        tracker = YieldTracker()
        tracker.record("a", 200)
        assert tracker.should_expand("a") is True

    def test_should_not_expand_below_cap(self):
        """A prefix recorded below RESULT_CAP should not be eligible for expansion."""
        tracker = YieldTracker()
        tracker.record("b", 50)
        assert tracker.should_expand("b") is False

    def test_unrecorded_prefix_should_not_expand(self):
        """A prefix that was never recorded should not be eligible for expansion."""
        tracker = YieldTracker()
        assert tracker.should_expand("c") is False

    def test_record_overwrites(self):
        """Recording a second value for the same prefix replaces the first."""
        tracker = YieldTracker()
        tracker.record("a", 200)
        tracker.record("a", 10)
        # After overwrite the count is 10 — below cap, so no expansion.
        assert tracker.should_expand("a") is False

    def test_empty_tracker(self):
        """A freshly created tracker should report no prefix as expandable."""
        tracker = YieldTracker()
        assert tracker.should_expand("x") is False


    def test_progress_callback_called(self):
        """Progress callback should be called for each query including expansions."""
        progress_calls = []

        async def mock_fetch(client, channel, query, limit=200):
            return _make_users(5, id_start=0), []

        mock_client = AsyncMock()
        mock_channel = MagicMock()

        test_queries = ["a", "b"]
        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch), \
             patch("tg_purge.enumeration.MINIMAL_QUERIES", test_queries):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=0,
                progress_callback=lambda done, total, found: progress_calls.append((done, total, found)),
            ))

        assert len(progress_calls) == 2
        # Last call should show completion
        assert progress_calls[-1][0] == progress_calls[-1][1]
