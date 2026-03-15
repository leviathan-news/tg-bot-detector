"""Tests for cross-channel cohort detection."""

import pytest
from datetime import datetime, timezone, timedelta

from tg_purge.cross_channel import find_cohorts, score_cohort


class TestFindCohorts:
    """Test find_cohorts() which identifies users appearing in multiple channels."""

    def test_no_overlap(self):
        """Completely disjoint user sets produce no cohorts."""
        channel_users = {
            "chan_a": {1, 2, 3, 4, 5},
            "chan_b": {6, 7, 8, 9, 10},
            "chan_c": {11, 12, 13, 14, 15},
        }
        result = find_cohorts(channel_users, min_cohort_size=1, min_shared_channels=3)
        assert result == []

    def test_large_overlap_detected(self):
        """60 users present in 3 channels are detected as a cohort."""
        # Create 60 users who appear in all three channels
        shared_users = set(range(1, 61))
        channel_users = {
            "chan_a": shared_users | {100, 101},  # extra non-shared users
            "chan_b": shared_users | {200, 201},
            "chan_c": shared_users | {300, 301},
        }
        result = find_cohorts(channel_users, min_cohort_size=50, min_shared_channels=3)
        assert len(result) >= 1
        # The main cohort must contain all 60 shared users
        found = result[0]
        assert found["user_ids"] == shared_users
        assert set(found["shared_channels"]) == {"chan_a", "chan_b", "chan_c"}

    def test_small_overlap_ignored(self):
        """Only 3 shared users in 3 channels — below min_cohort_size=50."""
        shared_users = {1, 2, 3}
        channel_users = {
            "chan_a": shared_users | {10, 11},
            "chan_b": shared_users | {20, 21},
            "chan_c": shared_users | {30, 31},
        }
        result = find_cohorts(channel_users, min_cohort_size=50, min_shared_channels=3)
        assert result == []

    def test_insufficient_channels(self):
        """With only 2 channels, min_shared_channels=3 can never be met — return []."""
        channel_users = {
            "chan_a": set(range(1, 100)),
            "chan_b": set(range(1, 100)),
        }
        result = find_cohorts(channel_users, min_cohort_size=1, min_shared_channels=3)
        assert result == []

    def test_custom_min_size(self):
        """With min_cohort_size=5, a group of 10 shared users should be detected."""
        shared_users = set(range(1, 11))  # 10 users
        channel_users = {
            "chan_a": shared_users | {100},
            "chan_b": shared_users | {200},
            "chan_c": shared_users | {300},
        }
        result = find_cohorts(channel_users, min_cohort_size=5, min_shared_channels=3)
        assert len(result) >= 1
        assert result[0]["user_ids"] == shared_users

    def test_result_structure(self):
        """Each returned item must have 'user_ids' (set) and 'shared_channels' (list)."""
        shared_users = set(range(1, 60))
        channel_users = {
            "chan_a": shared_users,
            "chan_b": shared_users,
            "chan_c": shared_users,
        }
        result = find_cohorts(channel_users, min_cohort_size=50, min_shared_channels=3)
        assert len(result) >= 1
        for item in result:
            assert "user_ids" in item
            assert "shared_channels" in item
            assert isinstance(item["user_ids"], set)
            assert isinstance(item["shared_channels"], list)

    def test_multiple_cohorts_different_channel_sets(self):
        """Users grouped by different channel-set combinations produce separate cohorts."""
        # Group A: in channels 1, 2, 3
        group_a = set(range(1, 61))
        # Group B: in channels 1, 2, 4 (different channel fingerprint)
        group_b = set(range(100, 161))
        channel_users = {
            "chan_1": group_a | group_b,
            "chan_2": group_a | group_b,
            "chan_3": group_a,
            "chan_4": group_b,
        }
        result = find_cohorts(channel_users, min_cohort_size=50, min_shared_channels=3)
        assert len(result) == 2

    def test_empty_channel_users(self):
        """Empty input returns empty list."""
        result = find_cohorts({}, min_cohort_size=1, min_shared_channels=3)
        assert result == []

    def test_single_channel(self):
        """Single channel cannot satisfy min_shared_channels=3."""
        channel_users = {"chan_a": set(range(1, 100))}
        result = find_cohorts(channel_users, min_cohort_size=1, min_shared_channels=3)
        assert result == []


class TestScoreCohort:
    """Test score_cohort() which evaluates how suspicious a cohort is."""

    def _make_join_times(self, user_ids, channels, base_time, spread_seconds=0):
        """Helper: build join_times dict with all users joining near base_time."""
        join_times = {}
        for i, uid in enumerate(user_ids):
            join_times[uid] = {}
            for ch in channels:
                # Evenly distribute within spread_seconds
                offset = (i * spread_seconds / max(len(user_ids) - 1, 1)) if len(user_ids) > 1 else 0
                join_times[uid][ch] = base_time + timedelta(seconds=offset)
        return join_times

    def test_coordinated_cohort_suspicious(self):
        """60 users, all joined within 2 hours, same profile → is_suspicious=True."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        # 2 hours = 7200 seconds spread
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=7200)
        # All users have identical profile features
        profiles = {uid: {"has_photo": False, "username": None, "status": "empty"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is True
        assert result["join_spread_hours"] < 48
        assert "cohort_size" in result
        assert result["cohort_size"] == 60
        assert "confidence" in result
        assert "profile_similarity" in result

    def test_organic_cohort_not_suspicious(self):
        """60 users who joined over several months with diverse profiles → not suspicious."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Spread over 180 days = 15_552_000 seconds
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=15_552_000)
        # Diverse profiles: each user has a unique username
        profiles = {uid: {"has_photo": True, "username": f"user_{uid}", "status": "online"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is False

    def test_confidence_high(self):
        """Tight join spread (<6h) and high similarity (>0.8) → confidence='high'."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        # 3 hours spread (under 6h threshold)
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=10800)
        # Perfectly uniform profiles (similarity = 1.0)
        profiles = {uid: {"has_photo": False, "status": "empty"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is True
        assert result["confidence"] == "high"

    def test_confidence_medium(self):
        """join_spread_hours stddev in [6, 24) → confidence='medium' (when suspicious).

        Spread is measured as pstdev of all join timestamps, not max-min.
        For 60 evenly-spaced users, pstdev ≈ total_range / (2*sqrt(3)).
        90000 seconds (25h range) → stddev ≈ 7.3h, which is in [6h, 24h).
        """
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        # 25-hour range → stddev ≈ 7.3h, inside the medium tier [6h, 24h)
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=90000)
        profiles = {uid: {"has_photo": False, "status": "empty"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is True
        assert result["confidence"] == "medium"

    def test_confidence_low(self):
        """join_spread_hours stddev in [24, 48) → confidence='low' (when suspicious).

        360000 seconds (100h range) → stddev ≈ 29.4h, inside the low tier [24h, 48h).
        """
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        # 100-hour range → stddev ≈ 29.4h, inside the low tier [24h, 48h)
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=360000)
        profiles = {uid: {"has_photo": False, "status": "empty"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is True
        assert result["confidence"] == "low"

    def test_confidence_none_when_not_suspicious(self):
        """Non-suspicious cohort → confidence='none'."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 180-day spread — way over 48h threshold
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=15_552_000)
        profiles = {uid: {"username": f"user_{uid}"} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is False
        assert result["confidence"] == "none"

    def test_empty_profiles(self):
        """Empty profiles dict → profile_similarity=0.0, not suspicious."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        join_times = self._make_join_times(user_ids, channels, base, spread_seconds=3600)
        profiles = {}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["profile_similarity"] == 0.0
        # Can't be suspicious with 0.0 similarity (< 0.6 threshold)
        assert result["is_suspicious"] is False

    def test_return_shape(self):
        """Return dict always contains all required keys."""
        user_ids = [1, 2, 3]
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        join_times = self._make_join_times(user_ids, channels, base)
        profiles = {uid: {"x": 1} for uid in user_ids}

        result = score_cohort(user_ids, channels, join_times, profiles)

        required_keys = {
            "is_suspicious", "join_spread_hours", "profile_similarity",
            "cohort_size", "confidence",
        }
        assert required_keys.issubset(result.keys())

    def test_single_user_not_suspicious(self):
        """A cohort of 1 user can never hit the size >= 50 threshold."""
        user_ids = [1]
        channels = ["chan_a", "chan_b", "chan_c"]
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        join_times = self._make_join_times(user_ids, channels, base)
        profiles = {1: {"status": "empty"}}

        result = score_cohort(user_ids, channels, join_times, profiles)

        assert result["is_suspicious"] is False

    def test_no_join_times_spread_zero(self):
        """Empty join_times → join_spread_hours = 0.0."""
        user_ids = list(range(1, 61))
        channels = ["chan_a", "chan_b", "chan_c"]

        result = score_cohort(user_ids, channels, join_times={}, profiles={})

        assert result["join_spread_hours"] == 0.0
        # size >= 50 BUT spread < 48 AND similarity == 0.0 — fails similarity threshold
        assert result["is_suspicious"] is False
