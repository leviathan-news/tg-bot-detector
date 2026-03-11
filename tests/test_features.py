"""
Tests for tg_purge/features.py — ML feature extraction module.

Tests are organised into classes mirroring the logical feature groups.
All tests run without a real Telegram connection — MockUser + mock status
classes from conftest.py are used throughout.
"""

import pytest
from datetime import datetime, timezone, timedelta

from tests.conftest import (
    UserStatusEmpty,
    UserStatusOnline,
    UserStatusRecently,
    UserStatusLastWeek,
    UserStatusLastMonth,
    UserStatusOffline,
    MockUser,
)
from tg_purge.features import extract_features, FEATURE_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offline(days_ago: int) -> UserStatusOffline:
    """Return a UserStatusOffline whose was_online is `days_ago` days in the past."""
    return UserStatusOffline(
        was_online=datetime.now(timezone.utc) - timedelta(days=days_ago)
    )


# ---------------------------------------------------------------------------
# TestAccountFlags
# ---------------------------------------------------------------------------

class TestAccountFlags:
    """Binary account flag features: is_deleted, is_bot, is_scam, is_fake,
    is_restricted, is_premium, has_emoji_status."""

    def test_deleted_flag_true(self, make_user):
        user = make_user(deleted=True, first_name=None)
        f = extract_features(user)
        assert f["is_deleted"] == 1.0

    def test_deleted_flag_false(self, make_user):
        user = make_user(deleted=False)
        f = extract_features(user)
        assert f["is_deleted"] == 0.0

    def test_bot_flag_true(self, make_user):
        user = make_user(bot=True)
        f = extract_features(user)
        assert f["is_bot"] == 1.0

    def test_bot_flag_false(self, make_user):
        user = make_user(bot=False)
        f = extract_features(user)
        assert f["is_bot"] == 0.0

    def test_scam_flag_true(self, make_user):
        user = make_user(scam=True)
        f = extract_features(user)
        assert f["is_scam"] == 1.0

    def test_scam_flag_false(self, make_user):
        user = make_user(scam=False)
        f = extract_features(user)
        assert f["is_scam"] == 0.0

    def test_fake_flag_true(self, make_user):
        user = make_user(fake=True)
        f = extract_features(user)
        assert f["is_fake"] == 1.0

    def test_fake_flag_false(self, make_user):
        user = make_user(fake=False)
        f = extract_features(user)
        assert f["is_fake"] == 0.0

    def test_restricted_flag_true(self, make_user):
        user = make_user(restricted=True)
        f = extract_features(user)
        assert f["is_restricted"] == 1.0

    def test_restricted_flag_false(self, make_user):
        user = make_user(restricted=False)
        f = extract_features(user)
        assert f["is_restricted"] == 0.0

    def test_premium_flag_true(self, make_user):
        user = make_user(premium=True)
        f = extract_features(user)
        assert f["is_premium"] == 1.0

    def test_premium_flag_false(self, make_user):
        user = make_user(premium=False)
        f = extract_features(user)
        assert f["is_premium"] == 0.0

    def test_emoji_status_present(self, make_user):
        user = make_user(emoji_status="star")
        f = extract_features(user)
        assert f["has_emoji_status"] == 1.0

    def test_emoji_status_absent(self, make_user):
        user = make_user(emoji_status=None)
        f = extract_features(user)
        assert f["has_emoji_status"] == 0.0


# ---------------------------------------------------------------------------
# TestProfileFeatures
# ---------------------------------------------------------------------------

class TestProfileFeatures:
    """Profile completeness features: has_photo, has_username, has_last_name,
    first_name_length, name_digit_ratio, script_count."""

    def test_has_photo_true(self, make_user):
        user = make_user(photo=True)
        f = extract_features(user)
        assert f["has_photo"] == 1.0

    def test_has_photo_false(self, make_user):
        user = make_user(photo=False)
        f = extract_features(user)
        assert f["has_photo"] == 0.0

    def test_has_username_true(self, make_user):
        user = make_user(username="alice")
        f = extract_features(user)
        assert f["has_username"] == 1.0

    def test_has_username_false(self, make_user):
        user = make_user(username=None)
        f = extract_features(user)
        assert f["has_username"] == 0.0

    def test_has_last_name_true(self, make_user):
        user = make_user(last_name="Smith")
        f = extract_features(user)
        assert f["has_last_name"] == 1.0

    def test_has_last_name_false(self, make_user):
        user = make_user(last_name=None)
        f = extract_features(user)
        assert f["has_last_name"] == 0.0

    def test_first_name_length_normal(self, make_user):
        user = make_user(first_name="Alice")
        f = extract_features(user)
        assert f["first_name_length"] == 5.0

    def test_first_name_length_none(self, make_user):
        # Deleted users or users with no first name — length should be 0.
        user = make_user(first_name=None)
        f = extract_features(user)
        assert f["first_name_length"] == 0.0

    def test_name_digit_ratio_all_digits(self, make_user):
        user = make_user(first_name="12345")
        f = extract_features(user)
        assert f["name_digit_ratio"] == pytest.approx(1.0)

    def test_name_digit_ratio_mixed(self, make_user):
        # "ab12" — 2 digits out of 4 chars = 0.5
        user = make_user(first_name="ab12")
        f = extract_features(user)
        assert f["name_digit_ratio"] == pytest.approx(0.5)

    def test_name_digit_ratio_no_digits(self, make_user):
        user = make_user(first_name="Alice")
        f = extract_features(user)
        assert f["name_digit_ratio"] == pytest.approx(0.0)

    def test_name_digit_ratio_no_name(self, make_user):
        # Guard against ZeroDivisionError when first_name is None/empty.
        user = make_user(first_name=None)
        f = extract_features(user)
        assert f["name_digit_ratio"] == pytest.approx(0.0)

    def test_script_count_latin_only(self, make_user):
        user = make_user(first_name="Alice")
        f = extract_features(user)
        assert f["script_count"] == 1.0

    def test_script_count_cyrillic_only(self, make_user):
        user = make_user(first_name="Алиса")
        f = extract_features(user)
        assert f["script_count"] == 1.0

    def test_script_count_arabic_only(self, make_user):
        user = make_user(first_name="عليسا")
        f = extract_features(user)
        assert f["script_count"] == 1.0

    def test_script_count_cjk_only(self, make_user):
        # CJK character — should count as its own script family.
        user = make_user(first_name="李")
        f = extract_features(user)
        assert f["script_count"] == 1.0

    def test_script_count_latin_and_cyrillic(self, make_user):
        # Mixed scripts: one Latin letter + one Cyrillic letter.
        user = make_user(first_name="Aа")
        f = extract_features(user)
        assert f["script_count"] == 2.0

    def test_script_count_no_name(self, make_user):
        user = make_user(first_name=None)
        f = extract_features(user)
        assert f["script_count"] == 0.0


# ---------------------------------------------------------------------------
# TestActivityFeatures
# ---------------------------------------------------------------------------

class TestActivityFeatures:
    """One-hot status features and days_since_last_seen."""

    def _assert_only_one_hot(self, features: dict, active_key: str):
        """Assert exactly one of the status_* keys is 1, and it's active_key."""
        status_keys = [
            "status_empty", "status_online", "status_recently",
            "status_last_week", "status_last_month", "status_offline",
        ]
        for key in status_keys:
            expected = 1.0 if key == active_key else 0.0
            assert features[key] == expected, (
                f"Expected {key}={expected}, got {features[key]}"
            )

    def test_status_none_maps_to_empty(self, make_user):
        # status=None is treated the same as UserStatusEmpty.
        user = make_user(status=None)
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_empty")

    def test_status_empty(self, make_user):
        user = make_user(status=UserStatusEmpty())
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_empty")

    def test_status_online(self, make_user):
        user = make_user(status=UserStatusOnline())
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_online")

    def test_status_recently(self, make_user):
        user = make_user(status=UserStatusRecently())
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_recently")

    def test_status_last_week(self, make_user):
        user = make_user(status=UserStatusLastWeek())
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_last_week")

    def test_status_last_month(self, make_user):
        user = make_user(status=UserStatusLastMonth())
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_last_month")

    def test_status_offline(self, make_user):
        user = make_user(status=_offline(10))
        f = extract_features(user)
        self._assert_only_one_hot(f, "status_offline")

    # days_since_last_seen ---------------------------------------------------

    def test_days_since_empty_is_minus_one(self, make_user):
        user = make_user(status=UserStatusEmpty())
        f = extract_features(user)
        assert f["days_since_last_seen"] == -1.0

    def test_days_since_none_is_minus_one(self, make_user):
        user = make_user(status=None)
        f = extract_features(user)
        assert f["days_since_last_seen"] == -1.0

    def test_days_since_online_is_zero(self, make_user):
        user = make_user(status=UserStatusOnline())
        f = extract_features(user)
        assert f["days_since_last_seen"] == 0.0

    def test_days_since_recently_is_one(self, make_user):
        user = make_user(status=UserStatusRecently())
        f = extract_features(user)
        assert f["days_since_last_seen"] == 1.0

    def test_days_since_last_week_is_seven(self, make_user):
        user = make_user(status=UserStatusLastWeek())
        f = extract_features(user)
        assert f["days_since_last_seen"] == 7.0

    def test_days_since_last_month_is_thirty(self, make_user):
        user = make_user(status=UserStatusLastMonth())
        f = extract_features(user)
        assert f["days_since_last_seen"] == 30.0

    def test_days_since_offline_actual_days(self, make_user):
        user = make_user(status=_offline(45))
        f = extract_features(user)
        # Allow ±1 day tolerance for test execution time drift.
        assert 44.0 <= f["days_since_last_seen"] <= 46.0


# ---------------------------------------------------------------------------
# TestTemporalFeatures
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    """Temporal features: is_spike_join, days_since_join, join_hour_utc,
    join_day_of_week. All default to -1 or 0 when join_date is None."""

    def test_no_join_date_defaults(self, make_user):
        user = make_user()
        f = extract_features(user)
        assert f["is_spike_join"] == 0.0
        assert f["days_since_join"] == -1.0
        assert f["join_hour_utc"] == -1.0
        assert f["join_day_of_week"] == -1.0

    def test_days_since_join_calculated(self, make_user):
        join_date = datetime.now(timezone.utc) - timedelta(days=100)
        user = make_user()
        f = extract_features(user, join_date=join_date)
        assert 99.0 <= f["days_since_join"] <= 101.0

    def test_join_hour_utc(self, make_user):
        # Fix a join time with a known hour.
        join_date = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date)
        assert f["join_hour_utc"] == 14.0

    def test_join_day_of_week(self, make_user):
        # 2024-06-15 is a Saturday (weekday index 5).
        join_date = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date)
        assert f["join_day_of_week"] == 5.0

    def test_is_spike_join_inside_window(self, make_user):
        join_date = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        window_start = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date, spike_windows=[(window_start, window_end)])
        assert f["is_spike_join"] == 1.0

    def test_is_spike_join_outside_window(self, make_user):
        join_date = datetime(2024, 6, 16, 10, 0, 0, tzinfo=timezone.utc)
        window_start = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date, spike_windows=[(window_start, window_end)])
        assert f["is_spike_join"] == 0.0

    def test_is_spike_join_no_windows(self, make_user):
        join_date = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date, spike_windows=None)
        assert f["is_spike_join"] == 0.0

    def test_is_spike_join_empty_windows(self, make_user):
        join_date = datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=join_date, spike_windows=[])
        assert f["is_spike_join"] == 0.0

    def test_is_spike_join_at_window_end_exclusive(self, make_user):
        # Boundary: join_date == window_end is NOT inside the window (half-open interval).
        window_start = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window_end = datetime(2024, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
        user = make_user()
        f = extract_features(user, join_date=window_end, spike_windows=[(window_start, window_end)])
        assert f["is_spike_join"] == 0.0


# ---------------------------------------------------------------------------
# TestHeuristicScoreFeature
# ---------------------------------------------------------------------------

class TestHeuristicScoreFeature:
    """heuristic_score is the raw output of score_user() with default config."""

    def test_clean_user_low_score(self, clean_user):
        # clean_user has photo, username, last name, recently status — should score 0.
        f = extract_features(clean_user)
        assert f["heuristic_score"] == 0.0

    def test_deleted_user_score(self, deleted_user):
        # Deleted account gets +5.
        f = extract_features(deleted_user)
        assert f["heuristic_score"] == 5.0

    def test_bot_like_user_high_score(self, make_user):
        # no photo, no username, no last name, no status, single-char name → multiple signals.
        user = make_user(
            first_name="A",
            last_name=None,
            username=None,
            photo=False,
            status=None,
        )
        f = extract_features(user)
        assert f["heuristic_score"] >= 3.0

    def test_premium_user_low_score(self, premium_user):
        # Premium + emoji_status are positive signals that reduce the score.
        f = extract_features(premium_user)
        # Premium users should score low (premium=-2, emoji=-1 offset other penalties).
        assert f["heuristic_score"] <= 2.0

    def test_scam_user_high_score(self, make_user):
        user = make_user(scam=True)
        f = extract_features(user)
        assert f["heuristic_score"] >= 5.0


# ---------------------------------------------------------------------------
# TestCohortFeatures
# ---------------------------------------------------------------------------

class TestCohortFeatures:
    """Cohort features: is_cohort_member, cohort_size, cohort_join_spread_hours,
    cohort_profile_similarity. All default to 0/0.0 when cohort_data is None."""

    def test_no_cohort_data_all_zeros(self, make_user):
        user = make_user()
        f = extract_features(user, cohort_data=None)
        assert f["is_cohort_member"] == 0.0
        assert f["cohort_size"] == 0.0
        assert f["cohort_join_spread_hours"] == 0.0
        assert f["cohort_profile_similarity"] == 0.0

    def test_cohort_data_is_member(self, make_user):
        user = make_user()
        cohort_data = {
            "is_member": True,
            "size": 42,
            "join_spread_hours": 3.5,
            "profile_similarity": 0.87,
        }
        f = extract_features(user, cohort_data=cohort_data)
        assert f["is_cohort_member"] == 1.0
        assert f["cohort_size"] == 42.0
        assert f["cohort_join_spread_hours"] == pytest.approx(3.5)
        assert f["cohort_profile_similarity"] == pytest.approx(0.87)

    def test_cohort_data_not_member(self, make_user):
        user = make_user()
        cohort_data = {
            "is_member": False,
            "size": 10,
            "join_spread_hours": 1.0,
            "profile_similarity": 0.5,
        }
        f = extract_features(user, cohort_data=cohort_data)
        assert f["is_cohort_member"] == 0.0
        # Size and other stats still populated even when not a member.
        assert f["cohort_size"] == 10.0
        assert f["cohort_join_spread_hours"] == pytest.approx(1.0)
        assert f["cohort_profile_similarity"] == pytest.approx(0.5)

    def test_cohort_data_zero_spread(self, make_user):
        # Edge case: cohort with a single-second spread (essentially simultaneous joins).
        user = make_user()
        cohort_data = {
            "is_member": True,
            "size": 200,
            "join_spread_hours": 0.0,
            "profile_similarity": 1.0,
        }
        f = extract_features(user, cohort_data=cohort_data)
        assert f["cohort_join_spread_hours"] == pytest.approx(0.0)
        assert f["cohort_profile_similarity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestFeatureCompleteness
# ---------------------------------------------------------------------------

class TestFeatureCompleteness:
    """Verify the feature dict always has exactly the expected keys and all
    values are numeric (int or float)."""

    EXPECTED_KEYS = {
        # Account flags
        "is_deleted", "is_bot", "is_scam", "is_fake", "is_restricted",
        "is_premium", "has_emoji_status",
        # Profile
        "has_photo", "has_username", "has_last_name",
        "first_name_length", "name_digit_ratio", "script_count",
        # Activity status (one-hot)
        "status_empty", "status_online", "status_recently",
        "status_last_week", "status_last_month", "status_offline",
        "days_since_last_seen",
        # Temporal
        "is_spike_join", "days_since_join", "join_hour_utc", "join_day_of_week",
        # Cohort
        "is_cohort_member", "cohort_size", "cohort_join_spread_hours",
        "cohort_profile_similarity",
        # Heuristic
        "heuristic_score",
    }

    def test_all_keys_present(self, make_user):
        user = make_user()
        f = extract_features(user)
        assert set(f.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_numeric(self, make_user):
        user = make_user()
        f = extract_features(user)
        for key, value in f.items():
            assert isinstance(value, (int, float)), (
                f"Feature '{key}' has non-numeric value {value!r} ({type(value).__name__})"
            )

    def test_feature_keys_constant_matches_dict_keys(self, make_user):
        user = make_user()
        f = extract_features(user)
        # FEATURE_KEYS is a list — check set equality with dict keys.
        assert set(FEATURE_KEYS) == set(f.keys())

    def test_feature_keys_has_no_duplicates(self):
        assert len(FEATURE_KEYS) == len(set(FEATURE_KEYS))

    def test_feature_count_is_29(self, make_user):
        user = make_user()
        f = extract_features(user)
        assert len(f) == 29

    def test_feature_keys_length_is_29(self):
        assert len(FEATURE_KEYS) == 29

    def test_all_keys_present_for_deleted_user(self, deleted_user):
        # Deleted users must still produce a complete feature vector.
        f = extract_features(deleted_user)
        assert set(f.keys()) == self.EXPECTED_KEYS

    def test_all_keys_present_for_premium_user(self, premium_user):
        f = extract_features(premium_user)
        assert set(f.keys()) == self.EXPECTED_KEYS
