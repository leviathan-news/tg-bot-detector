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
    MockProfilePhoto,
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
# TestPhotoFeatures
# ---------------------------------------------------------------------------

class TestPhotoFeatures:
    """Photo metadata and quality features: photo_dc_id, photo_has_video,
    photo_dc_is_1, photo_dc_is_5, photo_file_size, photo_edge_std,
    photo_lum_variance, photo_sat_mean."""

    # --- Metadata features (free, from user.photo object) ---

    def test_no_photo_all_zeros(self, make_user):
        """User with no photo should have all photo features at 0."""
        user = make_user(photo=False)
        f = extract_features(user)
        assert f["photo_dc_id"] == 0.0
        assert f["photo_has_video"] == 0.0
        assert f["photo_dc_is_1"] == 0.0
        assert f["photo_dc_is_5"] == 0.0

    def test_photo_dc_id_extracted(self, make_user):
        """DC ID should be extracted from photo metadata."""
        user = make_user(photo=True, photo_dc_id=4)
        f = extract_features(user)
        assert f["photo_dc_id"] == 4.0

    def test_photo_dc_1_flag(self, make_user):
        """DC 1 flag should be set when photo is on DC 1 (bot-heavy)."""
        user = make_user(photo=True, photo_dc_id=1)
        f = extract_features(user)
        assert f["photo_dc_is_1"] == 1.0
        assert f["photo_dc_is_5"] == 0.0

    def test_photo_dc_5_flag(self, make_user):
        """DC 5 flag should be set when photo is on DC 5 (zero bots observed)."""
        user = make_user(photo=True, photo_dc_id=5)
        f = extract_features(user)
        assert f["photo_dc_is_5"] == 1.0
        assert f["photo_dc_is_1"] == 0.0

    def test_photo_dc_other_no_flags(self, make_user):
        """DC 2/3/4 should have neither DC 1 nor DC 5 flag set."""
        user = make_user(photo=True, photo_dc_id=2)
        f = extract_features(user)
        assert f["photo_dc_is_1"] == 0.0
        assert f["photo_dc_is_5"] == 0.0

    def test_photo_has_video_true(self, make_user):
        """Animated video avatar should set has_video flag."""
        user = make_user(photo=True, photo_dc_id=2, photo_has_video=True)
        f = extract_features(user)
        assert f["photo_has_video"] == 1.0

    def test_photo_has_video_false(self, make_user):
        """Non-animated photo should have has_video = 0."""
        user = make_user(photo=True, photo_dc_id=2, photo_has_video=False)
        f = extract_features(user)
        assert f["photo_has_video"] == 0.0

    def test_photo_placeholder_string_no_metadata(self, make_user):
        """Legacy string placeholder (no dc_id attr) should gracefully default."""
        user = make_user(photo=True)  # No photo_dc_id → string placeholder
        f = extract_features(user)
        assert f["photo_dc_id"] == 0.0
        assert f["photo_has_video"] == 0.0

    # --- Quality features (from downloaded photo analysis) ---

    def test_photo_quality_defaults_to_zero(self, make_user):
        """Without photo_quality data, all quality features should be 0."""
        user = make_user(photo=True, photo_dc_id=2)
        f = extract_features(user)
        assert f["photo_file_size"] == 0.0
        assert f["photo_edge_std"] == 0.0
        assert f["photo_lum_variance"] == 0.0
        assert f["photo_sat_mean"] == 0.0

    def test_photo_quality_populated(self, make_user):
        """Photo quality metrics should be passed through from photo_quality dict."""
        user = make_user(photo=True, photo_dc_id=2)
        quality = {
            "photo_file_size": 63000.0,
            "photo_edge_std": 7.38,
            "photo_lum_variance": 3534.0,
            "photo_sat_mean": 0.303,
        }
        f = extract_features(user, photo_quality=quality)
        assert f["photo_file_size"] == pytest.approx(63000.0)
        assert f["photo_edge_std"] == pytest.approx(7.38)
        assert f["photo_lum_variance"] == pytest.approx(3534.0)
        assert f["photo_sat_mean"] == pytest.approx(0.303)

    def test_photo_quality_partial(self, make_user):
        """Missing keys in photo_quality should default to 0."""
        user = make_user(photo=True, photo_dc_id=2)
        quality = {"photo_file_size": 50000.0}
        f = extract_features(user, photo_quality=quality)
        assert f["photo_file_size"] == pytest.approx(50000.0)
        assert f["photo_edge_std"] == 0.0


# ---------------------------------------------------------------------------
# TestExtendedProfileFeatures
# ---------------------------------------------------------------------------

class TestExtendedProfileFeatures:
    """Extended profile signals: has_custom_color, has_profile_color,
    has_stories, has_contact_require_premium, usernames_count."""

    def test_defaults_all_zero(self, make_user):
        """Default MockUser has no extended profile customization."""
        user = make_user()
        f = extract_features(user)
        assert f["has_custom_color"] == 0.0
        assert f["has_profile_color"] == 0.0
        assert f["has_stories"] == 0.0
        assert f["has_contact_require_premium"] == 0.0
        assert f["usernames_count"] == 0.0

    def test_custom_color_set(self, make_user):
        """User with custom name/chat color should flag has_custom_color."""
        user = make_user(color="some_peer_color_object")
        f = extract_features(user)
        assert f["has_custom_color"] == 1.0

    def test_profile_color_set(self, make_user):
        """User with profile page color should flag has_profile_color."""
        user = make_user(profile_color="some_profile_color_object")
        f = extract_features(user)
        assert f["has_profile_color"] == 1.0

    def test_has_stories_from_max_id(self, make_user):
        """User with stories_max_id > 0 should flag has_stories."""
        user = make_user(stories_max_id=42)
        f = extract_features(user)
        assert f["has_stories"] == 1.0

    def test_no_stories_max_id_zero(self, make_user):
        """User with stories_max_id = 0 should not flag has_stories."""
        user = make_user(stories_max_id=0)
        f = extract_features(user)
        assert f["has_stories"] == 0.0

    def test_contact_require_premium_true(self, make_user):
        """User requiring premium to contact should flag."""
        user = make_user(contact_require_premium=True)
        f = extract_features(user)
        assert f["has_contact_require_premium"] == 1.0

    def test_usernames_count(self, make_user):
        """Multiple collectible usernames should be counted."""
        # Simulating Telethon Username objects as simple dicts
        user = make_user(usernames=[{"username": "primary"}, {"username": "extra"}])
        f = extract_features(user)
        assert f["usernames_count"] == 2.0

    def test_usernames_none(self, make_user):
        """No usernames list should yield 0."""
        user = make_user(usernames=None)
        f = extract_features(user)
        assert f["usernames_count"] == 0.0

    def test_stories_unavailable_true(self, make_user):
        """stories_unavailable flag should be extracted."""
        user = make_user(stories_unavailable=True)
        f = extract_features(user)
        assert f["stories_unavailable"] == 1.0

    def test_stories_unavailable_false(self, make_user):
        user = make_user(stories_unavailable=False)
        f = extract_features(user)
        assert f["stories_unavailable"] == 0.0

    def test_verified_true(self, make_user):
        """Telegram-verified accounts should flag is_verified."""
        user = make_user(verified=True)
        f = extract_features(user)
        assert f["is_verified"] == 1.0

    def test_verified_false(self, make_user):
        user = make_user(verified=False)
        f = extract_features(user)
        assert f["is_verified"] == 0.0

    def test_has_lang_code(self, make_user):
        """User with lang_code set should flag has_lang_code."""
        user = make_user(lang_code="en")
        f = extract_features(user)
        assert f["has_lang_code"] == 1.0

    def test_no_lang_code(self, make_user):
        user = make_user(lang_code=None)
        f = extract_features(user)
        assert f["has_lang_code"] == 0.0

    def test_paid_messages_set(self, make_user):
        """User charging Stars for messages should flag has_paid_messages."""
        user = make_user(send_paid_messages_stars=100)
        f = extract_features(user)
        assert f["has_paid_messages"] == 1.0

    def test_paid_messages_not_set(self, make_user):
        user = make_user(send_paid_messages_stars=None)
        f = extract_features(user)
        assert f["has_paid_messages"] == 0.0

    def test_stars_subscriber_from_participant_data(self, make_user):
        """Stars subscriber flag from ChannelParticipant metadata."""
        from datetime import datetime, timezone
        user = make_user()
        participant = {"subscription_until_date": datetime(2026, 12, 31, tzinfo=timezone.utc)}
        f = extract_features(user, participant_data=participant)
        assert f["is_stars_subscriber"] == 1.0

    def test_no_stars_subscriber(self, make_user):
        """No participant data means no Stars subscription."""
        user = make_user()
        f = extract_features(user)
        assert f["is_stars_subscriber"] == 0.0


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
# TestNameAnalysisFeatures
# ---------------------------------------------------------------------------

class TestNameAnalysisFeatures:
    """Tests for name_emoji_count, name_has_crypto_kw, name_username_sim."""

    def test_no_emoji_count_zero(self, make_user):
        """Normal name without emoji should have count 0."""
        user = make_user(first_name="Alice", last_name="Smith")
        f = extract_features(user)
        assert f["name_emoji_count"] == 0.0

    def test_emoji_counted_in_name(self, make_user):
        """Emoji in first_name should be counted."""
        user = make_user(first_name="VIA Drops\U0001f4a7 SEED\U0001f331")
        f = extract_features(user)
        assert f["name_emoji_count"] >= 2.0

    def test_emoji_counted_in_last_name(self, make_user):
        """Emoji in last_name are also counted."""
        user = make_user(first_name="Dasha", last_name="Dasha \U0001f408\u200d\u2b1b")
        f = extract_features(user)
        assert f["name_emoji_count"] >= 1.0

    def test_no_crypto_keyword_default(self, make_user):
        """Normal name without crypto keywords should be 0."""
        user = make_user(first_name="Bob", last_name="Jones")
        f = extract_features(user)
        assert f["name_has_crypto_kw"] == 0.0

    def test_crypto_keyword_meshchain(self, make_user):
        """Name containing 'Meshchain' triggers crypto keyword flag."""
        user = make_user(first_name="VIA Seraph Meshchain.Ai")
        f = extract_features(user)
        assert f["name_has_crypto_kw"] == 1.0

    def test_crypto_keyword_seed(self, make_user):
        """Name containing 'SEED' triggers crypto keyword flag."""
        user = make_user(first_name="Yori SEED Yescoiner")
        f = extract_features(user)
        assert f["name_has_crypto_kw"] == 1.0

    def test_crypto_keyword_case_insensitive(self, make_user):
        """Crypto keyword detection is case-insensitive."""
        user = make_user(first_name="john DeSpEeD fan")
        f = extract_features(user)
        assert f["name_has_crypto_kw"] == 1.0

    def test_crypto_keyword_airdrop(self, make_user):
        """'airdrop' in name triggers the flag."""
        user = make_user(first_name="Free Airdrop Hunter")
        f = extract_features(user)
        assert f["name_has_crypto_kw"] == 1.0

    def test_username_sim_matching_name(self, make_user):
        """Username derived from real name should have high similarity."""
        user = make_user(first_name="Alice", last_name="Smith", username="alicesmith")
        f = extract_features(user)
        assert f["name_username_sim"] > 0.5

    def test_username_sim_mismatched(self, make_user):
        """Mismatched name and username (bot pattern) should have low similarity."""
        user = make_user(first_name="Felipe", username="AnnPerez_720233")
        f = extract_features(user)
        assert f["name_username_sim"] < 0.2

    def test_username_sim_no_username(self, make_user):
        """No username should return 0 similarity (no signal)."""
        user = make_user(first_name="Test", username=None)
        f = extract_features(user)
        assert f["name_username_sim"] == 0.0

    def test_username_sim_no_name(self, make_user):
        """Empty first name should return 0 similarity."""
        user = make_user(first_name="", username="testuser")
        f = extract_features(user)
        assert f["name_username_sim"] == 0.0

    def test_username_sim_partial_match(self, make_user):
        """Partial overlap should produce intermediate similarity."""
        user = make_user(first_name="John", last_name="Doe", username="johncrypto")
        f = extract_features(user)
        # "john" matches, "doe" doesn't match "crypto" → partial sim
        assert 0.0 < f["name_username_sim"] < 1.0


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
        "name_emoji_count", "name_has_crypto_kw", "name_username_sim",
        # Activity status (one-hot)
        "status_empty", "status_online", "status_recently",
        "status_last_week", "status_last_month", "status_offline",
        "days_since_last_seen",
        # Temporal
        "is_spike_join", "days_since_join", "join_hour_utc", "join_day_of_week",
        # Cohort
        "is_cohort_member", "cohort_size", "cohort_join_spread_hours",
        "cohort_profile_similarity",
        # Photo metadata
        "photo_dc_id", "photo_has_video", "photo_dc_is_1", "photo_dc_is_5",
        # Photo quality
        "photo_file_size", "photo_edge_std", "photo_lum_variance", "photo_sat_mean",
        # Extended profile
        "has_custom_color", "has_profile_color", "has_stories",
        "stories_unavailable", "has_contact_require_premium", "usernames_count",
        "is_verified", "has_lang_code", "has_paid_messages", "is_stars_subscriber",
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

    def test_feature_count_is_50(self, make_user):
        user = make_user()
        f = extract_features(user)
        assert len(f) == 50

    def test_feature_keys_length_is_50(self):
        assert len(FEATURE_KEYS) == 50

    def test_all_keys_present_for_deleted_user(self, deleted_user):
        # Deleted users must still produce a complete feature vector.
        f = extract_features(deleted_user)
        assert set(f.keys()) == self.EXPECTED_KEYS

    def test_all_keys_present_for_premium_user(self, premium_user):
        f = extract_features(premium_user)
        assert set(f.keys()) == self.EXPECTED_KEYS
