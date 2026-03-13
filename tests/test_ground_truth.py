"""
Tests for scripts/train_ground_truth.py feature conversion and dataset assembly.

Validates that raw_to_feature_vector() produces valid 47-feature vectors
matching the FEATURE_KEYS schema, and that build_training_set() correctly
merges data from multiple sources with proper priority handling.
"""

import pytest
import sys
from pathlib import Path

# Add project root so the script module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_ground_truth import raw_to_feature_vector, build_training_set
from tg_purge.features import FEATURE_KEYS


class TestRawToFeatureVector:
    """Tests for converting raw validation fields to ML feature vectors."""

    def test_all_feature_keys_present(self):
        """Converted vector must contain exactly the keys in FEATURE_KEYS."""
        raw = {
            "user_id": 12345,
            "first_name": "TestUser",
            "has_photo": True,
            "has_username": True,
            "has_last_name": True,
            "is_premium": False,
            "has_emoji_status": False,
            "is_deleted": False,
            "is_bot_api": False,
            "is_scam": False,
            "is_fake": False,
            "is_restricted": False,
            "status_type": "UserStatusRecently",
            "photo_dc_id": 2,
            "photo_has_video": False,
            "has_custom_color": False,
            "has_profile_color": False,
            "has_stories": False,
            "stories_unavailable": True,
            "has_contact_require_premium": False,
            "usernames_count": 0,
            "is_verified": False,
            "has_lang_code": True,
            "lang_code": "en",
            "has_paid_messages": False,
            "heuristic_score": 3,
        }
        result = raw_to_feature_vector(raw)
        assert set(result.keys()) == set(FEATURE_KEYS)

    def test_all_values_are_float(self):
        """Every value in the feature vector must be a float."""
        raw = {
            "user_id": 1,
            "first_name": "Test",
            "has_photo": True,
            "has_username": True,
            "has_last_name": False,
            "is_premium": True,
            "has_emoji_status": True,
            "is_deleted": False,
            "is_bot_api": False,
            "is_scam": False,
            "is_fake": False,
            "is_restricted": False,
            "status_type": "UserStatusOnline",
            "photo_dc_id": 1,
            "photo_has_video": True,
            "has_custom_color": True,
            "has_profile_color": True,
            "has_stories": True,
            "stories_unavailable": False,
            "has_contact_require_premium": True,
            "usernames_count": 3,
            "is_verified": True,
            "has_lang_code": True,
            "lang_code": "ru",
            "has_paid_messages": True,
            "heuristic_score": 0,
        }
        result = raw_to_feature_vector(raw)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, not float"

    def test_status_one_hot_recently(self):
        """UserStatusRecently maps to status_recently=1, all others=0."""
        raw = self._minimal_raw(status_type="UserStatusRecently")
        result = raw_to_feature_vector(raw)
        assert result["status_recently"] == 1.0
        assert result["status_empty"] == 0.0
        assert result["status_online"] == 0.0
        assert result["status_offline"] == 0.0
        assert result["days_since_last_seen"] == 1.0

    def test_status_one_hot_empty(self):
        """None status maps to status_empty=1."""
        raw = self._minimal_raw(status_type="None")
        result = raw_to_feature_vector(raw)
        assert result["status_empty"] == 1.0
        assert result["days_since_last_seen"] == -1.0

    def test_status_one_hot_online(self):
        """UserStatusOnline maps correctly."""
        raw = self._minimal_raw(status_type="UserStatusOnline")
        result = raw_to_feature_vector(raw)
        assert result["status_online"] == 1.0
        assert result["days_since_last_seen"] == 0.0

    def test_dc_is_1_flag(self):
        """DC ID 1 sets photo_dc_is_1=1."""
        raw = self._minimal_raw(photo_dc_id=1)
        result = raw_to_feature_vector(raw)
        assert result["photo_dc_is_1"] == 1.0
        assert result["photo_dc_is_5"] == 0.0
        assert result["photo_dc_id"] == 1.0

    def test_dc_is_5_flag(self):
        """DC ID 5 sets photo_dc_is_5=1."""
        raw = self._minimal_raw(photo_dc_id=5)
        result = raw_to_feature_vector(raw)
        assert result["photo_dc_is_5"] == 1.0
        assert result["photo_dc_is_1"] == 0.0

    def test_name_analysis_digits(self):
        """First name with digits computes correct digit ratio."""
        raw = self._minimal_raw(first_name="abc123")
        result = raw_to_feature_vector(raw)
        assert result["first_name_length"] == 6.0
        assert result["name_digit_ratio"] == pytest.approx(0.5)

    def test_name_analysis_mixed_scripts(self):
        """Mixed Latin + Cyrillic triggers script_count=2."""
        raw = self._minimal_raw(first_name="Helloмир")
        result = raw_to_feature_vector(raw)
        assert result["script_count"] == 2.0

    def test_empty_first_name(self):
        """Empty first name doesn't cause division by zero."""
        raw = self._minimal_raw(first_name="")
        result = raw_to_feature_vector(raw)
        assert result["first_name_length"] == 0.0
        assert result["name_digit_ratio"] == 0.0
        assert result["script_count"] == 0.0

    def test_temporal_defaults(self):
        """Temporal features default to -1 (unavailable from validation data)."""
        raw = self._minimal_raw()
        result = raw_to_feature_vector(raw)
        assert result["days_since_join"] == -1.0
        assert result["join_hour_utc"] == -1.0
        assert result["join_day_of_week"] == -1.0
        assert result["is_spike_join"] == 0.0

    def test_cohort_defaults(self):
        """Cohort features default to 0."""
        raw = self._minimal_raw()
        result = raw_to_feature_vector(raw)
        assert result["is_cohort_member"] == 0.0
        assert result["cohort_size"] == 0.0

    def test_photo_quality_defaults(self):
        """Photo quality features default to 0 (not downloaded)."""
        raw = self._minimal_raw()
        result = raw_to_feature_vector(raw)
        assert result["photo_file_size"] == 0.0
        assert result["photo_edge_std"] == 0.0
        assert result["photo_lum_variance"] == 0.0
        assert result["photo_sat_mean"] == 0.0

    def test_boolean_flags_mapping(self):
        """Boolean flags from raw data map correctly to float 0/1."""
        raw = self._minimal_raw(
            is_premium=True,
            has_emoji_status=True,
            is_bot_api=True,
            is_scam=True,
            has_custom_color=True,
            is_verified=True,
        )
        result = raw_to_feature_vector(raw)
        assert result["is_premium"] == 1.0
        assert result["has_emoji_status"] == 1.0
        assert result["is_bot"] == 1.0
        assert result["is_scam"] == 1.0
        assert result["has_custom_color"] == 1.0
        assert result["is_verified"] == 1.0

    def test_heuristic_score_passthrough(self):
        """Heuristic score passes through from raw data."""
        raw = self._minimal_raw(heuristic_score=7)
        result = raw_to_feature_vector(raw)
        assert result["heuristic_score"] == 7.0

    def _minimal_raw(self, **overrides):
        """Create a minimal raw dict with sensible defaults.

        Accepts keyword overrides for any field. Returns a dict
        with all required fields for raw_to_feature_vector().
        """
        base = {
            "user_id": 1,
            "first_name": "Test",
            "has_photo": True,
            "has_username": True,
            "has_last_name": True,
            "is_premium": False,
            "has_emoji_status": False,
            "is_deleted": False,
            "is_bot_api": False,
            "is_scam": False,
            "is_fake": False,
            "is_restricted": False,
            "status_type": "UserStatusRecently",
            "photo_dc_id": 2,
            "photo_has_video": False,
            "has_custom_color": False,
            "has_profile_color": False,
            "has_stories": False,
            "stories_unavailable": False,
            "has_contact_require_premium": False,
            "usernames_count": 0,
            "is_verified": False,
            "has_lang_code": False,
            "lang_code": "",
            "has_paid_messages": False,
            "heuristic_score": 0,
        }
        base.update(overrides)
        return base


class TestBuildTrainingSet:
    """Tests for assembling training data from multiple sources."""

    def test_basic_assembly(self):
        """Bot and human data are assembled with correct labels."""
        bot_data = [
            {"user_id": 1, "first_name": "Bot1", "has_photo": False,
             "has_username": False, "has_last_name": False, "is_premium": False,
             "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
             "is_scam": False, "is_fake": False, "is_restricted": False,
             "status_type": "None", "photo_dc_id": 0, "photo_has_video": False,
             "has_custom_color": False, "has_profile_color": False,
             "has_stories": False, "stories_unavailable": True,
             "has_contact_require_premium": False, "usernames_count": 0,
             "is_verified": False, "has_lang_code": False, "lang_code": "",
             "has_paid_messages": False, "heuristic_score": 5},
        ]
        human_data = [
            {"user_id": 2, "first_name": "Human1", "has_photo": True,
             "has_username": True, "has_last_name": True, "is_premium": True,
             "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
             "is_scam": False, "is_fake": False, "is_restricted": False,
             "status_type": "UserStatusRecently", "photo_dc_id": 5,
             "photo_has_video": True, "has_custom_color": True,
             "has_profile_color": False, "has_stories": True,
             "stories_unavailable": False, "has_contact_require_premium": False,
             "usernames_count": 1, "is_verified": False, "has_lang_code": True,
             "lang_code": "en", "has_paid_messages": False, "heuristic_score": 0},
        ]

        features, labels, stats = build_training_set(
            bot_data, human_data, {}, {},
        )

        assert len(features) == 2
        assert labels == ["bot", "human"]
        assert stats["converted_bot"] == 1
        assert stats["converted_human"] == 1

    def test_skips_deleted_accounts(self):
        """Deleted accounts are skipped (no useful features)."""
        bot_data = [
            {"user_id": 1, "first_name": "", "is_deleted": True,
             "has_photo": False, "has_username": False, "has_last_name": False,
             "is_premium": False, "has_emoji_status": False, "is_bot_api": False,
             "is_scam": False, "is_fake": False, "is_restricted": False,
             "status_type": "None", "photo_dc_id": 0, "photo_has_video": False,
             "has_custom_color": False, "has_profile_color": False,
             "has_stories": False, "stories_unavailable": False,
             "has_contact_require_premium": False, "usernames_count": 0,
             "is_verified": False, "has_lang_code": False, "lang_code": "",
             "has_paid_messages": False, "heuristic_score": 5},
        ]

        features, labels, stats = build_training_set(bot_data, [], {}, {})

        assert len(features) == 0
        assert stats["skipped_deleted"] == 1

    def test_deduplicates_users(self):
        """Same user_id appearing in both bot and human data is only counted once."""
        shared = {"user_id": 1, "first_name": "Dup", "has_photo": True,
                  "has_username": True, "has_last_name": False, "is_premium": False,
                  "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
                  "is_scam": False, "is_fake": False, "is_restricted": False,
                  "status_type": "UserStatusRecently", "photo_dc_id": 2,
                  "photo_has_video": False, "has_custom_color": False,
                  "has_profile_color": False, "has_stories": False,
                  "stories_unavailable": False, "has_contact_require_premium": False,
                  "usernames_count": 0, "is_verified": False, "has_lang_code": False,
                  "lang_code": "", "has_paid_messages": False, "heuristic_score": 2}

        features, labels, stats = build_training_set(
            [shared], [shared], {}, {},
        )

        # Should appear only once (from bot_data, processed first).
        assert len(features) == 1
        assert labels == ["bot"]

    def test_prefers_cached_features(self):
        """When a user exists in feature cache, cached vector is used."""
        raw = {"user_id": 42, "first_name": "Cached", "has_photo": True,
               "has_username": True, "has_last_name": True, "is_premium": False,
               "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
               "is_scam": False, "is_fake": False, "is_restricted": False,
               "status_type": "UserStatusRecently", "photo_dc_id": 2,
               "photo_has_video": False, "has_custom_color": False,
               "has_profile_color": False, "has_stories": False,
               "stories_unavailable": False, "has_contact_require_premium": False,
               "usernames_count": 0, "is_verified": False, "has_lang_code": False,
               "lang_code": "", "has_paid_messages": False, "heuristic_score": 3}

        # Cached vector has a distinctive value we can check.
        # Use heuristic_score as marker — it's not a leaked feature.
        cached_vec = raw_to_feature_vector(raw)
        cached_vec["heuristic_score"] = 999.0  # Marker value.

        features, labels, stats = build_training_set(
            [], [raw], {"42": cached_vec}, {},
        )

        assert len(features) == 1
        # Should use cached vector with the marker value.
        assert features[0]["heuristic_score"] == 999.0
        assert stats["cached_human"] == 1
        assert stats["converted_human"] == 0

    def test_neutralizes_leaked_features(self):
        """Leaked features are set to defaults regardless of source."""
        raw = {"user_id": 1, "first_name": "Test", "has_photo": True,
               "has_username": True, "has_last_name": True, "is_premium": False,
               "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
               "is_scam": False, "is_fake": False, "is_restricted": False,
               "status_type": "UserStatusRecently", "photo_dc_id": 2,
               "photo_has_video": False, "has_custom_color": False,
               "has_profile_color": False, "has_stories": False,
               "stories_unavailable": False, "has_contact_require_premium": False,
               "usernames_count": 0, "is_verified": False, "has_lang_code": False,
               "lang_code": "", "has_paid_messages": False, "heuristic_score": 0}

        # Cached vector with real temporal values (would cause leakage).
        cached_vec = raw_to_feature_vector(raw)
        cached_vec["days_since_join"] = 42.0
        cached_vec["join_hour_utc"] = 14.0

        features, labels, stats = build_training_set(
            [], [raw], {"1": cached_vec}, {},
        )

        # Leaked features should be neutralized to defaults.
        assert features[0]["days_since_join"] == -1.0
        assert features[0]["join_hour_utc"] == -1.0
        assert features[0]["days_since_last_seen"] == -1.0

    def test_human_review_override(self):
        """Human-reviewed label overrides ground-truth source label."""
        bot_data = [
            {"user_id": 1, "first_name": "NotABot", "has_photo": True,
             "has_username": True, "has_last_name": True, "is_premium": False,
             "has_emoji_status": False, "is_deleted": False, "is_bot_api": False,
             "is_scam": False, "is_fake": False, "is_restricted": False,
             "status_type": "UserStatusRecently", "photo_dc_id": 2,
             "photo_has_video": False, "has_custom_color": False,
             "has_profile_color": False, "has_stories": False,
             "stories_unavailable": False, "has_contact_require_premium": False,
             "usernames_count": 0, "is_verified": False, "has_lang_code": False,
             "lang_code": "", "has_paid_messages": False, "heuristic_score": 0},
        ]

        # Human review says this user is actually human, not a bot.
        human_reviewed = {"1": "human"}

        features, labels, stats = build_training_set(
            bot_data, [], {}, human_reviewed,
        )

        assert labels == ["human"]
        assert stats["human_reviewed_override"] == 1
