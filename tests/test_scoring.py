"""Tests for scoring heuristics."""

import pytest
from datetime import datetime, timezone, timedelta

from tg_purge.scoring import score_user, format_name, status_label, ScoringConfig
from tests.conftest import (
    MockUser, UserStatusEmpty, UserStatusOnline, UserStatusRecently,
    UserStatusLastWeek, UserStatusLastMonth, UserStatusOffline,
)


class TestScoreUser:
    """Test the core scoring function."""

    def test_clean_user_scores_zero(self, clean_user):
        score, reasons = score_user(clean_user)
        assert score == 0
        assert reasons == []

    def test_deleted_user_scores_5(self, deleted_user):
        score, reasons = score_user(deleted_user)
        assert score == 5
        assert any("deleted_account" in r for r in reasons)

    def test_deleted_user_early_return(self, make_user):
        """Deleted accounts skip all other checks."""
        user = make_user(deleted=True, bot=True, scam=True)
        score, reasons = score_user(user)
        assert score == 5
        assert len(reasons) == 1  # Only deleted_account

    def test_scam_flag(self, make_user):
        user = make_user(scam=True)
        score, reasons = score_user(user)
        assert any("scam_flag" in r for r in reasons)

    def test_fake_flag(self, make_user):
        user = make_user(fake=True)
        score, reasons = score_user(user)
        assert any("fake_flag" in r for r in reasons)

    def test_is_bot(self, make_user):
        user = make_user(bot=True)
        score, reasons = score_user(user)
        assert any("is_bot" in r for r in reasons)

    def test_restricted(self, make_user):
        user = make_user(restricted=True)
        score, reasons = score_user(user)
        assert any("restricted" in r for r in reasons)

    def test_no_status_ever(self, make_user):
        user = make_user(status=None)
        score, reasons = score_user(user)
        assert any("no_status_ever" in r for r in reasons)

    def test_empty_status(self, make_user):
        user = make_user(status=UserStatusEmpty())
        score, reasons = score_user(user)
        assert any("no_status_ever" in r for r in reasons)

    def test_offline_over_365_days(self, make_user):
        old_date = datetime.now(timezone.utc) - timedelta(days=400)
        user = make_user(status=UserStatusOffline(was_online=old_date))
        score, reasons = score_user(user)
        assert any("offline_" in r and "+2" in r for r in reasons)

    def test_offline_over_180_days(self, make_user):
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        user = make_user(status=UserStatusOffline(was_online=old_date))
        score, reasons = score_user(user)
        assert any("offline_" in r and "+1" in r for r in reasons)

    def test_offline_recent_no_penalty(self, make_user):
        recent_date = datetime.now(timezone.utc) - timedelta(days=30)
        user = make_user(status=UserStatusOffline(was_online=recent_date))
        score, reasons = score_user(user)
        assert not any("offline_" in r for r in reasons)

    def test_last_month_status(self, make_user):
        user = make_user(status=UserStatusLastMonth())
        score, reasons = score_user(user)
        assert any("last_month" in r for r in reasons)

    def test_online_no_penalty(self, make_user):
        user = make_user(status=UserStatusOnline())
        score, reasons = score_user(user)
        assert not any("no_status" in r or "offline" in r or "last_month" in r for r in reasons)

    def test_recently_no_penalty(self, make_user):
        user = make_user(status=UserStatusRecently())
        score, reasons = score_user(user)
        assert not any("no_status" in r or "offline" in r or "last_month" in r for r in reasons)

    def test_last_week_no_penalty(self, make_user):
        user = make_user(status=UserStatusLastWeek())
        score, reasons = score_user(user)
        assert not any("no_status" in r or "offline" in r or "last_month" in r for r in reasons)

    def test_no_photo(self, make_user):
        user = make_user(photo=False)
        score, reasons = score_user(user)
        assert any("no_photo" in r for r in reasons)

    def test_no_username(self, make_user):
        user = make_user(username=None, last_name="Smith")
        score, reasons = score_user(user)
        assert any("no_username" in r for r in reasons)

    def test_short_name(self, make_user):
        user = make_user(first_name="A")
        score, reasons = score_user(user)
        assert any("short_name" in r for r in reasons)

    def test_empty_name(self, make_user):
        user = make_user(first_name="")
        score, reasons = score_user(user)
        assert any("short_name" in r for r in reasons)

    def test_no_last_no_username(self, make_user):
        user = make_user(last_name=None, username=None)
        score, reasons = score_user(user)
        assert any("no_last+no_user" in r for r in reasons)
        assert any("no_username" in r for r in reasons)

    def test_digit_name(self, make_user):
        user = make_user(first_name="User38291")
        score, reasons = score_user(user)
        assert any("digit_name" in r for r in reasons)

    def test_all_digit_name(self, make_user):
        user = make_user(first_name="12345")
        score, reasons = score_user(user)
        assert any("digit_name" in r for r in reasons)

    def test_low_digit_ratio_no_flag(self, make_user):
        user = make_user(first_name="Alice2")
        score, reasons = score_user(user)
        assert not any("digit_name" in r for r in reasons)

    def test_mixed_scripts_latin_cyrillic(self, make_user):
        user = make_user(first_name="Aliceа")  # Latin + Cyrillic а
        score, reasons = score_user(user)
        assert any("mixed_scripts" in r for r in reasons)

    def test_mixed_scripts_latin_arabic(self, make_user):
        user = make_user(first_name="Aliceم")  # Latin + Arabic
        score, reasons = score_user(user)
        assert any("mixed_scripts" in r for r in reasons)

    def test_single_script_no_flag(self, make_user):
        user = make_user(first_name="Alice")
        score, reasons = score_user(user)
        assert not any("mixed_scripts" in r for r in reasons)

    def test_premium_reduces_score(self, make_user):
        user = make_user(
            premium=True,
            status=None,  # +2
            photo=False,  # +1
        )
        score, reasons = score_user(user)
        # 2 + 1 - 2 = 1
        assert any("premium" in r for r in reasons)
        assert score < 3  # Would be 3 without premium

    def test_emoji_status_reduces_score(self, make_user):
        user = make_user(
            emoji_status="star",
            status=None,  # +2
        )
        score, reasons = score_user(user)
        assert any("emoji_status" in r for r in reasons)

    def test_premium_user_overall(self, premium_user):
        score, reasons = score_user(premium_user)
        assert score == 0
        assert any("premium" in r for r in reasons)

    def test_score_never_negative(self, make_user):
        """Score should be clamped to 0 minimum."""
        user = make_user(
            premium=True,  # -2
            emoji_status="star",  # -1
            status=UserStatusOnline(),
            photo=True,
            username="test",
            first_name="Alice",
            last_name="Smith",
        )
        score, reasons = score_user(user)
        assert score == 0

    def test_compound_bot_signals(self, make_user):
        """A user with many bot signals gets a high score."""
        user = make_user(
            status=None,       # +2
            photo=False,       # +1
            username=None,     # +1
            first_name="A",    # +1 short
            last_name=None,    # +1 no_last+no_user
        )
        score, reasons = score_user(user)
        assert score >= 5


class TestScoringConfig:
    """Test custom scoring configuration."""

    def test_custom_weights(self, make_user):
        config = ScoringConfig(no_photo=3, no_username=3)
        user = make_user(photo=False, username=None, last_name="X")
        score, reasons = score_user(user, config=config)
        assert any("+3" in r for r in reasons)

    def test_zero_weight_disables_signal(self, make_user):
        config = ScoringConfig(no_photo=0)
        user = make_user(photo=False)
        score, reasons = score_user(user, config=config)
        assert not any("no_photo" in r for r in reasons) or any("+0" in r for r in reasons)

    def test_custom_digit_threshold(self, make_user):
        config = ScoringConfig(digit_name_threshold=0.8)
        user = make_user(first_name="User38291")  # ~55% digits
        score, reasons = score_user(user, config=config)
        assert not any("digit_name" in r for r in reasons)


class TestFormatName:
    """Test name formatting."""

    def test_deleted_user(self, deleted_user):
        name = format_name(deleted_user)
        assert "Deleted" in name
        assert str(deleted_user.id) in name

    def test_full_name_with_username(self, make_user):
        user = make_user(first_name="Alice", last_name="Smith", username="asmith")
        name = format_name(user)
        assert "Alice" in name
        assert "Smith" in name
        assert "@asmith" in name

    def test_first_name_only(self, make_user):
        user = make_user(first_name="Alice", last_name=None, username=None)
        name = format_name(user)
        assert name == "Alice"

    def test_empty_name(self, make_user):
        user = make_user(first_name="", last_name="", username=None)
        name = format_name(user)
        assert name == ""


class TestStatusLabel:
    """Test status label formatting."""

    def test_deleted(self, deleted_user):
        assert status_label(deleted_user) == "DELETED"

    def test_never_seen(self, make_user):
        user = make_user(status=None)
        assert status_label(user) == "never seen"

    def test_empty_status(self, make_user):
        user = make_user(status=UserStatusEmpty())
        assert status_label(user) == "never seen"

    def test_online(self, make_user):
        user = make_user(status=UserStatusOnline())
        assert status_label(user) == "online NOW"

    def test_recently(self, make_user):
        user = make_user(status=UserStatusRecently())
        assert status_label(user) == "recently"

    def test_last_week(self, make_user):
        user = make_user(status=UserStatusLastWeek())
        assert status_label(user) == "last week"

    def test_last_month(self, make_user):
        user = make_user(status=UserStatusLastMonth())
        assert status_label(user) == "last month"

    def test_offline_with_days(self, make_user):
        old_date = datetime.now(timezone.utc) - timedelta(days=42)
        user = make_user(status=UserStatusOffline(was_online=old_date))
        label = status_label(user)
        assert "offline" in label
        assert "42" in label
