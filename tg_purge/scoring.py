"""
Heuristic scoring for Telegram broadcast channel subscribers.

Scores are probabilistic signals, not definitive bot classifications.
Each signal adds or subtracts from a cumulative score. Higher scores
indicate greater likelihood of being a bot or inactive/fake account.

The scoring system was developed through empirical analysis of
@leviathan_news channel subscribers (20K+ accounts) and validated
against a database of known active contributors.

Canonical source — reconciled from multiple analysis scripts.
See docs/scoring-methodology.md for the full breakdown.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ScoringConfig:
    """Configurable weights for bot scoring heuristics.

    Defaults represent the canonical weights validated against
    real channel data. Override for experimentation only.
    """
    # Telegram flags
    deleted_account: int = 5
    scam_flag: int = 5
    fake_flag: int = 5
    is_bot: int = 3
    restricted: int = 2

    # Activity status
    no_status_ever: int = 2
    offline_over_365d: int = 2
    offline_over_180d: int = 1
    last_month: int = 1

    # Profile completeness
    no_photo: int = 1
    no_username: int = 1
    short_name: int = 1
    no_last_no_username: int = 1

    # Name patterns
    digit_name: int = 1
    digit_name_threshold: float = 0.3
    mixed_scripts: int = 1

    # Join date clustering
    spike_join: int = 2

    # Positive signals (subtracted)
    premium: int = -2
    emoji_status: int = -1


# Sentinel for status type checks without importing Telethon at module level.
# The actual isinstance checks use string comparison on type names so this
# module can be tested without Telethon installed.
_STATUS_TYPE_NAMES = {
    "UserStatusEmpty": "empty",
    "UserStatusOnline": "online",
    "UserStatusRecently": "recently",
    "UserStatusLastWeek": "last_week",
    "UserStatusLastMonth": "last_month",
    "UserStatusOffline": "offline",
}


def _status_type_name(status):
    """Get a normalized status type name without requiring Telethon imports."""
    if status is None:
        return "empty"
    return _STATUS_TYPE_NAMES.get(type(status).__name__, "unknown")


def score_user(user, config=None, join_date=None, spike_windows=None):
    """Score a Telethon User object for bot likelihood.

    Args:
        user: A Telethon User object (or any object with compatible attributes).
        config: Optional ScoringConfig to override default weights.
        join_date: Optional datetime of when the user joined the channel.
            Used with spike_windows to detect bulk-subscription events.
        spike_windows: Optional list of (start_dt, end_dt) tuples defining
            time windows identified as bulk-subscription spikes. If join_date
            falls within any window, the spike_join penalty is applied.

    Returns:
        Tuple of (score: int, reasons: list[str]) where score >= 0.
        Higher scores indicate greater bot likelihood.
    """
    if config is None:
        config = ScoringConfig()

    score = 0
    reasons = []

    # Deleted account — instant high score, skip other checks
    if user.deleted:
        score += config.deleted_account
        reasons.append(f"deleted_account(+{config.deleted_account})")
        return score, reasons

    # Telegram-applied flags
    if getattr(user, "scam", False):
        score += config.scam_flag
        reasons.append(f"scam_flag(+{config.scam_flag})")
    if getattr(user, "fake", False):
        score += config.fake_flag
        reasons.append(f"fake_flag(+{config.fake_flag})")
    if user.bot:
        score += config.is_bot
        reasons.append(f"is_bot(+{config.is_bot})")
    if getattr(user, "restricted", False):
        score += config.restricted
        reasons.append(f"restricted(+{config.restricted})")

    # Activity status
    status = user.status
    status_type = _status_type_name(status)

    if status_type == "empty":
        score += config.no_status_ever
        reasons.append(f"no_status_ever(+{config.no_status_ever})")
    elif status_type == "offline":
        now = datetime.now(timezone.utc)
        days_offline = (now - status.was_online).days
        if days_offline > 365:
            score += config.offline_over_365d
            reasons.append(f"offline_{days_offline}d(+{config.offline_over_365d})")
        elif days_offline > 180:
            score += config.offline_over_180d
            reasons.append(f"offline_{days_offline}d(+{config.offline_over_180d})")
    elif status_type == "last_month":
        score += config.last_month
        reasons.append(f"last_month(+{config.last_month})")
    # recently, online, last_week — no penalty

    # Profile completeness
    if not getattr(user, "photo", None):
        score += config.no_photo
        reasons.append(f"no_photo(+{config.no_photo})")
    if not user.username:
        score += config.no_username
        reasons.append(f"no_username(+{config.no_username})")

    first = user.first_name or ""
    last = user.last_name or ""

    if len(first) <= 1:
        score += config.short_name
        reasons.append(f"short_name(+{config.short_name})")
    if not last and not user.username:
        score += config.no_last_no_username
        reasons.append(f"no_last+no_user(+{config.no_last_no_username})")

    # Name pattern analysis
    if first:
        digit_ratio = sum(c.isdigit() for c in first) / len(first)
        if digit_ratio > config.digit_name_threshold:
            score += config.digit_name
            reasons.append(f"digit_name({digit_ratio:.0%})(+{config.digit_name})")

        has_latin = bool(re.search(r"[a-zA-Z]", first))
        has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", first))
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", first))
        script_count = sum([has_latin, has_cyrillic, has_arabic])
        if script_count > 1:
            score += config.mixed_scripts
            reasons.append(f"mixed_scripts(+{config.mixed_scripts})")

    # Join date clustering: penalize users who joined during detected spike windows.
    # Both join_date and spike_windows must be provided for this check.
    if join_date is not None and spike_windows:
        for window_start, window_end in spike_windows:
            if window_start <= join_date < window_end:
                score += config.spike_join
                reasons.append(f"spike_join(+{config.spike_join})")
                break  # Only penalize once even if windows overlap

    # Positive signals (reduce score)
    if getattr(user, "premium", False):
        score += config.premium  # negative value
        reasons.append(f"premium({config.premium})")
    if getattr(user, "emoji_status", None):
        score += config.emoji_status  # negative value
        reasons.append(f"emoji_status({config.emoji_status})")

    return max(score, 0), reasons


def format_name(user):
    """Format a Telethon User's display name."""
    if user.deleted:
        return f"[Deleted Account #{user.id}]"
    first = user.first_name or ""
    last = user.last_name or ""
    username = f" (@{user.username})" if user.username else ""
    return f"{first} {last}{username}".strip()


def status_label(user):
    """Return a human-readable label for a user's online status."""
    if user.deleted:
        return "DELETED"
    status = user.status
    status_type = _status_type_name(status)

    if status_type == "empty":
        return "never seen"
    if status_type == "online":
        return "online NOW"
    if status_type == "recently":
        return "recently"
    if status_type == "last_week":
        return "last week"
    if status_type == "last_month":
        return "last month"
    if status_type == "offline":
        days = (datetime.now(timezone.utc) - status.was_online).days
        return f"offline {days}d"
    return str(type(status).__name__)
