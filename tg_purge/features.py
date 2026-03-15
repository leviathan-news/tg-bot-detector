"""
ML feature extraction for Telegram subscriber bot detection.

This module is a parallel path to scoring.py's heuristic score_user().
It extracts a numeric feature vector (dict[str, float]) from a Telethon
User object suitable for training or running ML models.

It does NOT replace score_user() — instead it includes the heuristic score
as one feature among many, giving ML models access to both raw signals and
the hand-crafted composite score.

Design constraints (mirrors scoring.py):
- Uses type(status).__name__ string comparison so this module can be
  imported and tested without Telethon installed.
- No numpy/pandas dependency — returns plain Python dicts.
- All values are float (or int coerced to float) for uniform dtype handling.
"""

import re
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any

from .scoring import score_user, ScoringConfig


# ---------------------------------------------------------------------------
# Status type name resolution — mirrors scoring._status_type_name()
# ---------------------------------------------------------------------------

# Mapping from Telethon class name (as string) to a normalized token.
# None status is treated the same as UserStatusEmpty.
_STATUS_TYPE_NAMES: Dict[str, str] = {
    "UserStatusEmpty": "empty",
    "UserStatusOnline": "online",
    "UserStatusRecently": "recently",
    "UserStatusLastWeek": "last_week",
    "UserStatusLastMonth": "last_month",
    "UserStatusOffline": "offline",
}


def _status_type(status) -> str:
    """Resolve a status object to its normalized type name.

    Returns "empty" for None (user has no status info).
    Resolves by inspecting type(status).__name__ so no Telethon import is
    needed — identical pattern to scoring._status_type_name().
    """
    if status is None:
        return "empty"
    return _STATUS_TYPE_NAMES.get(type(status).__name__, "empty")


# ---------------------------------------------------------------------------
# FEATURE_KEYS — stable, ordered list of all feature names.
# Updating this list is the single source of truth for the feature schema.
# ---------------------------------------------------------------------------

FEATURE_KEYS: List[str] = [
    # --- Account flags (binary 0/1) ---
    "is_deleted",         # Account has been deleted by Telegram
    "is_bot",             # Account is registered as a bot
    "is_scam",            # Telegram has flagged as scam
    "is_fake",            # Telegram has flagged as fake/impersonator
    "is_restricted",      # Account is restricted by Telegram
    "is_premium",         # Account has Telegram Premium
    "has_emoji_status",   # Account has a custom emoji status set

    # --- Profile completeness ---
    "has_photo",          # Account has a profile photo
    "has_username",       # Account has a public username (@handle)
    "has_last_name",      # Account has a last name set
    "first_name_length",  # Character count of first_name (0 if absent)
    "name_digit_ratio",   # Fraction of digit characters in first_name [0, 1]
    "script_count",       # Number of distinct script families in first_name
                          # (Latin, Cyrillic, Arabic, CJK — max 4)
    "name_emoji_count",   # Number of emoji codepoints in first_name+last_name
                          # Airdrop farmers average 3-6 emoji; real users 0-1
    "name_has_crypto_kw", # 1 if name contains known crypto/airdrop project keywords
                          # (Meshchain, SEED, Drops, DeSpeed, VIA, PadTON, etc.)
    "name_username_sim",  # Jaccard similarity between name tokens and username tokens
                          # Mismatch bots: "Felipe" with @AnnPerez_720233 → sim≈0

    # --- Activity status (one-hot encoded) ---
    "status_empty",       # Never seen / no status info
    "status_online",      # Currently online
    "status_recently",    # Online recently (within ~1 day)
    "status_last_week",   # Online within the last week
    "status_last_month",  # Online within the last month
    "status_offline",     # Has an explicit last-seen timestamp
    "days_since_last_seen",  # Continuous approximation; -1 if unknown

    # --- Temporal (join date derived) ---
    "is_spike_join",      # 1 if join_date falls inside a spike window
    "days_since_join",    # Days since channel join; -1 if join_date unknown
    "join_hour_utc",      # UTC hour of join (0–23); -1 if unknown
    "join_day_of_week",   # Day-of-week of join (0=Mon … 6=Sun); -1 if unknown

    # --- Cohort (populated from external cohort_data dict) ---
    "is_cohort_member",          # 1 if user is part of a detected cohort
    "cohort_size",               # Number of members in the cohort (0 if none)
    "cohort_join_spread_hours",  # Time spread of cohort joins in hours (0 if none)
    "cohort_profile_similarity", # Profile similarity score within cohort (0 if none)

    # --- Photo metadata (available without downloading the full photo) ---
    "photo_dc_id",        # Data center ID where photo is stored (1-5); 0 if no photo
                          # Validated on 3.5K+ bots vs 3K+ humans:
                          #   DC1: 63.5% bots vs 16.3% humans (bot cluster)
                          #   DC4: 35.7% bots vs 37.2% humans (neutral — NOT a signal)
                          #   DC5:  0.2% bots vs 29.1% humans (strong human indicator)
    "photo_has_video",    # 1 if profile has animated video avatar (0% bots, 4.7% humans)
    "photo_dc_is_1",      # 1 if photo stored on DC 1 (bot-heavy data center)
    "photo_dc_is_5",      # 1 if photo stored on DC 5 (near-exclusively human)

    # --- Photo quality metrics (requires downloaded photo, optional) ---
    "photo_file_size",    # Downloaded photo file size in bytes; 0 if unavailable
    "photo_edge_std",     # Edge density StdDev from gradient analysis; 0 if unavailable
    "photo_lum_variance", # Luminance variance across pixels; 0 if unavailable
    "photo_sat_mean",     # Mean color saturation (HSV); 0 if unavailable

    # --- Extended profile signals ---
    "has_custom_color",            # 1 if user has set a custom name/chat color
    "has_profile_color",           # 1 if user has set a profile page color theme
    "has_stories",                 # 1 if user has posted at least one story
    "stories_unavailable",         # 1 if user's stories are unavailable (100% on bots)
    "has_contact_require_premium", # 1 if user requires premium to contact them
    "usernames_count",             # Number of collectible/additional usernames (0-N)
    "is_verified",                 # 1 if Telegram-verified (strong human signal)
    "has_lang_code",               # 1 if user has a language/locale code set
    "has_paid_messages",           # 1 if user charges Stars for incoming messages
    "is_stars_subscriber",         # 1 if user has a Stars subscription to the channel

    # --- Heuristic aggregate ---
    "heuristic_score",    # Raw output of score_user() with default ScoringConfig
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_account_flags(user) -> Dict[str, float]:
    """Extract binary account-level flags set by Telegram or the user.

    Returns a dict with keys: is_deleted, is_bot, is_scam, is_fake,
    is_restricted, is_premium, has_emoji_status. All values 0.0 or 1.0.
    """
    return {
        "is_deleted":       float(bool(getattr(user, "deleted", False))),
        "is_bot":           float(bool(getattr(user, "bot", False))),
        "is_scam":          float(bool(getattr(user, "scam", False))),
        "is_fake":          float(bool(getattr(user, "fake", False))),
        "is_restricted":    float(bool(getattr(user, "restricted", False))),
        "is_premium":       float(bool(getattr(user, "premium", False))),
        "has_emoji_status": float(bool(getattr(user, "emoji_status", None))),
    }


def _count_emoji(text: str) -> int:
    """Count emoji codepoints in a string.

    Uses Unicode emoji ranges to detect emoji characters. Counts individual
    codepoints, not grapheme clusters (so a flag emoji = 2, skin tone = 2).
    This is intentional — airdrop farmers pack many emoji codepoints.
    """
    count = 0
    for ch in text:
        cp = ord(ch)
        # Common emoji ranges:
        # - Emoticons: U+1F600-U+1F64F
        # - Misc Symbols: U+1F300-U+1F5FF
        # - Transport/Map: U+1F680-U+1F6FF
        # - Supplemental: U+1F900-U+1F9FF
        # - Symbols/Pictographs: U+1FA00-U+1FAFF
        # - Dingbats: U+2700-U+27BF
        # - Misc Symbols: U+2600-U+26FF
        # - Variation selectors and ZWJ are not counted (they're modifiers).
        if (0x1F300 <= cp <= 0x1F9FF or
                0x1FA00 <= cp <= 0x1FAFF or
                0x2600 <= cp <= 0x27BF or
                0x2300 <= cp <= 0x23FF or
                0xFE00 <= cp <= 0xFE0F or
                0x200D == cp):
            count += 1
    return count


# Known crypto/airdrop project keywords found in bot farm names.
# Lowercase for case-insensitive matching. Ordered by frequency
# observed in review sessions.
_CRYPTO_KEYWORDS = {
    "meshchain", "seed", "drops", "despeed", "via", "padton",
    "sparkchain", "yescoiner", "bits", "airdrop", "dnnl",
    "seraph", "crypto", "defi", "nft", "web3", "blockchain",
    "token", "mining", "staking", "swap", "yield", "dao",
    "metaverse", "gamefi",
}


def _has_crypto_keywords(text: str) -> bool:
    """Check if text contains known crypto/airdrop project keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _CRYPTO_KEYWORDS)


def _name_username_similarity(first_name: str, last_name: str,
                              username: str) -> float:
    """Compute fraction of name tokens found as substrings in the username.

    Extracts lowercase alpha tokens (2+ chars) from the display name
    (first + last), then checks how many appear as substrings in the
    lowered username. Returns matched_count / total_name_tokens.

    This handles concatenated usernames like "alicesmith" where regex
    tokenization would yield a single token that doesn't match
    {"alice", "smith"}.

    Bot farms often generate usernames from a different name pool than
    the display name, yielding similarity ≈ 0. Real users typically have
    username matching or derived from their real name.

    Returns 0.0 if either name or username is empty (no signal).
    """
    if not username or not first_name:
        return 0.0

    # Tokenize name: split on whitespace and non-alpha chars.
    name_text = f"{first_name} {last_name or ''}".strip().lower()
    name_tokens = set(re.findall(r"[a-z]{2,}", name_text))

    if not name_tokens:
        return 0.0

    # Check how many name tokens appear as substrings in the username.
    # Handles concatenated usernames like "alicesmith" → matches "alice" + "smith".
    username_lower = username.lower()
    matched = sum(1 for tok in name_tokens if tok in username_lower)
    return matched / len(name_tokens)


def _extract_profile_features(user) -> Dict[str, float]:
    """Extract profile completeness and name analysis features.

    Analyses the first_name string for length, digit ratio, script families,
    emoji count, crypto keywords, and name/username similarity.

    Returns keys: has_photo, has_username, has_last_name, first_name_length,
    name_digit_ratio, script_count, name_emoji_count, name_has_crypto_kw,
    name_username_sim.
    """
    first: str = user.first_name or ""
    last: str = getattr(user, "last_name", None) or ""
    username: str = getattr(user, "username", None) or ""

    # --- Digit ratio ---
    # Guard against ZeroDivisionError on empty first name.
    if first:
        digit_ratio = sum(c.isdigit() for c in first) / len(first)
    else:
        digit_ratio = 0.0

    # --- Script count ---
    # Count how many distinct writing systems appear in the first name.
    # Each system is detected via a regex on the Unicode block range.
    has_latin    = bool(re.search(r"[a-zA-Z]", first))
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", first))
    has_arabic   = bool(re.search(r"[\u0600-\u06FF]", first))
    # CJK Unified Ideographs + CJK Extension A, plus CJK Compatibility Ideographs.
    has_cjk      = bool(re.search(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]", first))
    script_count = sum([has_latin, has_cyrillic, has_arabic, has_cjk])

    # --- Emoji count across full display name ---
    # Airdrop farmers typically have 3-6 emoji in their name; real users 0-1.
    full_name = f"{first} {last}".strip()
    emoji_count = _count_emoji(full_name)

    # --- Crypto/airdrop keyword detection ---
    has_crypto = _has_crypto_keywords(full_name)

    # --- Name/username similarity ---
    # Bot farms generate display names and usernames from different pools.
    # Real users' usernames typically relate to their display name.
    sim = _name_username_similarity(first, last, username)

    return {
        "has_photo":         float(bool(getattr(user, "photo", None))),
        "has_username":      float(bool(getattr(user, "username", None))),
        "has_last_name":     float(bool(last)),
        "first_name_length": float(len(first)),
        "name_digit_ratio":  digit_ratio,
        "script_count":      float(script_count),
        "name_emoji_count":  float(emoji_count),
        "name_has_crypto_kw": float(has_crypto),
        "name_username_sim": sim,
    }


def _extract_activity_features(user) -> Dict[str, float]:
    """Extract activity/status features as a one-hot vector plus a
    continuous days_since_last_seen approximation.

    The one-hot keys are: status_empty, status_online, status_recently,
    status_last_week, status_last_month, status_offline.

    days_since_last_seen approximation:
        empty / None  → -1  (unknown)
        online        →  0
        recently      →  1  (Telegram shows this for ~1 day)
        last_week     →  7
        last_month    → 30
        offline       → actual elapsed days from status.was_online
    """
    status = getattr(user, "status", None)
    status_key = _status_type(status)

    # Build the one-hot dict: all zeros, then set the active bucket to 1.
    one_hot = {
        "status_empty":      0.0,
        "status_online":     0.0,
        "status_recently":   0.0,
        "status_last_week":  0.0,
        "status_last_month": 0.0,
        "status_offline":    0.0,
    }
    # Map the normalized status key to its one-hot column name.
    one_hot_key = f"status_{status_key}"
    if one_hot_key in one_hot:
        one_hot[one_hot_key] = 1.0

    # Continuous approximation of recency.
    if status_key == "empty":
        days = -1.0
    elif status_key == "online":
        days = 0.0
    elif status_key == "recently":
        days = 1.0
    elif status_key == "last_week":
        days = 7.0
    elif status_key == "last_month":
        days = 30.0
    elif status_key == "offline":
        # Compute actual elapsed days from the stored timestamp.
        now = datetime.now(timezone.utc)
        days = float((now - status.was_online).days)
    else:
        days = -1.0

    one_hot["days_since_last_seen"] = days
    return one_hot


def _extract_temporal_features(
    join_date: Optional[datetime],
    spike_windows: Optional[List[Tuple[datetime, datetime]]],
) -> Dict[str, float]:
    """Extract temporal features derived from the channel join date.

    Args:
        join_date: UTC datetime when the user joined the channel, or None.
        spike_windows: List of (start, end) datetime tuples representing
            bulk-subscription spike windows (half-open intervals: start <= t < end).

    Returns keys: is_spike_join, days_since_join, join_hour_utc,
    join_day_of_week. All default to -1.0 (or 0.0 for the binary flag) if
    join_date is None.
    """
    if join_date is None:
        return {
            "is_spike_join":    0.0,
            "days_since_join":  -1.0,
            "join_hour_utc":    -1.0,
            "join_day_of_week": -1.0,
        }

    # Check whether join_date falls inside any spike window (half-open: [start, end)).
    is_spike = 0.0
    if spike_windows:
        for window_start, window_end in spike_windows:
            if window_start <= join_date < window_end:
                is_spike = 1.0
                break  # Only flag once even if windows overlap.

    now = datetime.now(timezone.utc)
    days_since = float((now - join_date).days)

    return {
        "is_spike_join":    is_spike,
        "days_since_join":  days_since,
        "join_hour_utc":    float(join_date.hour),
        "join_day_of_week": float(join_date.weekday()),  # 0=Monday … 6=Sunday
    }


def _extract_cohort_features(
    cohort_data: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    """Extract cohort membership features from pre-computed cohort_data.

    cohort_data is expected to be a dict with keys:
        is_member (bool), size (int), join_spread_hours (float),
        profile_similarity (float).

    All values default to 0.0 when cohort_data is None.
    """
    if cohort_data is None:
        return {
            "is_cohort_member":          0.0,
            "cohort_size":               0.0,
            "cohort_join_spread_hours":  0.0,
            "cohort_profile_similarity": 0.0,
        }

    return {
        "is_cohort_member":          float(bool(cohort_data.get("is_member", False))),
        "cohort_size":               float(cohort_data.get("size", 0)),
        "cohort_join_spread_hours":  float(cohort_data.get("join_spread_hours", 0.0)),
        "cohort_profile_similarity": float(cohort_data.get("profile_similarity", 0.0)),
    }


def _extract_extended_profile(user, participant_data=None) -> Dict[str, float]:
    """Extract extended profile signals available from the User TL object.

    These fields indicate profile customization effort — real users are more
    likely to customize colors, post stories, or set up collectible usernames.
    Bot farm accounts almost never touch these settings.

    Args:
        user: Telethon User or MockUser object.
        participant_data: Optional dict with ChannelParticipant metadata:
            {subscription_until_date: datetime or None}. Passed separately
            because participant data comes from the participants vector,
            not the users vector.

    Fields:
      - color: Custom name/chat color (PeerColor object, layer 160+)
      - profile_color: Custom profile page background color (layer 166+)
      - stories_max_id: Non-zero if user has posted at least one story
      - stories_unavailable: Stories are blocked/unavailable (100% on bots)
      - contact_require_premium: Privacy setting requiring premium to message
      - usernames: List of additional/collectible usernames beyond the main one
      - verified: Telegram-verified account (strong negative signal)
      - lang_code: User's locale code (set by client during auth)
      - send_paid_messages_stars: User charges Stars for incoming messages
      - subscription_until_date: Stars subscription to the channel (from participant)
    """
    # Custom name/chat color — requires effort to set, very rare in bot accounts.
    has_color = float(bool(getattr(user, "color", None)))

    # Profile page color theme — another customization signal.
    has_profile_color = float(bool(getattr(user, "profile_color", None)))

    # Stories — indicates active usage beyond passive channel subscription.
    # stories_max_id > 0 means the user has posted at least one story.
    stories_max_id = getattr(user, "stories_max_id", None) or 0
    has_stories = float(stories_max_id > 0)

    # Stories unavailable — True when user hasn't posted stories or has them
    # blocked. 100% True on bot accounts in our testing. Distinct from
    # has_stories because stories_unavailable can be True even if the user
    # once had stories but deleted/hid them.
    stories_unavail = float(bool(getattr(user, "stories_unavailable", False)))

    # Contact require premium — privacy setting that bots never use.
    has_contact_require = float(bool(getattr(user, "contact_require_premium", False)))

    # Collectible/additional usernames — e.g., Fragment-purchased usernames.
    # Bot farms don't invest in these.
    usernames = getattr(user, "usernames", None) or []
    usernames_count = float(len(usernames))

    # Verified — Telegram grants this to official/trusted accounts. Definitively
    # not a bot subscriber. Very rare.
    is_verified = float(bool(getattr(user, "verified", False)))

    # Language code — set by the Telegram client during auth. Bot farms often
    # register via specific locales or have no lang_code at all.
    has_lang = float(bool(getattr(user, "lang_code", None)))

    # Paid messages — user charges Stars for incoming messages. Strong human
    # signal since bots have no reason to enable this.
    paid_stars = getattr(user, "send_paid_messages_stars", None)
    has_paid = float(bool(paid_stars and paid_stars > 0))

    # Stars subscription — from ChannelParticipant data. If the user is paying
    # a Stars subscription to this channel, they are definitively human.
    is_subscriber = 0.0
    if participant_data:
        sub_date = participant_data.get("subscription_until_date")
        is_subscriber = float(bool(sub_date))

    return {
        "has_custom_color":            has_color,
        "has_profile_color":           has_profile_color,
        "has_stories":                 has_stories,
        "stories_unavailable":         stories_unavail,
        "has_contact_require_premium": has_contact_require,
        "usernames_count":             usernames_count,
        "is_verified":                 is_verified,
        "has_lang_code":               has_lang,
        "has_paid_messages":           has_paid,
        "is_stars_subscriber":         is_subscriber,
    }


def _extract_photo_features(
    user,
    photo_quality: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract profile photo metadata features.

    Two tiers of features:
    1. Metadata (free, no API call): dc_id, has_video, dc_is_1, dc_is_5
       Extracted from user.photo (UserProfilePhoto) attributes.
    2. Quality metrics (requires downloaded photo): file_size, edge_std,
       lum_variance, sat_mean. Passed in via photo_quality dict.

    Args:
        user: Telethon User or MockUser. user.photo may be:
            - None (no photo)
            - A UserProfilePhoto-like object with dc_id, has_video attrs
            - A string placeholder (legacy MockUser compat — treated as
              photo-present but no metadata available)
        photo_quality: Optional dict with keys matching the photo quality
            feature names (photo_file_size, photo_edge_std, etc.).
            Typically populated by a separate download+analysis pass.
    """
    photo = getattr(user, "photo", None)

    # Determine if we have a real photo object with metadata attributes,
    # or just a placeholder indicating photo presence.
    dc_id = 0
    has_video = 0.0

    if photo is not None:
        # Try to read metadata attributes — works for real UserProfilePhoto
        # objects and MockProfilePhoto. Falls back to 0 for string placeholders.
        dc_id = getattr(photo, "dc_id", 0) or 0
        has_video = float(bool(getattr(photo, "has_video", False)))

    features = {
        "photo_dc_id":      float(dc_id),
        "photo_has_video":  has_video,
        "photo_dc_is_1":    float(dc_id == 1),
        "photo_dc_is_5":    float(dc_id == 5),
    }

    # Quality metrics — populated from external analysis pass if available.
    if photo_quality:
        features["photo_file_size"]    = float(photo_quality.get("photo_file_size", 0))
        features["photo_edge_std"]     = float(photo_quality.get("photo_edge_std", 0))
        features["photo_lum_variance"] = float(photo_quality.get("photo_lum_variance", 0))
        features["photo_sat_mean"]     = float(photo_quality.get("photo_sat_mean", 0))
    else:
        features["photo_file_size"]    = 0.0
        features["photo_edge_std"]     = 0.0
        features["photo_lum_variance"] = 0.0
        features["photo_sat_mean"]     = 0.0

    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    user,
    join_date: Optional[datetime] = None,
    spike_windows: Optional[List[Tuple[datetime, datetime]]] = None,
    cohort_data: Optional[Dict[str, Any]] = None,
    photo_quality: Optional[Dict[str, float]] = None,
    participant_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Extract a numeric feature vector from a Telethon User object.

    This is a parallel path to score_user() — NOT a replacement. It exposes
    each individual signal as a separate feature plus includes the heuristic
    composite score as the final feature, giving downstream ML models access
    to both granular signals and the hand-crafted aggregate.

    Args:
        user: A Telethon User object (or MockUser for tests). Must expose the
            same attribute interface: id, deleted, bot, scam, fake, restricted,
            status, photo, username, first_name, last_name, premium,
            emoji_status.
        join_date: Optional UTC datetime when the user joined the channel.
            When provided, enables temporal features (days_since_join,
            join_hour_utc, join_day_of_week) and spike detection.
        spike_windows: Optional list of (start_dt, end_dt) half-open datetime
            intervals identifying bulk-subscription spikes. Requires join_date
            to have any effect.
        cohort_data: Optional dict with cohort membership data:
            {is_member: bool, size: int, join_spread_hours: float,
             profile_similarity: float}. Pass None when no cohort analysis
            has been run.
        photo_quality: Optional dict with downloaded photo quality metrics:
            {photo_file_size: float, photo_edge_std: float,
             photo_lum_variance: float, photo_sat_mean: float}.
            Pass None when photos have not been downloaded/analyzed.
        participant_data: Optional dict with ChannelParticipant metadata:
            {subscription_until_date: datetime or None}. Provides Stars
            subscription status for the channel.

    Returns:
        dict[str, float] with exactly the keys listed in FEATURE_KEYS.
        All values are Python float (or int that passes isinstance(v, float)
        for numpy-compatibility — callers should cast if needed).
    """
    # --- Account flags ---
    features: Dict[str, float] = {}
    features.update(_extract_account_flags(user))

    # --- Profile completeness ---
    features.update(_extract_profile_features(user))

    # --- Activity status (one-hot + continuous) ---
    features.update(_extract_activity_features(user))

    # --- Temporal (join date derived) ---
    features.update(_extract_temporal_features(join_date, spike_windows))

    # --- Cohort ---
    features.update(_extract_cohort_features(cohort_data))

    # --- Photo metadata + quality ---
    features.update(_extract_photo_features(user, photo_quality))

    # --- Extended profile signals ---
    features.update(_extract_extended_profile(user, participant_data))

    # --- Heuristic score (call canonical scorer with default config) ---
    heuristic, _reasons = score_user(
        user,
        config=ScoringConfig(),
        join_date=join_date,
        spike_windows=spike_windows,
    )
    features["heuristic_score"] = float(heuristic)

    return features
