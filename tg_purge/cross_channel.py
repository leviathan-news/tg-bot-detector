"""Cross-channel cohort detection for coordinated bot activity.

Identifies groups of users who appear together across multiple Telegram channels
and evaluates whether their join-time patterns and profile features suggest
coordinated, automated subscription behaviour.

Key concepts:
  - A "cohort" is a set of users sharing membership in the same N+ channels.
  - "Coordinated" means: large cohort, tight join-time clustering, and uniform
    profile features — all three together signal bulk bot subscription runs.
"""

import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_cohorts(
    channel_users: dict,
    min_cohort_size: int = 50,
    min_shared_channels: int = 3,
) -> list:
    """Find groups of users who appear together in multiple channels.

    The algorithm groups users by their exact channel-membership fingerprint
    (the frozenset of channels they belong to) and returns only the groups
    large enough to be worth investigating.

    Args:
        channel_users: Mapping of channel_name -> set of user_ids.
        min_cohort_size: Minimum number of users required for a group to be
            returned.  Smaller groups are discarded as noise.
        min_shared_channels: Minimum number of shared channels a user must
            appear in to be considered part of a cohort.  Also acts as a
            short-circuit: if the total number of channels in the input is
            below this threshold, we return immediately because the condition
            can never be satisfied.

    Returns:
        List of dicts, each with:
            "user_ids"       – set of user IDs in this cohort
            "shared_channels" – list of channel names they all share
        Sorted by cohort size descending.
    """
    # Short-circuit: we can never satisfy min_shared_channels with too few channels.
    if len(channel_users) < min_shared_channels:
        return []

    # Step 1: Build the reverse mapping: user_id -> set of channels they appear in.
    user_channels: dict = defaultdict(set)
    for channel_name, user_ids in channel_users.items():
        for uid in user_ids:
            user_channels[uid].add(channel_name)

    # Step 2: Keep only users who appear in at least min_shared_channels channels.
    qualifying_users = {
        uid: channels
        for uid, channels in user_channels.items()
        if len(channels) >= min_shared_channels
    }

    if not qualifying_users:
        return []

    # Step 3: Group users by their exact channel-set fingerprint (frozenset).
    # Users with the same fingerprint were co-subscribed to exactly the same
    # set of channels — the most precise signal of coordinated activity.
    fingerprint_to_users: dict = defaultdict(set)
    for uid, channels in qualifying_users.items():
        fingerprint = frozenset(channels)
        fingerprint_to_users[fingerprint].add(uid)

    # Step 4: Filter by size and build result list.
    cohorts = []
    for channel_set, user_ids in fingerprint_to_users.items():
        if len(user_ids) >= min_cohort_size:
            cohorts.append({
                "user_ids": user_ids,
                "shared_channels": sorted(channel_set),  # deterministic order
            })

    # Return largest cohorts first for intuitive iteration.
    cohorts.sort(key=lambda c: len(c["user_ids"]), reverse=True)
    return cohorts


def score_cohort(
    user_ids: list,
    shared_channels: list,
    join_times: dict,
    profiles: dict,
) -> dict:
    """Score a cohort for suspiciousness based on join timing and profile uniformity.

    A cohort is considered suspicious when ALL of these conditions hold:
      - size >= 50
      - join_spread_hours < 48  (users all joined within a 48-hour window)
      - profile_similarity > 0.6 (most profile features are identical across members)
      - len(shared_channels) >= 3

    Args:
        user_ids: List of user IDs in this cohort.
        shared_channels: List of channel names the cohort shares.
        join_times: Nested dict: user_id -> {channel_name -> datetime}.
            Datetime objects should be timezone-aware or consistently naive.
        profiles: Dict of user_id -> {feature_name -> value}.
            Feature values can be any hashable type.

    Returns:
        Dict with keys:
            "is_suspicious"      – bool
            "join_spread_hours"  – float, stddev of all join timestamps in hours
            "profile_similarity" – float in [0, 1], fraction of features that
                                   are uniform (>80% same value) across members
            "cohort_size"        – int
            "confidence"         – "high" | "medium" | "low" | "none"
    """
    cohort_size = len(user_ids)

    # --- Join-time spread -------------------------------------------------------
    # Collect every timestamp for every (user, channel) pair in the cohort.
    # We measure spread as the stddev of offsets from the earliest timestamp,
    # expressed in hours.  A tight cluster → small spread → likely automated.
    all_timestamps = []
    for uid in user_ids:
        user_jtimes = join_times.get(uid, {})
        for ch in shared_channels:
            ts = user_jtimes.get(ch)
            if ts is not None:
                all_timestamps.append(ts)

    join_spread_hours = _compute_spread_hours(all_timestamps)

    # --- Profile similarity -----------------------------------------------------
    # For each feature key, determine whether >80% of members share the same
    # value.  The similarity score is the proportion of such "uniform" features.
    profile_similarity = _compute_profile_similarity(user_ids, profiles)

    # --- Suspiciousness gate ----------------------------------------------------
    # Every condition must hold simultaneously; any single miss clears the flag.
    is_suspicious = (
        cohort_size >= 50
        and join_spread_hours < 48
        and profile_similarity > 0.6
        and len(shared_channels) >= 3
    )

    # --- Confidence tier --------------------------------------------------------
    # Derived solely from the two continuous metrics when the cohort is suspicious.
    # The tighter the timing and the more uniform the profiles, the higher the
    # confidence that this is a coordinated bot run rather than an organic spike.
    if not is_suspicious:
        confidence = "none"
    elif join_spread_hours < 6 and profile_similarity > 0.8:
        confidence = "high"
    elif join_spread_hours < 24:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_suspicious": is_suspicious,
        "join_spread_hours": join_spread_hours,
        "profile_similarity": profile_similarity,
        "cohort_size": cohort_size,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_spread_hours(timestamps: list) -> float:
    """Compute the standard deviation of a list of datetimes, in hours.

    Args:
        timestamps: List of datetime objects (all should share the same
            timezone awareness — mixing aware and naive will raise ValueError).

    Returns:
        Standard deviation in hours.  Returns 0.0 for 0 or 1 timestamps
        (stddev is undefined for a single point, treated as zero spread).
    """
    if len(timestamps) < 2:
        return 0.0

    # Convert all datetimes to float seconds since the earliest timestamp.
    # Using the earliest as the reference keeps the numbers small and avoids
    # large floating-point magnitudes when working with Unix epochs.
    earliest = min(timestamps)
    offsets_seconds = [
        (ts - earliest).total_seconds() for ts in timestamps
    ]

    # Population stddev — we have the full cohort, not a sample.
    stddev_seconds = statistics.pstdev(offsets_seconds)
    return stddev_seconds / 3600.0


def _compute_profile_similarity(user_ids: list, profiles: dict) -> float:
    """Measure how uniform profile features are across cohort members.

    For each feature key that appears in at least one member's profile:
      - Count the most-common value across all members.
      - If that value appears in > 80% of members, the feature is "uniform".

    Similarity = number_of_uniform_features / total_distinct_features.

    Args:
        user_ids: List of user IDs to consider.
        profiles: Dict of user_id -> {feature_name -> value}.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when profiles is empty or contains
        no features, rather than raising a ZeroDivisionError.
    """
    if not profiles:
        return 0.0

    # Gather feature values across all members who have a profile entry.
    feature_values: dict = defaultdict(list)
    for uid in user_ids:
        member_profile = profiles.get(uid, {})
        for key, value in member_profile.items():
            feature_values[key].append(value)

    total_features = len(feature_values)
    if total_features == 0:
        return 0.0

    member_count = len(user_ids)
    threshold = 0.8  # >80% of members must share the same value

    uniform_count = 0
    for key, values in feature_values.items():
        if not values:
            continue
        # Find the most common value for this feature.
        most_common_value, most_common_count = Counter(values).most_common(1)[0]
        # Compare against all cohort members (not just those with an entry).
        if most_common_count / member_count > threshold:
            uniform_count += 1

    return uniform_count / total_features
