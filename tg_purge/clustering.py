"""
Auto-detection of bulk-subscription spikes from join-date timestamps.

Uses a sliding-window approach: slide a fixed-width window across the
timeline, count joins per window position, flag positions where the
count exceeds mean + N*sigma. Merge overlapping flagged windows into
contiguous spike regions.

The detected windows are compatible with score_user()'s spike_windows
parameter -- drop-in integration.
"""

from datetime import timedelta, timezone
from math import sqrt


def merge_windows(windows):
    """Merge overlapping or adjacent (start, end) datetime tuples.

    Takes a list of potentially overlapping time windows and collapses
    them into non-overlapping contiguous regions. This is done by sorting
    windows by start time, then iterating and extending the last merged
    window whenever the next window's start falls within (or at) the
    previous window's end.

    Args:
        windows: List of (start_dt, end_dt) tuples, not necessarily sorted.

    Returns:
        Sorted list of merged (start_dt, end_dt) tuples with no overlaps.
    """
    if not windows:
        return []

    # Sort by start datetime so we can merge in a single left-to-right pass
    sorted_w = sorted(windows, key=lambda w: w[0])
    merged = [sorted_w[0]]

    for start, end in sorted_w[1:]:
        prev_start, prev_end = merged[-1]
        # If the current window overlaps or is adjacent to the previous,
        # extend the previous window's end to cover both
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # No overlap -- start a new merged region
            merged.append((start, end))

    return merged


def detect_spike_windows(
    join_dates,
    window_size=timedelta(hours=1),
    step_size=timedelta(minutes=15),
    sigma_multiplier=2.0,
    min_cluster_size=5,
):
    """Detect bulk-subscription spikes from join-date timestamps.

    Slides a fixed-width window across the timeline of join dates, counts
    how many joins fall into each window position, then flags any window
    where the join count exceeds mean + sigma_multiplier * stddev AND
    meets the min_cluster_size threshold. Flagged windows are merged into
    contiguous spike regions.

    Algorithm:
        1. Sort all join timestamps.
        2. Slide a window of width `window_size` from earliest to latest
           timestamp, advancing by `step_size` each iteration.
        3. For each window position, count how many joins fall within
           [window_start, window_end).
        4. Compute mean and stddev of all window counts.
        5. Flag windows where count > mean + sigma_multiplier * stddev
           AND count >= min_cluster_size.
        6. Merge overlapping flagged windows into contiguous regions.

    Args:
        join_dates: Dict of user_id -> datetime (timezone-aware).
            Maps user identifiers to their join timestamps.
        window_size: Width of each sliding window (default: 1 hour).
            Controls the granularity of spike detection.
        step_size: Step between window positions (default: 15 minutes).
            Smaller steps give finer resolution but more computation.
        sigma_multiplier: Number of standard deviations above mean to
            flag as a spike (default: 2.0). Lower values detect more
            spikes; higher values are stricter.
        min_cluster_size: Minimum joins in a window to be considered a
            spike, regardless of statistics (default: 5). Prevents
            flagging low-count statistical anomalies.

    Returns:
        List of (start_dt, end_dt) tuples representing detected spike
        regions. Compatible with score_user()'s spike_windows parameter.
        Returns empty list if fewer than 10 join dates are provided
        (insufficient data for meaningful statistical analysis).
    """
    # Guard: need at least 10 data points for any meaningful statistical
    # analysis; fewer would produce unreliable mean/stddev values
    if len(join_dates) < 10:
        return []

    # Sort timestamps to enable efficient sliding-window counting
    timestamps = sorted(join_dates.values())
    earliest = timestamps[0]
    latest = timestamps[-1]

    # Slide the window across the full timeline, counting joins per position.
    # For each position [current_start, current_end), we count timestamps
    # that fall within that half-open interval.
    window_counts = []
    current_start = earliest

    while current_start + window_size <= latest + step_size:
        current_end = current_start + window_size
        # Count timestamps in [current_start, current_end) using linear scan.
        # Since timestamps are sorted, we break early when past current_end.
        count = 0
        for ts in timestamps:
            if ts >= current_end:
                break
            if ts >= current_start:
                count += 1
        window_counts.append((current_start, current_end, count))
        current_start += step_size

    if not window_counts:
        return []

    # Compute population mean and standard deviation of window counts.
    # Using population stats (not sample) because we have the full set
    # of window positions, not a sample.
    counts = [c for _, _, c in window_counts]
    n = len(counts)
    mean = sum(counts) / n
    variance = sum((c - mean) ** 2 for c in counts) / n
    stddev = sqrt(variance)

    # Handle the zero-stddev edge case separately.
    # When stddev is zero, all windows have identical counts, so the normal
    # "count > mean + N*sigma" test can never flag anything. Two sub-cases:
    #   a) Uniform distribution over a long span (e.g., 5 joins/hour for 24h):
    #      The data spans many window widths -- this is NOT a spike.
    #   b) All joins in one tight burst (e.g., 50 joins in 50 minutes):
    #      The entire dataset fits within ~1 window width -- this IS a spike.
    # We distinguish them by comparing the data's time span to window_size.
    # If the span is <= 2x window_size, the concentration is suspicious.
    if stddev == 0:
        time_span = latest - earliest
        is_concentrated_burst = (
            time_span <= 2 * window_size and mean >= min_cluster_size
        )
        if is_concentrated_burst:
            # All data is in one tight burst -- flag all windows
            spike_windows = [(start, end) for start, end, _ in window_counts]
            return merge_windows(spike_windows)
        else:
            # Uniform distribution across a long span -- no spikes
            return []

    # Normal case: stddev > 0, use mean + N*sigma threshold
    threshold = mean + sigma_multiplier * stddev

    # Flag windows that exceed both the statistical threshold AND the
    # absolute minimum cluster size. The dual condition prevents:
    # - Statistical false positives on low-count data (min_cluster_size)
    # - Flagging normal variation in high-count data (threshold)
    spike_windows = []
    for start, end, count in window_counts:
        if count > threshold and count >= min_cluster_size:
            spike_windows.append((start, end))

    # Merge overlapping flagged windows into contiguous spike regions
    # so callers get clean, non-overlapping time ranges
    return merge_windows(spike_windows)
