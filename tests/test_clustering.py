"""Tests for join-date spike auto-detection."""

from datetime import datetime, timedelta, timezone

from tg_purge.clustering import detect_spike_windows, merge_windows


def _make_dates(base, counts_per_hour):
    """Helper: generate join dates. counts_per_hour is a list of ints,
    one per hour starting from base. Returns dict of fake_uid -> datetime."""
    dates = {}
    uid = 1000
    for hour_offset, count in enumerate(counts_per_hour):
        for minute in range(count):
            dt = base + timedelta(hours=hour_offset, minutes=minute)
            dates[uid] = dt
            uid += 1
    return dates


class TestDetectSpikeWindows:

    def test_empty_returns_empty(self):
        assert detect_spike_windows({}) == []

    def test_insufficient_data_returns_empty(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = {i: base + timedelta(hours=i) for i in range(9)}
        assert detect_spike_windows(dates) == []

    def test_uniform_distribution_no_spikes(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = _make_dates(base, [5] * 24)
        assert detect_spike_windows(dates) == []

    def test_single_obvious_spike(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [5] * 24 + [50] + [5] * 24
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) >= 1
        spike_time = base + timedelta(hours=24)
        assert any(start <= spike_time < end for start, end in windows)

    def test_two_separate_spikes(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [3] * 72
        counts[10] = 40
        counts[50] = 40
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) == 2

    def test_adjacent_spike_hours_merge(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [3] * 48
        counts[20] = 40
        counts[21] = 40
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) == 1
        start, end = windows[0]
        assert end - start >= timedelta(hours=1)

    def test_min_cluster_size_filters_small_spikes(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [1] * 48
        counts[10] = 3
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates, min_cluster_size=5)
        assert len(windows) == 0

    def test_all_joins_in_one_burst_stddev_zero(self):
        base = datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc)
        dates = {i: base + timedelta(minutes=i) for i in range(50)}
        windows = detect_spike_windows(dates)
        assert len(windows) >= 1

    def test_configurable_window_size(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [2] * 48
        counts[20] = 30
        counts[21] = 30
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates, window_size=timedelta(hours=2))
        assert len(windows) >= 1

    def test_configurable_sigma_multiplier(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [5] * 48
        counts[10] = 20
        dates = _make_dates(base, counts)
        windows_loose = detect_spike_windows(dates, sigma_multiplier=1.0)
        windows_strict = detect_spike_windows(dates, sigma_multiplier=5.0)
        assert len(windows_loose) >= len(windows_strict)


class TestMergeWindows:

    def test_empty_input(self):
        assert merge_windows([]) == []

    def test_single_window(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w = [(base, base + timedelta(hours=1))]
        assert merge_windows(w) == w

    def test_non_overlapping_stay_separate(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w1 = (base, base + timedelta(hours=1))
        w2 = (base + timedelta(hours=3), base + timedelta(hours=4))
        result = merge_windows([w2, w1])
        assert result == [w1, w2]

    def test_overlapping_merge(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w1 = (base, base + timedelta(hours=2))
        w2 = (base + timedelta(hours=1), base + timedelta(hours=3))
        result = merge_windows([w1, w2])
        assert result == [(base, base + timedelta(hours=3))]

    def test_adjacent_merge(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w1 = (base, base + timedelta(hours=1))
        w2 = (base + timedelta(hours=1), base + timedelta(hours=2))
        result = merge_windows([w1, w2])
        assert result == [(base, base + timedelta(hours=2))]

    def test_three_overlapping_merge_to_one(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w1 = (base, base + timedelta(hours=2))
        w2 = (base + timedelta(hours=1), base + timedelta(hours=3))
        w3 = (base + timedelta(hours=2, minutes=30), base + timedelta(hours=4))
        result = merge_windows([w3, w1, w2])
        assert result == [(base, base + timedelta(hours=4))]
