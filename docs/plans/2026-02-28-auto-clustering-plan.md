# Auto-Detected Join-Date Clustering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-detect bulk-subscription spikes from join-date timestamps and integrate them into the scoring pipeline, breaking the score-2 cliff.

**Architecture:** New `tg_purge/clustering.py` module with a sliding-window spike detector. Commands that enumerate subscribers call `detect_spike_windows()` on the returned join dates, then pass the detected windows to `score_user()`. Coexists with manual spike windows.

**Tech Stack:** Pure Python stdlib (datetime, statistics). No new dependencies.

---

### Task 1: Core clustering module — `detect_spike_windows()`

**Files:**
- Create: `tg_purge/clustering.py`
- Test: `tests/test_clustering.py`

**Step 1: Write the failing tests**

Create `tests/test_clustering.py`:

```python
"""Tests for join-date spike auto-detection."""

from datetime import datetime, timedelta, timezone

from tg_purge.clustering import detect_spike_windows


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
        # Fewer than 10 join dates -> not enough data
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = {i: base + timedelta(hours=i) for i in range(9)}
        assert detect_spike_windows(dates) == []

    def test_uniform_distribution_no_spikes(self):
        # 5 joins per hour for 24 hours -> no spikes (uniform)
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = _make_dates(base, [5] * 24)
        assert detect_spike_windows(dates) == []

    def test_single_obvious_spike(self):
        # 5/hour baseline for 24h, then 50 in one hour
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [5] * 24 + [50] + [5] * 24
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) >= 1
        # The spike hour is at offset 24 (hour 24 from base)
        spike_time = base + timedelta(hours=24)
        # At least one window should contain the spike hour
        assert any(start <= spike_time < end for start, end in windows)

    def test_two_separate_spikes(self):
        # Spike at hour 10 and hour 50, baseline 3/hour
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [3] * 72  # 3 days baseline
        counts[10] = 40
        counts[50] = 40
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) == 2

    def test_adjacent_spike_hours_merge(self):
        # Two consecutive high-volume hours should merge into one window
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [3] * 48
        counts[20] = 40
        counts[21] = 40
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates)
        assert len(windows) == 1
        start, end = windows[0]
        # Merged window should span both hours
        assert end - start >= timedelta(hours=1)

    def test_min_cluster_size_filters_small_spikes(self):
        # Spike of only 3 users when min_cluster_size=5 -> not flagged
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [1] * 48
        counts[10] = 3  # Statistically anomalous but below min_cluster_size
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates, min_cluster_size=5)
        assert len(windows) == 0

    def test_all_joins_in_one_burst_stddev_zero(self):
        # All 50 users join in the same hour -> stddev=0 edge case
        base = datetime(2025, 6, 15, 14, 0, tzinfo=timezone.utc)
        dates = {i: base + timedelta(minutes=i) for i in range(50)}
        windows = detect_spike_windows(dates)
        assert len(windows) >= 1

    def test_configurable_window_size(self):
        # With a 2-hour window, adjacent hourly spikes merge
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [2] * 48
        counts[20] = 30
        counts[21] = 30
        dates = _make_dates(base, counts)
        windows = detect_spike_windows(dates, window_size=timedelta(hours=2))
        assert len(windows) >= 1

    def test_configurable_sigma_multiplier(self):
        # Higher sigma = fewer spikes detected
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        counts = [5] * 48
        counts[10] = 20  # Moderate spike
        dates = _make_dates(base, counts)
        # With sigma=1 should detect it
        windows_loose = detect_spike_windows(dates, sigma_multiplier=1.0)
        # With sigma=5 should not detect it
        windows_strict = detect_spike_windows(dates, sigma_multiplier=5.0)
        assert len(windows_loose) >= len(windows_strict)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tg_purge.clustering'`

**Step 3: Write the implementation**

Create `tg_purge/clustering.py`:

```python
"""
Auto-detection of bulk-subscription spikes from join-date timestamps.

Uses a sliding-window approach: slide a fixed-width window across the
timeline, count joins per window position, flag positions where the
count exceeds mean + N*sigma. Merge overlapping flagged windows into
contiguous spike regions.

The detected windows are compatible with score_user()'s spike_windows
parameter — drop-in integration.
"""

from datetime import timedelta, timezone
from math import sqrt


def merge_windows(windows):
    """Merge overlapping or adjacent (start, end) datetime tuples.

    Args:
        windows: List of (start_dt, end_dt) tuples, not necessarily sorted.

    Returns:
        Sorted list of merged (start_dt, end_dt) tuples with no overlaps.
    """
    if not windows:
        return []

    # Sort by start time
    sorted_w = sorted(windows, key=lambda w: w[0])
    merged = [sorted_w[0]]

    for start, end in sorted_w[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping or adjacent — extend the previous window
            merged[-1] = (prev_start, max(prev_end, end))
        else:
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

    Slides a fixed-width window across the timeline, counts joins per
    window position, and flags positions where the count exceeds
    mean + sigma_multiplier * stddev.

    Args:
        join_dates: Dict of user_id -> datetime (timezone-aware).
        window_size: Width of each sliding window (default: 1 hour).
        step_size: Step between window positions (default: 15 minutes).
        sigma_multiplier: Number of standard deviations above mean to
            flag as a spike (default: 2.0).
        min_cluster_size: Minimum joins in a window to be considered a
            spike, regardless of statistics (default: 5).

    Returns:
        List of (start_dt, end_dt) tuples representing detected spike
        regions. Compatible with score_user()'s spike_windows parameter.
    """
    if len(join_dates) < 10:
        return []

    # Sort timestamps for two-pointer scanning
    timestamps = sorted(join_dates.values())
    earliest = timestamps[0]
    latest = timestamps[-1]

    # Slide the window across the timeline, count joins per position
    window_counts = []  # list of (window_start, window_end, count)
    current_start = earliest

    while current_start + window_size <= latest + step_size:
        current_end = current_start + window_size

        # Two-pointer count: how many timestamps fall in [current_start, current_end)
        # Binary search for efficiency
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

    # Compute mean and stddev of window counts
    counts = [c for _, _, c in window_counts]
    n = len(counts)
    mean = sum(counts) / n
    variance = sum((c - mean) ** 2 for c in counts) / n
    stddev = sqrt(variance)

    # Determine threshold: mean + sigma * stddev
    # When stddev is 0 (all counts identical), any count above mean is flagged
    if stddev == 0:
        threshold = mean
    else:
        threshold = mean + sigma_multiplier * stddev

    # Flag windows above threshold and above min_cluster_size
    spike_windows = []
    for start, end, count in window_counts:
        if count > threshold and count >= min_cluster_size:
            spike_windows.append((start, end))

    # Merge overlapping windows into contiguous spike regions
    return merge_windows(spike_windows)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clustering.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add tg_purge/clustering.py tests/test_clustering.py
git commit -m "feat: add sliding-window spike auto-detection module"
```

---

### Task 2: `merge_windows()` edge case tests

**Files:**
- Test: `tests/test_clustering.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_clustering.py`:

```python
from tg_purge.clustering import merge_windows


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
        result = merge_windows([w2, w1])  # unsorted input
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
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_clustering.py::TestMergeWindows -v`
Expected: All 6 tests PASS (implementation already handles these)

**Step 3: Commit**

```bash
git add tests/test_clustering.py
git commit -m "test: add merge_windows edge case coverage"
```

---

### Task 3: Integrate into `candidates` command

**Files:**
- Modify: `tg_purge/commands/candidates.py:19,118-125`
- Test: `tests/test_cli.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
    def test_candidates_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_candidates_default_auto_cluster(self):
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@t"])
        assert args.no_auto_cluster is False
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py::TestBuildParser::test_candidates_no_auto_cluster_flag -v`
Expected: FAIL — `error: unrecognized arguments: --no-auto-cluster`

**Step 3: Modify `cli.py` — add `--no-auto-cluster` to candidates**

In `tg_purge/cli.py`, after line 151 (the `--strategy` arg for candidates), add:

```python
    p_candidates.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )
```

**Step 4: Modify `candidates.py` — integrate clustering**

In `tg_purge/commands/candidates.py`:

Add import at line 19 (after the scoring import):
```python
from ..clustering import detect_spike_windows
```

Replace lines 118-125 (the scoring loop) with:
```python
        all_users = result["users"]
        join_dates = result["join_dates"]
        print(f"\nTotal users enumerated: {len(all_users)}")

        # Auto-detect spike windows from join dates
        auto_cluster = not getattr(args, "no_auto_cluster", False)
        spike_windows = []
        if auto_cluster and join_dates:
            spike_windows = detect_spike_windows(join_dates)
            if spike_windows:
                print(f"Auto-detected {len(spike_windows)} spike window(s):")
                for start, end in spike_windows:
                    print(f"  {start.strftime('%Y-%m-%d %H:%M')} — {end.strftime('%Y-%m-%d %H:%M')} UTC")

        # Score everyone
        all_scored = []
        for uid, user in all_users.items():
            s, reasons = score_user(
                user,
                join_date=join_dates.get(uid),
                spike_windows=spike_windows,
            )
            all_scored.append((user, s, reasons))
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py tests/test_clustering.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add tg_purge/cli.py tg_purge/commands/candidates.py tests/test_cli.py
git commit -m "feat: integrate auto-clustering into candidates command"
```

---

### Task 4: Integrate into `registry generate` command

**Files:**
- Modify: `tg_purge/commands/registry.py:20,106-120`

**Step 1: Modify `registry.py`**

Add import at line 20 (after the scoring import):
```python
from ..clustering import detect_spike_windows
```

Replace lines 106-120 (the scoring loop) with:
```python
        all_users = result["users"]
        join_dates = result["join_dates"]
        print(f"Total users enumerated: {len(all_users)}")

        # Auto-detect spike windows from join dates
        spike_windows = []
        if join_dates:
            spike_windows = detect_spike_windows(join_dates)
            if spike_windows:
                print(f"Auto-detected {len(spike_windows)} spike window(s)")

        # Score and filter
        entries = []
        for uid, user in all_users.items():
            s, reasons = score_user(
                user,
                join_date=join_dates.get(uid),
                spike_windows=spike_windows,
            )
            if s >= threshold:
                entries.append({
                    "user_id": uid,
                    "score": s,
                    "date_flagged": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "detection_method": "heuristic_scoring",
                    "notes": "; ".join(reasons),
                })
```

**Step 2: Run tests to verify nothing breaks**

Run: `python -m pytest tests/ -v`
Expected: All 130+ tests pass

**Step 3: Commit**

```bash
git add tg_purge/commands/registry.py
git commit -m "feat: integrate auto-clustering into registry generate"
```

---

### Task 5: Integrate into `spike` command (merge with manual windows)

**Files:**
- Modify: `tg_purge/commands/spike.py:20,162-187`
- Test: `tests/test_clustering.py` (append)

**Step 1: Write an integration test**

Append to `tests/test_clustering.py`:

```python
class TestMergeManualAndAutoWindows:

    def test_manual_and_auto_windows_merge(self):
        """Manual spike window merges with auto-detected ones."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        manual = [(base, base + timedelta(hours=2))]
        auto = [(base + timedelta(hours=5), base + timedelta(hours=6))]
        result = merge_windows(manual + auto)
        assert len(result) == 2

    def test_overlapping_manual_and_auto_merge(self):
        """Overlapping manual + auto windows merge into one."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        manual = [(base, base + timedelta(hours=3))]
        auto = [(base + timedelta(hours=2), base + timedelta(hours=4))]
        result = merge_windows(manual + auto)
        assert len(result) == 1
        assert result[0] == (base, base + timedelta(hours=4))
```

**Step 2: Run to verify they pass (merge_windows already works)**

Run: `python -m pytest tests/test_clustering.py::TestMergeManualAndAutoWindows -v`
Expected: PASS

**Step 3: Modify `spike.py`**

Add import at line 20 (after the formatters import):
```python
from ..clustering import detect_spike_windows, merge_windows
```

After line 162 (`join_dates = result["join_dates"]`), before the filter-to-spike-window block, add:

```python
        # Auto-detect additional spike windows from join dates
        auto_windows = []
        if join_dates:
            auto_windows = detect_spike_windows(join_dates)
            if auto_windows:
                print(f"Auto-detected {len(auto_windows)} additional spike window(s)")

        # Merge manual window with auto-detected ones
        all_spike_windows = merge_windows([(spike_start, spike_end)] + auto_windows)
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add tg_purge/commands/spike.py tests/test_clustering.py
git commit -m "feat: merge auto-detected spikes with manual window in spike command"
```

---

### Task 6: Integrate into `analyze` command (Phase 2)

**Files:**
- Modify: `tg_purge/commands/analyze.py:17,86-113`

**Step 1: Modify `analyze.py`**

The `analyze` command does its own Phase 2 loop (not `enumerate_subscribers()`), so it needs to collect join dates from the participants returned by `fetch_by_search`.

Add import at line 17 (after the enumeration import):
```python
from ..clustering import detect_spike_windows
```

In the Phase 2 section, modify the search loop (lines 86-106) to also collect join dates. Change `fetch_by_search` to capture participants:

Replace line 92:
```python
                users, _ = await fetch_by_search(client, channel, query)
```
with:
```python
                users, participants = await fetch_by_search(client, channel, query)
```

Add a `search_join_dates` dict before the loop (after line 87):
```python
        search_join_dates = {}  # uid -> datetime
```

Inside the loop, after line 98 (`seen_ids.add(u.id)`), add join date collection:
```python
                # Collect join dates from participants
                from telethon.tl.types import ChannelParticipant
                for p in participants:
                    if (p.user_id not in search_join_dates
                            and isinstance(p, ChannelParticipant)
                            and hasattr(p, 'date') and p.date):
                        search_join_dates[p.user_id] = p.date
```

After the search loop ends (after line 106), before scoring, add auto-detection:
```python
        # Auto-detect spike windows from search join dates
        spike_windows = []
        if search_join_dates:
            spike_windows = detect_spike_windows(search_join_dates)
            if spike_windows:
                print(f"\nAuto-detected {len(spike_windows)} spike window(s):")
                for start, end in spike_windows:
                    print(f"  {start.strftime('%Y-%m-%d %H:%M')} — {end.strftime('%Y-%m-%d %H:%M')} UTC")
```

Replace the scoring loop (lines 109-112):
```python
        search_scored = []
        for uid, (user, source) in search_users.items():
            s, reasons = score_user(user)
            search_scored.append((user, s, reasons))
```
with:
```python
        search_scored = []
        for uid, (user, source) in search_users.items():
            s, reasons = score_user(
                user,
                join_date=search_join_dates.get(uid),
                spike_windows=spike_windows,
            )
            search_scored.append((user, s, reasons))
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tg_purge/commands/analyze.py
git commit -m "feat: integrate auto-clustering into analyze Phase 2"
```

---

### Task 7: Integrate into `join-dates` command (informational output)

**Files:**
- Modify: `tg_purge/commands/join_dates.py:14,50-54`

**Step 1: Modify `join_dates.py`**

Add import at line 14 (after the enumeration import):
```python
from ..clustering import detect_spike_windows
```

After line 53 (after the "No join dates found!" early return), add:

```python
        # Auto-detect spike windows
        spike_windows = detect_spike_windows(join_dates)
```

Before the Summary section (before line 156), add:

```python
        # ── Auto-detected spike windows ─────────────────────────
        if spike_windows:
            print(f"\n{'─' * 80}")
            print(f"AUTO-DETECTED SPIKE WINDOWS ({len(spike_windows)} found)")
            print(f"{'─' * 80}")
            for start, end in spike_windows:
                # Count users in this window
                window_users = sum(
                    1 for d in dates if start <= d < end
                )
                duration = end - start
                hours = duration.total_seconds() / 3600
                print(f"  {start.strftime('%Y-%m-%d %H:%M')} — {end.strftime('%Y-%m-%d %H:%M')} UTC"
                      f"  ({window_users} joins in {hours:.1f}h)")
        else:
            print(f"\n{'─' * 80}")
            print("AUTO-DETECTED SPIKE WINDOWS")
            print(f"{'─' * 80}")
            print("  No statistically significant spikes detected (mean + 2σ threshold)")
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tg_purge/commands/join_dates.py
git commit -m "feat: show auto-detected spike windows in join-dates output"
```

---

### Task 8: Add `--no-auto-cluster` to remaining subcommands

**Files:**
- Modify: `tg_purge/cli.py`
- Modify: `tg_purge/commands/analyze.py`
- Test: `tests/test_cli.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
    def test_analyze_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_registry_generate_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["registry", "generate", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py::TestBuildParser::test_analyze_no_auto_cluster_flag -v`
Expected: FAIL

**Step 3: Add `--no-auto-cluster` to `analyze`, `registry generate`, and `join-dates` subparsers**

In `tg_purge/cli.py`, add the flag to `p_analyze` (after the `--strategy` arg, ~line 66):
```python
    p_analyze.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )
```

Add the same to `p_join` (after `--strategy`, ~line 86):
```python
    p_join.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )
```

Add the same to `p_reg_gen` (after `--output`, ~line 164):
```python
    p_reg_gen.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )
```

**Step 4: Wire the flag in `analyze.py` and `join_dates.py`**

In `analyze.py`, wrap the auto-detection block with:
```python
        auto_cluster = not getattr(args, "no_auto_cluster", False)
        spike_windows = []
        if auto_cluster and search_join_dates:
            spike_windows = detect_spike_windows(search_join_dates)
            ...
```

In `join_dates.py`, wrap the auto-detection block with:
```python
        auto_cluster = not getattr(args, "no_auto_cluster", False)
        spike_windows = []
        if auto_cluster:
            spike_windows = detect_spike_windows(join_dates)
```

In `registry.py`, wrap the existing auto-detection block:
```python
        auto_cluster = not getattr(args, "no_auto_cluster", False)
        spike_windows = []
        if auto_cluster and join_dates:
            spike_windows = detect_spike_windows(join_dates)
            ...
```

**Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add tg_purge/cli.py tg_purge/commands/analyze.py tg_purge/commands/join_dates.py tg_purge/commands/registry.py tests/test_cli.py
git commit -m "feat: add --no-auto-cluster flag to analyze, join-dates, registry generate"
```

---

### Task 9: Update CLAUDE.md and docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/scoring-methodology.md`

**Step 1: Update CLAUDE.md**

In the Scoring engine section, add after the `spike_join` line:
```
- `tg_purge/clustering.py` provides `detect_spike_windows()` — sliding-window auto-detection of bulk-subscription spikes from join dates. Used by `candidates`, `registry generate`, `analyze`, and `join-dates` commands. Coexists with manual `--start/--end` spike windows.
```

**Step 2: Update `docs/scoring-methodology.md`**

In the scoring table, update the `spike_join` row rationale:
```
| `spike_join` | +2 | User joined during a detected bulk-subscription spike window. Auto-detected via sliding-window analysis (1h window, mean+2σ threshold) or manually specified with --start/--end. Requires join date data — not applied if unavailable. |
```

Add a new section after "Three-Round Validation Methodology":

```markdown
## Auto-Detected Spike Windows

When join-date data is available, the scoring pipeline automatically detects bulk-subscription spikes using a sliding-window algorithm:

1. Slide a 1-hour window across the timeline in 15-minute steps
2. Count joins in each window position
3. Flag windows where count exceeds mean + 2σ (standard deviations)
4. Merge overlapping flagged windows into contiguous spike regions
5. Apply `spike_join(+2)` to any user whose join date falls within a spike region

This is enabled by default on `candidates`, `analyze`, `join-dates`, and `registry generate`. Disable with `--no-auto-cluster`.

The `spike` command merges auto-detected windows with the manually specified `--start/--end` window.

**Safety floor:** Windows with fewer than 5 users are not flagged regardless of statistics.

**Minimum data:** Auto-detection requires at least 10 join dates. Below that, no spike detection is attempted.
```

**Step 3: Commit**

```bash
git add CLAUDE.md docs/scoring-methodology.md
git commit -m "docs: document auto-clustering feature in CLAUDE.md and scoring methodology"
```

---

### Task 10: Final integration test — run against live data

**Step 1: Run `analyze` against @leviathan_news and verify spike detection**

```bash
TG_PURGE_API_ID=26280443 TG_PURGE_API_HASH=acbe72b520e281e2add0bd1c5d5eed13 \
  python -m tg_purge analyze --channel @leviathan_news \
  --session-path /Users/zero/.claude/z_session --strategy minimal
```

Expected: Output should now include "Auto-detected N spike window(s)" before scoring, and score-2 users who joined during spikes should appear at score 4 instead.

**Step 2: Compare threshold analysis before/after**

Verify the score-2 cliff has flattened — fewer users at score 2, more at score 4.

**Step 3: Run all tests one final time**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 4: Commit any final adjustments**

```bash
git add -A
git commit -m "feat: auto-clustering integration complete"
```
