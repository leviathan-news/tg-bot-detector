# Auto-Detected Join-Date Clustering — Design Document

**Goal:** Automatically detect bulk-subscription spikes from join-date timestamps and integrate them as a scoring signal, breaking the score-2 cliff where 64.4% of sampled users accumulate.

**Problem:** On @leviathan_news, 4,635 of 7,202 sampled users score exactly 2 from `no_username(+1) + no_last+no_user(+1)`. This population is a mix of real humans with sparse profiles and bot-farm accounts. The current heuristics cannot distinguish between them. Join-date clustering can — bot farms subscribe accounts in temporal bursts, real users join sporadically.

## Architecture

### Core algorithm: sliding window spike detection

New module `tg_purge/clustering.py` with a single public function:

```
detect_spike_windows(join_dates, window_size, step_size, sigma_multiplier, min_cluster_size)
    → list[(datetime, datetime)]
```

**Algorithm:**
1. Sort all timestamps into an array
2. Slide a 1-hour window across the timeline in 15-minute steps
3. For each position, count joins using a two-pointer approach (O(n) total scan after O(n log n) sort)
4. Compute mean and stddev across all window counts
5. Flag windows where count > mean + 2σ
6. Merge overlapping flagged windows into contiguous spike regions
7. Return list of (start, end) tuples

**Configurable defaults:**
- `window_size`: 1 hour
- `step_size`: 15 minutes
- `sigma_multiplier`: 2.0
- `min_cluster_size`: 5 (safety floor — ignore statistically anomalous windows with < 5 users)

**Edge cases:**
- < 10 join dates → return empty (insufficient data)
- All joins in one burst (stddev = 0) → every window with count > mean is flagged
- No spikes detected → return empty, scoring proceeds without spike signal

### Helper: window merging

```
merge_windows(windows) → list[(datetime, datetime)]
```

Merges overlapping or adjacent (start, end) tuples into contiguous regions. Used both for auto-detected windows and for combining auto-detected + manual windows in the `spike` command.

## Integration

### Pipeline flow

```
enumerate_subscribers() → detect_spike_windows(join_dates) → score_user(user, join_date, spike_windows)
```

### Commands affected

| Command | Change |
|---------|--------|
| `candidates` | Auto-detect spikes from join_dates, pass to score_user() |
| `registry generate` | Same as candidates |
| `spike` | Merge auto-detected windows with manual --start/--end window |
| `analyze` | Pass join dates from Phase 2 search results to scoring |
| `join-dates` | Print detected spike windows as informational output |
| `validate` | No change (scores individual known users, not channel-wide) |

### CLI flag

`--no-auto-cluster` added to subcommands that use auto-detection. Disables the feature when not wanted (e.g., testing, or channels where join dates are unreliable).

### Coexistence with manual spike windows

The existing `spike_windows` parameter on `score_user()` is preserved. The `spike` command merges auto-detected windows with the manual `--start/--end` window via `merge_windows()`. Manual windows are always included.

### Backward compatibility

If `enumerate_subscribers()` returns no join dates (some participant types don't have them), auto-detection returns empty and scoring proceeds as before. Zero breakage.

## Expected impact on score-2 cliff

**Before:** 4,635 users at score 2 (64.4% of sample)

**After:** Users at score 2 who joined during a detected spike get `spike_join(+2)`, pushing them to score 4. The cliff splits:
- Score 2: real humans with sparse profiles (joined organically)
- Score 4: bot-farm accounts with sparse profiles (joined in bursts)

**What this does NOT solve:**
- Bot farms that trickle accounts slowly (no temporal signal)
- Bot farms with well-crafted profiles (score 0 regardless)
- Real users who joined during organic spikes (false positives, mitigated by 2σ threshold)
- `analyze` Phase 2 still uses its own enumeration loop (not `enumerate_subscribers()`), so join dates are only available from the participants returned by search, not from the separate phased fetch

## Testing

### Unit tests (`tests/test_clustering.py`)
- Empty / insufficient join dates → empty result
- Uniform distribution → no spikes
- Single obvious spike → correct window returned
- Two separate spikes → two windows
- Overlapping window positions → merged into single region
- stddev=0 edge case → flags the burst
- Configurable params honored
- merge_windows() with overlapping, adjacent, and disjoint inputs

### Integration tests
- score_user() with auto-detected windows → spike_join reason present
- Score-2 user inside spike → score 4
- Score-2 user outside spike → stays score 2
- Manual + auto-detected windows merge correctly

## Files

- **Create:** `tg_purge/clustering.py`, `tests/test_clustering.py`
- **Modify:** `candidates.py`, `registry.py`, `spike.py`, `analyze.py`, `join_dates.py`, `cli.py`
- **No changes:** `scoring.py`, `enumeration.py`, `client.py`, `config.py`, `formatters.py`
