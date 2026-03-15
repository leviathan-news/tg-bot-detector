# ML-Based Bot Detection & Coverage Expansion — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ML classification, expanded enumeration, cross-channel cohort detection, and statistical sampling to tg-bot-detector while preserving the existing heuristic workflow as default.

**Architecture:** Dual-track — feature extraction + statistics are independent foundations. ML pipeline (labeling, training, inference) builds on features. Coverage expansion (API optimization + collectors) is independent. Cross-channel cohort detection depends on collectors. All new ML dependencies are optional (`[ml]` extra).

**Tech Stack:** Python 3.9+, Telethon (existing), scikit-learn, LightGBM, XGBoost, numpy, pandas (all optional via `[ml]`)

**Spec:** `docs/superpowers/specs/2026-03-11-ml-detection-design.md`

**Note on serialization:** Model artifacts use native LightGBM/XGBoost serialization formats to avoid arbitrary code execution risks. The sklearn RandomForest fallback uses joblib — documented in metadata as a known tradeoff since sklearn lacks a safe native format. Users should only load models from trusted sources (their own training or official GitHub releases).

---

## Chunk 1: Foundation

### Task 1: Project Configuration Updates

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update .gitignore**

Add to `.gitignore`:
```
# Datasets (contain real user IDs — PII sensitive)
datasets/

# Trained ML models
models/

# Output files
output/
```

- [ ] **Step 2: Update pyproject.toml**

Add ML optional dependency group:
```toml
[project.optional-dependencies]
toml = ["tomli>=1.0; python_version < '3.11'"]
ml = [
    "scikit-learn>=1.3",
    "numpy>=1.24",
    "pandas>=2.0",
    "lightgbm>=4.0",
    "xgboost>=2.0",
]
```

- [ ] **Step 3: Verify install**

Run: `pip install -e ".[ml]"`
Expected: installs all ML dependencies without errors.

- [ ] **Step 4: Verify existing tests still pass**

Run: `python -m pytest tests/ -v`
Expected: all 154 tests pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml
git commit -m "chore: add ML optional deps and gitignore datasets/models"
```

---

### Task 2: Feature Extraction Module

**Files:**
- Create: `tg_purge/features.py`
- Create: `tests/test_features.py`

This module extracts numeric feature vectors from User objects for ML training and inference. It parallels `scoring.py` but outputs structured numerics instead of an aggregated score.

- [ ] **Step 1: Write tests for account flag features**

Create `tests/test_features.py`:
```python
"""Tests for ML feature extraction.

Feature extraction mirrors the scoring module's inspection logic but outputs
numeric feature vectors (dict[str, float]) instead of aggregated integer scores.
Tests use the same MockUser fixtures from conftest.py.
"""

from tg_purge.features import extract_features


class TestAccountFlags:
    """Test binary account flag features (0/1)."""

    def test_clean_user_flags(self, clean_user):
        """A clean user should have all flag features at 0 except non-flag fields."""
        features = extract_features(clean_user)
        assert features["is_deleted"] == 0
        assert features["is_bot"] == 0
        assert features["is_scam"] == 0
        assert features["is_fake"] == 0
        assert features["is_restricted"] == 0

    def test_deleted_user(self, deleted_user):
        """Deleted user should have is_deleted=1."""
        features = extract_features(deleted_user)
        assert features["is_deleted"] == 1

    def test_bot_user(self, make_user):
        """Bot user should have is_bot=1."""
        user = make_user(bot=True)
        features = extract_features(user)
        assert features["is_bot"] == 1

    def test_scam_user(self, make_user):
        """Scam-flagged user should have is_scam=1."""
        user = make_user(scam=True)
        features = extract_features(user)
        assert features["is_scam"] == 1

    def test_fake_user(self, make_user):
        """Fake-flagged user should have is_fake=1."""
        user = make_user(fake=True)
        features = extract_features(user)
        assert features["is_fake"] == 1

    def test_restricted_user(self, make_user):
        """Restricted user should have is_restricted=1."""
        user = make_user(restricted=True)
        features = extract_features(user)
        assert features["is_restricted"] == 1

    def test_premium_user(self, premium_user):
        """Premium user should have is_premium=1 and has_emoji_status=1."""
        features = extract_features(premium_user)
        assert features["is_premium"] == 1
        assert features["has_emoji_status"] == 1

    def test_non_premium_user(self, clean_user):
        """Non-premium user should have is_premium=0."""
        features = extract_features(clean_user)
        assert features["is_premium"] == 0
        assert features["has_emoji_status"] == 0
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_features.py::TestAccountFlags -v`
Expected: FAIL (module does not exist yet)

- [ ] **Step 3: Write tests for profile completeness features**

Append to `tests/test_features.py`:
```python
class TestProfileFeatures:
    """Test profile completeness features."""

    def test_complete_profile(self, clean_user):
        """User with full profile should have all completeness features positive."""
        features = extract_features(clean_user)
        assert features["has_photo"] == 1
        assert features["has_username"] == 1
        assert features["has_last_name"] == 1
        assert features["first_name_length"] == 5  # "Alice"

    def test_empty_profile(self, make_user):
        """User with bare profile should have completeness features at 0."""
        user = make_user(photo=False, username=None, last_name=None, first_name="A")
        features = extract_features(user)
        assert features["has_photo"] == 0
        assert features["has_username"] == 0
        assert features["has_last_name"] == 0
        assert features["first_name_length"] == 1

    def test_digit_name_ratio(self, make_user):
        """Digit ratio should be calculated correctly for numeric names."""
        user = make_user(first_name="User123")
        features = extract_features(user)
        # 3 digits out of 7 chars = ~0.4286
        assert abs(features["name_digit_ratio"] - 3 / 7) < 0.01

    def test_digit_name_ratio_no_digits(self, clean_user):
        """Digit ratio should be 0 for alphabetic names."""
        features = extract_features(clean_user)
        assert features["name_digit_ratio"] == 0.0

    def test_mixed_scripts_latin_cyrillic(self, make_user):
        """Mixed Latin+Cyrillic name should have script_count=2."""
        user = make_user(first_name="Abc\u0430\u0431\u0432")
        features = extract_features(user)
        assert features["script_count"] == 2

    def test_single_script(self, clean_user):
        """Pure Latin name should have script_count=1."""
        features = extract_features(clean_user)
        assert features["script_count"] == 1

    def test_no_name(self, make_user):
        """Deleted user with no name should have safe defaults."""
        user = make_user(first_name=None, last_name=None, deleted=True)
        features = extract_features(user)
        assert features["first_name_length"] == 0
        assert features["name_digit_ratio"] == 0.0
        assert features["script_count"] == 0
```

- [ ] **Step 4: Write tests for activity status features**

Append to `tests/test_features.py`:
```python
from tests.conftest import (
    UserStatusEmpty, UserStatusOnline, UserStatusRecently,
    UserStatusLastWeek, UserStatusLastMonth, UserStatusOffline,
)
from datetime import datetime, timezone, timedelta


class TestActivityFeatures:
    """Test activity status one-hot encoding and days_since_last_seen."""

    def test_status_empty(self, make_user):
        """UserStatusEmpty should produce correct one-hot."""
        user = make_user(status=UserStatusEmpty())
        features = extract_features(user)
        assert features["status_empty"] == 1
        assert features["status_online"] == 0
        assert features["status_recently"] == 0
        assert features["status_last_week"] == 0
        assert features["status_last_month"] == 0
        assert features["status_offline"] == 0
        assert features["days_since_last_seen"] == -1

    def test_status_online(self, make_user):
        """UserStatusOnline should produce correct one-hot."""
        user = make_user(status=UserStatusOnline())
        features = extract_features(user)
        assert features["status_online"] == 1
        assert features["days_since_last_seen"] == 0

    def test_status_offline_computes_days(self, make_user):
        """UserStatusOffline should compute days since last seen."""
        was_online = datetime.now(timezone.utc) - timedelta(days=42)
        user = make_user(status=UserStatusOffline(was_online))
        features = extract_features(user)
        assert features["status_offline"] == 1
        assert abs(features["days_since_last_seen"] - 42) <= 1

    def test_status_recently(self, make_user):
        """UserStatusRecently should set days_since_last_seen to 1."""
        user = make_user(status=UserStatusRecently())
        features = extract_features(user)
        assert features["status_recently"] == 1
        assert features["days_since_last_seen"] == 1

    def test_none_status_treated_as_empty(self, make_user):
        """None status should be treated same as UserStatusEmpty."""
        user = make_user(status=None)
        features = extract_features(user)
        assert features["status_empty"] == 1
        assert features["days_since_last_seen"] == -1
```

- [ ] **Step 5: Write tests for temporal and heuristic features**

Append to `tests/test_features.py`:
```python
class TestTemporalFeatures:
    """Test join-date derived features."""

    def test_no_join_date(self, clean_user):
        """Without join_date, temporal features should be defaults."""
        features = extract_features(clean_user)
        assert features["is_spike_join"] == 0
        assert features["days_since_join"] == -1
        assert features["join_hour_utc"] == -1
        assert features["join_day_of_week"] == -1

    def test_with_join_date(self, clean_user):
        """With join_date, temporal features should be populated."""
        # Wednesday 2025-11-12 at 14:30 UTC
        join = datetime(2025, 11, 12, 14, 30, tzinfo=timezone.utc)
        features = extract_features(clean_user, join_date=join)
        assert features["join_hour_utc"] == 14
        assert features["join_day_of_week"] == 2  # Wednesday
        assert features["days_since_join"] > 0

    def test_spike_join_detected(self, clean_user):
        """User joining during spike window should have is_spike_join=1."""
        join = datetime(2025, 11, 12, 14, 30, tzinfo=timezone.utc)
        windows = [
            (datetime(2025, 11, 12, 14, 0, tzinfo=timezone.utc),
             datetime(2025, 11, 12, 15, 0, tzinfo=timezone.utc))
        ]
        features = extract_features(clean_user, join_date=join, spike_windows=windows)
        assert features["is_spike_join"] == 1

    def test_spike_join_not_in_window(self, clean_user):
        """User joining outside spike window should have is_spike_join=0."""
        join = datetime(2025, 11, 12, 16, 0, tzinfo=timezone.utc)
        windows = [
            (datetime(2025, 11, 12, 14, 0, tzinfo=timezone.utc),
             datetime(2025, 11, 12, 15, 0, tzinfo=timezone.utc))
        ]
        features = extract_features(clean_user, join_date=join, spike_windows=windows)
        assert features["is_spike_join"] == 0


class TestHeuristicScoreFeature:
    """Test that heuristic_score is included as a feature."""

    def test_clean_user_score(self, clean_user):
        """Clean user should have a low heuristic score."""
        features = extract_features(clean_user)
        assert features["heuristic_score"] == 0

    def test_bot_user_score(self, make_user):
        """Bot-like user should have a high heuristic score."""
        user = make_user(
            bot=True, photo=False, username=None, first_name="A",
            last_name=None, status=UserStatusEmpty(),
        )
        features = extract_features(user)
        assert features["heuristic_score"] >= 3


class TestCohortFeatures:
    """Test cohort features when cohort_data is provided."""

    def test_no_cohort_data(self, clean_user):
        """Without cohort_data, cohort features should be 0."""
        features = extract_features(clean_user)
        assert features["is_cohort_member"] == 0
        assert features["cohort_size"] == 0
        assert features["cohort_join_spread_hours"] == 0.0
        assert features["cohort_profile_similarity"] == 0.0

    def test_with_cohort_data(self, clean_user):
        """With cohort_data, cohort features should be populated."""
        cohort = {
            "is_member": True,
            "size": 150,
            "join_spread_hours": 4.2,
            "profile_similarity": 0.87,
        }
        features = extract_features(clean_user, cohort_data=cohort)
        assert features["is_cohort_member"] == 1
        assert features["cohort_size"] == 150
        assert features["cohort_join_spread_hours"] == 4.2
        assert features["cohort_profile_similarity"] == 0.87


class TestFeatureCompleteness:
    """Test that feature vector always has the expected keys."""

    def test_all_feature_keys_present(self, clean_user):
        """Every call should return all expected feature keys."""
        features = extract_features(clean_user)
        expected_keys = {
            "is_deleted", "is_bot", "is_scam", "is_fake",
            "is_restricted", "is_premium", "has_emoji_status",
            "has_photo", "has_username", "has_last_name",
            "first_name_length", "name_digit_ratio", "script_count",
            "status_empty", "status_online", "status_recently",
            "status_last_week", "status_last_month", "status_offline",
            "days_since_last_seen",
            "is_spike_join", "days_since_join",
            "join_hour_utc", "join_day_of_week",
            "is_cohort_member", "cohort_size",
            "cohort_join_spread_hours", "cohort_profile_similarity",
            "heuristic_score",
        }
        assert set(features.keys()) == expected_keys

    def test_all_values_are_numeric(self, clean_user):
        """Every feature value should be int or float."""
        features = extract_features(clean_user)
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}"
```

- [ ] **Step 6: Implement features.py**

Create `tg_purge/features.py`:
```python
"""
ML feature extraction from Telethon User objects.

Parallel path to scoring.py — extracts numeric feature vectors (dict[str, float])
for ML training and inference. Uses the same type(status).__name__ pattern for
testability without Telethon installed.

Does NOT depend on numpy/pandas — returns plain dicts. ML code converts to arrays
at training/inference time.
"""

import re
from datetime import datetime, timezone

from .scoring import score_user, ScoringConfig

# Same status type mapping as scoring.py — kept in sync manually.
# Using string comparison on type(status).__name__ so this module
# can be tested without Telethon installed.
_STATUS_TYPE_NAMES = {
    "UserStatusEmpty": "empty",
    "UserStatusOnline": "online",
    "UserStatusRecently": "recently",
    "UserStatusLastWeek": "last_week",
    "UserStatusLastMonth": "last_month",
    "UserStatusOffline": "offline",
}

# Canonical list of all feature keys, in stable order.
# Used for consistent column ordering in training data.
FEATURE_KEYS = [
    "is_deleted", "is_bot", "is_scam", "is_fake",
    "is_restricted", "is_premium", "has_emoji_status",
    "has_photo", "has_username", "has_last_name",
    "first_name_length", "name_digit_ratio", "script_count",
    "status_empty", "status_online", "status_recently",
    "status_last_week", "status_last_month", "status_offline",
    "days_since_last_seen",
    "is_spike_join", "days_since_join",
    "join_hour_utc", "join_day_of_week",
    "is_cohort_member", "cohort_size",
    "cohort_join_spread_hours", "cohort_profile_similarity",
    "heuristic_score",
]


def _status_type_name(status):
    """Get normalized status type name without requiring Telethon imports."""
    if status is None:
        return "empty"
    return _STATUS_TYPE_NAMES.get(type(status).__name__, "unknown")


def extract_features(user, join_date=None, spike_windows=None, cohort_data=None):
    """Extract a numeric feature vector from a Telethon User object.

    This is the ML-facing counterpart to score_user(). It produces a flat
    dict of numeric values suitable for feeding into sklearn/lightgbm/xgboost.

    Args:
        user: A Telethon User object (or MockUser with compatible attributes).
        join_date: Optional datetime of when the user joined the channel.
        spike_windows: Optional list of (start_dt, end_dt) tuples for spike detection.
        cohort_data: Optional dict with keys: is_member (bool), size (int),
            join_spread_hours (float), profile_similarity (float).
            Provided by cross-channel cohort detection.

    Returns:
        dict[str, float] with all keys from FEATURE_KEYS. Every value is
        int or float. Missing data uses sentinel values (-1 for unknown
        continuous, 0 for unknown binary).
    """
    features = {}

    # ---- Account flags (binary 0/1) ----
    features["is_deleted"] = int(getattr(user, "deleted", False))
    features["is_bot"] = int(getattr(user, "bot", False))
    features["is_scam"] = int(getattr(user, "scam", False))
    features["is_fake"] = int(getattr(user, "fake", False))
    features["is_restricted"] = int(getattr(user, "restricted", False))
    features["is_premium"] = int(getattr(user, "premium", False))
    features["has_emoji_status"] = int(bool(getattr(user, "emoji_status", None)))

    # ---- Profile completeness ----
    features["has_photo"] = int(bool(getattr(user, "photo", None)))
    features["has_username"] = int(bool(getattr(user, "username", None)))
    features["has_last_name"] = int(bool(getattr(user, "last_name", None)))

    first = getattr(user, "first_name", None) or ""
    features["first_name_length"] = len(first)

    # Digit ratio: fraction of characters in first_name that are digits
    if first:
        features["name_digit_ratio"] = sum(c.isdigit() for c in first) / len(first)
    else:
        features["name_digit_ratio"] = 0.0

    # Script count: how many distinct script families are present in first_name.
    # Latin, Cyrillic, Arabic, CJK. Value >= 2 indicates mixed scripts.
    if first:
        has_latin = bool(re.search(r"[a-zA-Z]", first))
        has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", first))
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", first))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", first))
        features["script_count"] = sum([has_latin, has_cyrillic, has_arabic, has_cjk])
    else:
        features["script_count"] = 0

    # ---- Activity status (one-hot + continuous) ----
    status_type = _status_type_name(user.status)
    for st in ["empty", "online", "recently", "last_week", "last_month", "offline"]:
        features[f"status_{st}"] = int(status_type == st)

    # days_since_last_seen: continuous. -1 means unknown.
    if status_type == "offline" and hasattr(user.status, "was_online"):
        delta = datetime.now(timezone.utc) - user.status.was_online
        features["days_since_last_seen"] = delta.days
    elif status_type == "online":
        features["days_since_last_seen"] = 0
    elif status_type == "recently":
        features["days_since_last_seen"] = 1  # approximation
    elif status_type == "last_week":
        features["days_since_last_seen"] = 7  # approximation
    elif status_type == "last_month":
        features["days_since_last_seen"] = 30  # approximation
    else:
        features["days_since_last_seen"] = -1  # unknown (empty/none)

    # ---- Temporal features (join-date derived) ----
    if join_date is not None:
        features["join_hour_utc"] = join_date.hour
        features["join_day_of_week"] = join_date.weekday()
        features["days_since_join"] = (datetime.now(timezone.utc) - join_date).days
    else:
        features["join_hour_utc"] = -1
        features["join_day_of_week"] = -1
        features["days_since_join"] = -1

    # Spike join: 1 if join_date falls within any spike window
    features["is_spike_join"] = 0
    if join_date is not None and spike_windows:
        for window_start, window_end in spike_windows:
            if window_start <= join_date < window_end:
                features["is_spike_join"] = 1
                break

    # ---- Cross-channel cohort features ----
    if cohort_data and cohort_data.get("is_member"):
        features["is_cohort_member"] = 1
        features["cohort_size"] = cohort_data.get("size", 0)
        features["cohort_join_spread_hours"] = cohort_data.get("join_spread_hours", 0.0)
        features["cohort_profile_similarity"] = cohort_data.get("profile_similarity", 0.0)
    else:
        features["is_cohort_member"] = 0
        features["cohort_size"] = 0
        features["cohort_join_spread_hours"] = 0.0
        features["cohort_profile_similarity"] = 0.0

    # ---- Heuristic score (existing scoring engine output as a feature) ----
    heuristic_score, _ = score_user(
        user, config=ScoringConfig(),
        join_date=join_date, spike_windows=spike_windows,
    )
    features["heuristic_score"] = heuristic_score

    return features
```

- [ ] **Step 7: Run tests — verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add tg_purge/features.py tests/test_features.py
git commit -m "feat: add ML feature extraction module"
```

---

### Task 3: Statistical Sampling Framework

**Files:**
- Create: `tg_purge/statistics.py`
- Create: `tests/test_statistics.py`

Pure-math module for confidence intervals, bias estimation, and stratified extrapolation. No external dependencies — stdlib `math` only.

- [ ] **Step 1: Write tests**

Create `tests/test_statistics.py`:
```python
"""Tests for statistical sampling framework.

All statistical functions use stdlib math only — no numpy/scipy required.
Tests verify mathematical correctness of confidence intervals, bias
estimation, and stratified extrapolation.
"""

from tg_purge.statistics import (
    estimate_bot_rate,
    sample_quality_report,
    wilson_score_interval,
)


class TestWilsonScoreInterval:
    """Test Wilson score confidence interval calculation."""

    def test_50_percent_rate(self):
        """50% observed rate with large sample should have tight CI."""
        lower, upper = wilson_score_interval(successes=500, total=1000, z=1.96)
        assert 0.46 < lower < 0.50
        assert 0.50 < upper < 0.54

    def test_zero_successes(self):
        """0 successes should return lower bound of 0."""
        lower, upper = wilson_score_interval(successes=0, total=100, z=1.96)
        assert lower == 0.0
        assert upper > 0.0

    def test_all_successes(self):
        """All successes should return upper bound of 1."""
        lower, upper = wilson_score_interval(successes=100, total=100, z=1.96)
        assert lower < 1.0
        assert upper == 1.0

    def test_small_sample_wide_interval(self):
        """Small sample should produce wider CI than large sample."""
        lower_s, upper_s = wilson_score_interval(successes=5, total=10, z=1.96)
        lower_l, upper_l = wilson_score_interval(successes=500, total=1000, z=1.96)
        assert (upper_s - lower_s) > (upper_l - lower_l)

    def test_zero_total(self):
        """Zero total should return (0, 0) without error."""
        lower, upper = wilson_score_interval(successes=0, total=0, z=1.96)
        assert lower == 0.0
        assert upper == 0.0


class TestEstimateBotRate:
    """Test bot rate estimation with confidence intervals."""

    def test_basic_estimation(self):
        """Estimate with known counts should return correct point estimate."""
        scored_above = [("u", 4, []) for _ in range(30)]
        scored_below = [("u", 1, []) for _ in range(70)]
        scored = scored_above + scored_below
        result = estimate_bot_rate(scored, total_subscribers=1000, threshold=3)
        assert abs(result["point_estimate"] - 0.30) < 0.01
        assert result["ci_lower"] < 0.30
        assert result["ci_upper"] > 0.30
        assert result["margin_of_error"] > 0

    def test_empty_sample(self):
        """Empty sample should return zeros."""
        result = estimate_bot_rate([], total_subscribers=1000, threshold=3)
        assert result["point_estimate"] == 0.0


class TestSampleQualityReport:
    """Test sample quality and bias estimation."""

    def test_full_coverage(self):
        """100% coverage should report high quality."""
        report = sample_quality_report(
            enumerated=1000, total=1000,
            query_stats=[("a", 100, 100)] * 10,
        )
        assert report["coverage_pct"] == 100.0

    def test_low_coverage(self):
        """Low coverage should be reported accurately."""
        report = sample_quality_report(
            enumerated=100, total=10000,
            query_stats=[("a", 100, 100)],
        )
        assert report["coverage_pct"] == 1.0

    def test_zero_total(self):
        """Zero total subscribers should not divide by zero."""
        report = sample_quality_report(
            enumerated=0, total=0, query_stats=[],
        )
        assert report["coverage_pct"] == 0.0
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_statistics.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement statistics.py**

Create `tg_purge/statistics.py`:
```python
"""
Statistical sampling framework for bot rate estimation.

Provides confidence intervals (Wilson score), bias estimation, and
stratified extrapolation. All computations use stdlib math only.
If numpy is available (via [ml] extra), it is used for faster
computation but is not required.
"""

import math
from collections import Counter


def wilson_score_interval(successes, total, z=1.96):
    """Compute Wilson score confidence interval for a proportion.

    More accurate than the normal approximation for small samples or
    extreme proportions (near 0 or 1). Handles edge cases (0 or total
    successes) without returning nonsensical bounds.

    Args:
        successes: Number of observed successes (e.g., users above threshold).
        total: Total number of observations (e.g., total sampled users).
        z: Z-score for confidence level (default 1.96 = 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) as floats in [0, 1].
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total

    # Wilson score center and margin
    center = (p_hat + z2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        (p_hat * (1 - p_hat) / total) + (z2 / (4 * total * total))
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def estimate_bot_rate(scored_users, total_subscribers, threshold=3):
    """Estimate true bot rate with confidence interval.

    Uses Wilson score interval on the observed proportion of users
    scoring at or above the threshold.

    Args:
        scored_users: List of (user, score, reasons) tuples from scoring.
        total_subscribers: Total channel subscriber count.
        threshold: Minimum score to classify as bot for rate estimation.

    Returns:
        Dict with keys: point_estimate, ci_lower, ci_upper,
        margin_of_error, sample_size, flagged_count, total_subscribers.
    """
    n = len(scored_users)
    if n == 0:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "margin_of_error": 0.0,
            "sample_size": 0,
            "flagged_count": 0,
            "total_subscribers": total_subscribers,
        }

    flagged = sum(1 for _, score, _ in scored_users if score >= threshold)
    point_estimate = flagged / n
    ci_lower, ci_upper = wilson_score_interval(flagged, n)
    margin = (ci_upper - ci_lower) / 2

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": margin,
        "sample_size": n,
        "flagged_count": flagged,
        "total_subscribers": total_subscribers,
    }


def sample_quality_report(enumerated, total, query_stats):
    """Assess sample quality and estimate enumeration bias.

    Estimates bias by analyzing query yield concentration. If most
    results come from a few queries, the sample is biased toward
    whatever name patterns those queries match (usually Latin names).

    Uses a simplified Gini-like concentration metric rather than
    chi-squared (avoids scipy dependency).

    Args:
        enumerated: Number of unique users enumerated.
        total: Total channel subscriber count.
        query_stats: List of (query, result_count, new_count) tuples
            from enumeration.

    Returns:
        Dict with keys: coverage_pct, estimated_bias, query_efficiency.
    """
    if total == 0:
        return {
            "coverage_pct": 0.0,
            "estimated_bias": "unknown",
            "query_efficiency": 0.0,
        }

    coverage = (enumerated / total) * 100

    if query_stats:
        yields = [new for _, _, new in query_stats]
        total_yield = sum(yields)
        if total_yield > 0:
            # Gini-like concentration: if top 20% of queries produce
            # >80% of results, bias is high
            sorted_yields = sorted(yields, reverse=True)
            top_20_count = max(1, len(sorted_yields) // 5)
            top_20_yield = sum(sorted_yields[:top_20_count])
            concentration = top_20_yield / total_yield

            if concentration > 0.8:
                bias = "high (name-pattern concentration detected)"
            elif concentration > 0.6:
                bias = "moderate (some name-pattern concentration)"
            else:
                bias = "low (well-distributed across query patterns)"
        else:
            bias = "unknown (no results)"

        efficiency = total_yield / len(query_stats) if query_stats else 0.0
    else:
        bias = "unknown"
        efficiency = 0.0

    return {
        "coverage_pct": coverage,
        "estimated_bias": bias,
        "query_efficiency": efficiency,
    }


def format_stats_summary(bot_rate_result, quality_report):
    """Format statistical results for terminal display.

    Args:
        bot_rate_result: Output of estimate_bot_rate().
        quality_report: Output of sample_quality_report().

    Returns:
        Multi-line string ready for print().
    """
    lines = []
    lines.append("")
    lines.append("\u2500" * 80)
    lines.append("STATISTICAL SUMMARY")
    lines.append("\u2500" * 80)

    pe = bot_rate_result["point_estimate"] * 100
    ci_l = bot_rate_result["ci_lower"] * 100
    ci_u = bot_rate_result["ci_upper"] * 100
    lines.append(f"  Bot rate estimate: {pe:.1f}% (95% CI: {ci_l:.1f}% \u2013 {ci_u:.1f}%)")
    lines.append(
        f"  Sample coverage: {bot_rate_result['sample_size']:,} / "
        f"{bot_rate_result['total_subscribers']:,} "
        f"({quality_report['coverage_pct']:.1f}%)"
    )
    lines.append(f"  Sampling bias: {quality_report['estimated_bias']}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_statistics.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tg_purge/statistics.py tests/test_statistics.py
git commit -m "feat: add statistical sampling framework"
```

---

## Chunk 2: ML Pipeline

### Task 4: Labeling Infrastructure

**Files:**
- Create: `tg_purge/labeling.py`
- Create: `tests/test_labeling.py`

- [ ] **Step 1: Write tests**

Create `tests/test_labeling.py`:
```python
"""Tests for labeling infrastructure.

Tests cover bootstrap label assignment, label persistence (save/load),
and label statistics. Uses MockUser fixtures — no Telegram API calls.
"""

import json
import os
from datetime import datetime, timezone
from tg_purge.labeling import (
    bootstrap_labels,
    save_labels,
    load_labels,
    label_stats,
)


class TestBootstrapLabels:
    """Test heuristic-based weak label assignment."""

    def test_high_score_labeled_bot(self, make_user):
        """Users with score >= 4 should be labeled bot."""
        user = make_user(deleted=True, first_name=None, last_name=None,
                         username=None, photo=False, status=None)
        labels = bootstrap_labels(
            users={user.id: user},
            scored={user.id: (5, ["deleted_account(+5)"])},
        )
        assert labels[user.id]["label"] == "bot"
        assert labels[user.id]["source"] == "heuristic_bootstrap"

    def test_zero_score_labeled_human(self, clean_user):
        """Users with score 0 should be labeled human."""
        labels = bootstrap_labels(
            users={clean_user.id: clean_user},
            scored={clean_user.id: (0, [])},
        )
        assert labels[clean_user.id]["label"] == "human"

    def test_mid_score_unlabeled(self, make_user):
        """Users with score 1-3 should be labeled unlabeled."""
        user = make_user(photo=False, username=None, first_name="A",
                         last_name=None, status=None)
        labels = bootstrap_labels(
            users={user.id: user},
            scored={user.id: (2, ["no_photo(+1)", "no_username(+1)"])},
        )
        assert labels[user.id]["label"] == "unlabeled"


class TestLabelPersistence:
    """Test save/load of label files."""

    def test_roundtrip(self, tmp_path):
        """Labels should survive a save/load roundtrip."""
        labels = {
            123: {"label": "bot", "source": "human",
                  "timestamp": "2026-03-11T00:00:00Z"},
            456: {"label": "human", "source": "heuristic_bootstrap",
                  "timestamp": "2026-03-11T00:00:00Z"},
        }
        path = tmp_path / "labels.json"
        save_labels(labels, "@test_channel", str(path))
        loaded = load_labels(str(path))
        assert loaded["labels"][123]["label"] == "bot"
        assert loaded["labels"][456]["label"] == "human"
        assert loaded["channel"] == "@test_channel"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading from nonexistent file should return empty structure."""
        path = tmp_path / "nonexistent.json"
        loaded = load_labels(str(path))
        assert loaded["labels"] == {}

    def test_file_permissions(self, tmp_path):
        """Saved label files should have restrictive permissions (chmod 600)."""
        labels = {123: {"label": "bot", "source": "human",
                        "timestamp": "2026-03-11T00:00:00Z"}}
        path = tmp_path / "labels.json"
        save_labels(labels, "@test", str(path))
        mode = oct(os.stat(str(path)).st_mode)[-3:]
        assert mode == "600"


class TestLabelStats:
    """Test label statistics computation."""

    def test_counts(self):
        """Stats should correctly count labels by type."""
        labels = {
            1: {"label": "bot", "source": "human"},
            2: {"label": "human", "source": "human"},
            3: {"label": "bot", "source": "heuristic_bootstrap"},
            4: {"label": "unlabeled", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["bot"] == 2
        assert stats["human"] == 1
        assert stats["unlabeled"] == 1
        assert stats["total"] == 4
        assert stats["human_labeled"] == 2
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: FAIL

- [ ] **Step 3: Implement labeling.py**

Create `tg_purge/labeling.py`:
```python
"""
Labeling infrastructure for ML training data.

Handles bootstrap label assignment (weak labels from heuristic scores),
label persistence to JSON, and label statistics. The interactive active
learning loop lives in commands/label.py — this module is the data layer.

Label files contain real Telegram user IDs and are PII-sensitive.
File permissions are set to 600 (owner read/write only), consistent
with session file handling in client.py.
"""

import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def bootstrap_labels(users, scored):
    """Assign weak labels based on heuristic scores.

    Score >= 4: labeled bot (high confidence heuristic)
    Score == 0: labeled human (high confidence clean)
    Score 1-3: labeled unlabeled (ambiguous, needs human review)

    Args:
        users: Dict of user_id -> User object.
        scored: Dict of user_id -> (score, reasons) from score_user().

    Returns:
        Dict of user_id -> {label, source, timestamp}
    """
    now = datetime.now(timezone.utc).isoformat()
    labels = {}

    for user_id in users:
        score, _ = scored.get(user_id, (0, []))
        if score >= 4:
            label = "bot"
        elif score == 0:
            label = "human"
        else:
            label = "unlabeled"

        labels[user_id] = {
            "label": label,
            "source": "heuristic_bootstrap",
            "timestamp": now,
        }

    return labels


def save_labels(labels, channel, path):
    """Save labels to JSON file with restrictive permissions.

    Args:
        labels: Dict of user_id -> {label, source, timestamp}.
        channel: Channel identifier (e.g., @leviathan_news).
        path: File path for the JSON output.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Set directory to owner-only (700)
    try:
        file_path.parent.chmod(stat.S_IRWXU)
    except OSError:
        pass

    data = {
        "channel": channel,
        "version": 1,
        "labels": {str(uid): info for uid, info in labels.items()},
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Set file to owner read/write only (600) — PII-sensitive content
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def load_labels(path):
    """Load labels from JSON file.

    Args:
        path: File path to the labels JSON.

    Returns:
        Dict with keys: channel, version, labels.
        Returns empty structure if file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {"channel": "", "version": 1, "labels": {}}

    with open(path, "r") as f:
        data = json.load(f)

    # Convert string keys back to int user IDs
    data["labels"] = {int(uid): info for uid, info in data.get("labels", {}).items()}
    return data


def label_stats(labels):
    """Compute statistics on a label set.

    Args:
        labels: Dict of user_id -> {label, source, ...}.

    Returns:
        Dict with counts: bot, human, unlabeled, total, human_labeled.
    """
    label_counts = Counter(info["label"] for info in labels.values())
    human_labeled = sum(
        1 for info in labels.values() if info.get("source") == "human"
    )

    return {
        "bot": label_counts.get("bot", 0),
        "human": label_counts.get("human", 0),
        "unlabeled": label_counts.get("unlabeled", 0),
        "total": len(labels),
        "human_labeled": human_labeled,
    }
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tg_purge/labeling.py tests/test_labeling.py
git commit -m "feat: add labeling infrastructure for ML training data"
```

---

### Task 5: ML Training and Inference

**Files:**
- Create: `tg_purge/ml.py`
- Create: `tests/test_ml.py`

See spec Section 3. Tests require `[ml]` deps and are skipped if unavailable. Model files use LightGBM/XGBoost native format. The sklearn fallback uses joblib (documented tradeoff — no safe native format exists for sklearn). Only load models from trusted sources.

- [ ] **Step 1: Write tests**

Create `tests/test_ml.py` — tests for `ml_available()`, `train_model()`, `predict()`, and `load_model_metadata()`. Skip all tests if sklearn is not installed. Test training with 30+30 minimal feature vectors (bot-like and human-like), verify model file creation, metadata structure, and prediction output format. Test failure cases: insufficient data (<10 samples), single class only.

- [ ] **Step 2: Implement ml.py**

Create `tg_purge/ml.py` — `ml_available()` checks sklearn import. `_get_available_models()` probes LightGBM, XGBoost, sklearn in order. `train_model()` trains all available, picks best F1, saves native format + metadata JSON. `predict()` loads model, converts features to numpy array, returns probability + label dicts. `load_model_metadata()` reads the JSON companion file.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_ml.py -v`
Expected: all pass (or all skipped if no ML deps).

- [ ] **Step 4: Commit**

```bash
git add tg_purge/ml.py tests/test_ml.py
git commit -m "feat: add ML training and inference pipeline"
```

---

### Task 6: ML CLI Commands

**Files:**
- Create: `tg_purge/commands/label.py`
- Create: `tg_purge/commands/ml_cmd.py`
- Modify: `tg_purge/cli.py` (add `label`, `ml` subcommands + `--scoring`/`--stats` flags)
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write CLI parser tests for new commands**

Add `TestLabelCommand` and `TestMLCommand` classes to `tests/test_cli.py`. Test that `label --channel @test --bootstrap` parses correctly. Test `ml train`, `ml info`, `ml export-features --output out.json` parse correctly.

- [ ] **Step 2: Add subcommands to cli.py**

Add `label` subparser (with `--bootstrap`, `--strategy` flags) and `ml` subparser (with `train`, `info`, `export-features` sub-actions). Add `--scoring` and `--stats` to `_add_common_args()`. Add dispatch cases to `main()`.

- [ ] **Step 3: Create commands/label.py**

Bootstrap mode: enumerate, score, assign weak labels, save. Interactive mode: stub with "not yet implemented" message directing user to --bootstrap first.

- [ ] **Step 4: Create commands/ml_cmd.py**

`train`: load labels, validate, train model. `info`: find and display model metadata. `export-features`: stub for future implementation.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tg_purge/cli.py tg_purge/commands/label.py tg_purge/commands/ml_cmd.py tests/test_cli.py
git commit -m "feat: add label and ml CLI commands"
```

---

## Chunk 3: Coverage Expansion

### Task 7: Enumeration Optimizations

**Files:**
- Modify: `tg_purge/enumeration.py`
- Modify: `tests/test_enumeration.py`

- [ ] **Step 1: Add YieldTracker class and tests**

Add `YieldTracker` to `enumeration.py` after `RESULT_CAP`. It records per-prefix result counts and exposes `should_expand(prefix)` returning True only if prefix hit the cap. Add tests in `test_enumeration.py`.

- [ ] **Step 2: Run tests and commit**

```bash
git add tg_purge/enumeration.py tests/test_enumeration.py
git commit -m "feat: add yield tracker for adaptive prefix expansion"
```

---

### Task 8: Collectors Package

**Files:**
- Create: `tg_purge/collectors/__init__.py`
- Create: `tg_purge/collectors/base.py` (CollectorResult dataclass + merge)
- Create: `tg_purge/collectors/api.py` (wraps enumerate_subscribers)
- Create: `tg_purge/collectors/message_authors.py` (GetHistoryRequest)
- Create: `tg_purge/collectors/admin_log.py` (GetAdminLogRequest)
- Create: `tests/test_collectors.py`

- [ ] **Step 1: Write tests for CollectorResult merge**

Test empty results, deduplication on merge, first-seen-wins behavior.

- [ ] **Step 2: Implement base.py, api.py, message_authors.py, admin_log.py**

- [ ] **Step 3: Run tests and commit**

```bash
git add tg_purge/collectors/ tests/test_collectors.py
git commit -m "feat: add pluggable collector framework"
```

---

## Chunk 4: Cross-Channel and Integration

### Task 9: Cross-Channel Cohort Detection

**Files:**
- Create: `tg_purge/cross_channel.py`
- Create: `tests/test_cross_channel.py`
- Modify: `tg_purge/scoring.py` (add `bot_cohort_member` signal + `**kwargs`)

- [ ] **Step 1: Write tests for find_cohorts and score_cohort**

Test: no overlap returns no cohorts, large overlap (60 users in 3 channels) detected, small overlap (<50) ignored. Test: coordinated cohort (tight joins, similar profiles) scored as suspicious, organic cohort (spread joins, diverse profiles) not flagged.

- [ ] **Step 2: Implement cross_channel.py**

`find_cohorts()`: build user->channels mapping, group by exact channel set, filter by min size. `score_cohort()`: compute join time stddev, profile similarity (fraction of features >80% identical), suspicion decision requires ALL conditions met.

- [ ] **Step 3: Add bot_cohort_member to scoring.py**

Add `bot_cohort_member: int = 2` to ScoringConfig. Add `**kwargs` to `score_user()` signature. Add cohort check after spike_join block using `kwargs.get("cohort_data")`.

- [ ] **Step 4: Run tests and commit**

```bash
git add tg_purge/cross_channel.py tests/test_cross_channel.py tg_purge/scoring.py
git commit -m "feat: add cross-channel cohort detection and bot_cohort_member signal"
```

---

### Task 10: --stats Integration

**Files:**
- Modify: `tg_purge/commands/analyze.py`

- [ ] **Step 1: Add stats output when --stats flag is set**

Import `format_stats_summary` from `statistics`. Call after existing output at end of `run()` when `args.stats` is True and total_subscribers is known.

- [ ] **Step 2: Run full test suite and commit**

```bash
git add tg_purge/commands/analyze.py
git commit -m "feat: integrate --stats output into analyze command"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all tests pass, zero failures.

- [ ] **Step 2: Verify CLI help renders**

Run: `python -m tg_purge --help && python -m tg_purge label --help && python -m tg_purge ml --help`

- [ ] **Step 3: Verify imports without ML deps**

Run: `python -c "from tg_purge.features import extract_features; print('OK')"`
Run: `python -c "from tg_purge.statistics import estimate_bot_rate; print('OK')"`
