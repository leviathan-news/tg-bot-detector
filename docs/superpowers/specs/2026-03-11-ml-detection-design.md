# ML-Based Bot Detection & Coverage Expansion — Design Spec

**Date:** 2026-03-11
**Branch:** TBD (new branch from main, porting current `fix/v0.1.0-review-findings` changes)
**Status:** Approved

## Goal

Evolve tg-bot-detector from a heuristic-only scorer to a hybrid detection system with ML classification, expanded enumeration coverage, cross-channel cohort detection, and statistical rigor — while preserving the existing heuristic workflow as the default.

## Approach

Dual-track development:
- **Track A (Data):** Coverage expansion + labeling pipeline
- **Track B (ML):** Feature engineering + model architecture + training pipeline (developed against weak labels)
- **Converge:** Retrain on real labeled data when Track A delivers
- **Then:** Cross-channel cohort detection, statistical sampling, model distribution

## Section 1: Feature Engineering

**New module:** `tg_purge/features.py`

**Function:** `extract_features(user, join_date=None, spike_windows=None, cohort_data=None) -> dict[str, float]`

Outputs a flat dict of ~29 numeric features:

| Category | Features |
|----------|----------|
| Account flags | `is_deleted`, `is_bot`, `is_scam`, `is_fake`, `is_restricted`, `is_premium`, `has_emoji_status` (all 0/1) |
| Profile completeness | `has_photo`, `has_username`, `has_last_name`, `first_name_length`, `name_digit_ratio`, `script_count` (int: number of distinct script families detected — Latin, Cyrillic, Arabic, CJK; value >= 2 indicates mixed scripts) |
| Activity | `status_type` (one-hot: empty/online/recently/last_week/last_month/offline), `days_since_last_seen` (continuous, -1 if unknown) |
| Temporal | `is_spike_join` (0/1), `days_since_join` (continuous), `join_hour_utc` (0-23), `join_day_of_week` (0-6) |
| Cohort (Section 5) | `is_cohort_member` (0/1), `cohort_size` (0 if not in cohort), `cohort_join_spread_hours` (0.0 if N/A), `cohort_profile_similarity` (0.0 if N/A) |
| Heuristic score | `heuristic_score` (existing integer score — kept as a feature) |

Design constraints:
- `extract_features()` is a parallel path to `score_user()`, not a replacement
- Same testability requirement: must work without Telethon imported (uses `type(status).__name__` pattern)
- Returns plain dict — no numpy/pandas dependency at extraction time

## Section 2: Labeling Infrastructure

**New module:** `tg_purge/labeling.py`
**New CLI command:** `tg-purge label`

### Bootstrap Phase

1. Run enumeration on target channel
2. Auto-assign weak labels: score >= 4 -> `bot`, score 0 -> `human`, scores 1-3 -> `unlabeled`
3. Save to `datasets/<channel>/labels.json`

### Active Learning Loop (`tg-purge label --channel @foo`)

1. Load current labeled + unlabeled data
2. Train a lightweight sklearn `RandomForestClassifier` inline (not the full Section 3 pipeline — this is a simple uncertainty estimator, not production training). Section 3's full pipeline is only used for `tg-purge ml train`.
3. Select ~20 most uncertain predictions (probability closest to 0.5)
4. Present interactively in terminal:
   ```
   [1/20] ID: 123456789 | @username | "Ivan K" | score: 2 | no_photo, no_status_ever
          Status: offline 90d | Joined: 2025-03-15 03:22 UTC (spike window)
          Model confidence: 0.48
          Label: [b]ot / [h]uman / [s]kip?
   ```
5. Save human labels, retrain, repeat

### Data Format (`datasets/<channel>/labels.json`)

```json
{
  "channel": "@leviathan_news",
  "version": 1,
  "labels": {
    "123456789": {"label": "bot", "source": "human", "timestamp": "..."},
    "987654321": {"label": "human", "source": "heuristic_bootstrap", "timestamp": "..."}
  }
}
```

`datasets/` directory is gitignored (contains user IDs). Label files contain real Telegram user IDs and must be treated as PII-sensitive — same protection as session files: directory `chmod 700`, files `chmod 600` (best-effort). Never share raw label files; use `ml export-features` for anonymized contribution.

## Section 3: ML Training & Inference Pipeline

**New module:** `tg_purge/ml.py`

### Training (`tg-purge ml train`)

1. Load labels from `datasets/<channel>/labels.json`
2. Load/extract feature vectors
3. Stratified 80/20 train/test split (preserve class ratio)
4. Train available models, pick best by F1-score. Availability is determined by import check at runtime:
   - LightGBM (`lightgbm.LGBMClassifier`) — preferred, tried first
   - XGBoost (`xgboost.XGBClassifier`) — tried if LightGBM unavailable
   - Sklearn Random Forest — always available (sklearn is the minimum `[ml]` dep)
   - If only sklearn is installed, only Random Forest is trained (no error). Training always succeeds if at least sklearn is present.
5. Handle class imbalance via `scale_pos_weight` (auto-calculated from label ratio)
6. Output: model file + metrics report + feature importance ranking

### Model Artifact

Metadata (`models/v<N>-<channel>.json`):
```json
{
  "version": 2,
  "trained_on": ["@leviathan_news"],
  "algorithm": "lightgbm",
  "n_samples": {"bot": 342, "human": 158},
  "metrics": {"f1": 0.89, "precision": 0.92, "recall": 0.86, "auc_roc": 0.94},
  "feature_names": ["is_deleted", "is_bot", "..."],
  "threshold": 0.5,
  "created": "2026-03-11T..."
}
```

Binary: `models/v<N>-<channel>.model` (LightGBM/XGBoost native format — no arbitrary code execution risk).

### Inference

```python
def predict(users, model_path=None) -> list[dict]:
    """Returns per-user: {user_id, probability, label, heuristic_score, top_features}"""
```

- Lazy model loading (cached after first load)
- Graceful fallback to heuristic-only if no model or ML deps missing
- `top_features`: top 3 features driving prediction (LightGBM built-in feature contribution)

### CLI Scoring Modes

The `--scoring` flag is accepted by: `analyze`, `candidates`, `join-dates`, `spike`, and `registry generate` (all commands that score users). `validate` always uses heuristic (it measures heuristic FP rate by design).

- `--scoring heuristic` (default, current behavior)
- `--scoring ml` (ML probability, requires trained model)
- `--scoring hybrid` (both side by side)

## Section 4: Coverage Expansion

Note: "L1" and "L3" are option labels from the design brainstorming (the user selected options 1 and 3 out of three proposed approaches). They are not sequential section numbers — there is no missing "L2". Same applies to "W3" in Section 7.

### API Optimization (L1)

Extend `tg_purge/enumeration.py`:

1. **Smarter prefix tree** — prioritize prefixes by expected yield. Track which hit the 200 cap and focus expansion there. Skip zero-yield prefixes.
2. **Adaptive depth** — increase `max_depth` only on high-yield prefixes. Low-yield stops at depth 1.
3. **Parallel query batching** — `asyncio.gather` with semaphore (3 concurrent, 0.5s spacing). Current implementation is sequential.
4. **Filter combinations** — enumerate via `ChannelParticipantsBots`, `ChannelParticipantsRecent`, `ChannelParticipantsAdmins`, `ChannelParticipantsKicked`, `ChannelParticipantsBanned` as separate passes.

### Hybrid External Sources (L3)

**New package:** `tg_purge/collectors/`

| Collector | Source | Data |
|-----------|--------|------|
| `message_authors.py` | `GetHistoryRequest` | Unique user IDs from message senders/forwarders |
| `admin_log.py` | `GetAdminLogRequest` | Join/leave events, ban history (admin required) |
| `profile_enricher.py` | `GetFullUserRequest` (MTProto) | Bio text, profile photo count, common chats count. Uses the authenticated MTProto API (not HTTP scraping) — stays within Telegram's API ToS. Rate-limited same as other API calls via `--delay`. |
| `cross_reference.py` | Multiple channels | Feeds into cohort detection (Section 5) |

**Unified pipeline:**
```python
async def collect(client, channel, collectors=["api", "messages", "admin_log"], ...):
    """Merges results from multiple collectors, deduplicates by user_id"""
```

Each collector returns a `CollectorResult` dataclass. `enumerate_subscribers()` becomes the `api` collector. Existing commands gain `--collectors` flag.

## Section 5: Cross-Channel Cohort Detection

**New module:** `tg_purge/cross_channel.py`
**New CLI command:** `tg-purge scan`

### Permission Model

`tg-purge scan` requires the authenticated user to be a **member** of each listed channel (search-based enumeration does not require admin). The command:
- Checks membership before scanning each channel
- Skips channels where the user is not a member (logs warning to stderr, continues with remaining channels)
- Never auto-joins channels — joining is a visible action the user must do manually
- Requires at least 2 accessible channels to produce meaningful cohort analysis; errors if fewer are reachable

### The Signal

Not "user X appears in 5 channels" (that is just an active user). The signal is coordinated cohort behavior: N users appearing together across K channels with correlated join times and similar profiles.

### Detection Algorithm

1. **Co-occurrence matrix** — for each user pair across scanned channels, count shared channels
2. **Cluster extraction** — find groups of N+ users (default 50) sharing K+ channels (default 3) via set intersection
3. **Cohort suspicion scoring:**

| Factor | Measures | Logic |
|--------|----------|-------|
| `cohort_size` | Users moving together | Organic groups rarely exceed ~30 |
| `join_time_correlation` | Coordinated joins? | Stddev of join timestamps across shared channels — low = coordinated |
| `profile_similarity` | Generated profiles? | Intra-cohort similarity of no_photo, no_username, name_length |
| `channel_diversity` | Unrelated channels? | Bot farms hit random channels; real communities cluster by topic |

4. **Individual attribution** — score boost only if cohort passes ALL of:
   - Size >= 50
   - Join time stddev < 48 hours across shared channels
   - Profile similarity above threshold
   - At least 3 shared channels

### Heuristic Signal

`bot_cohort_member` (+2): belongs to a statistically suspicious coordinated group.

### What It Does NOT Flag

- Individual users in many channels (no cohort pattern)
- Small organic friend groups
- Tech enthusiasts who independently join popular channels

### ML Features

`is_cohort_member` (0/1), `cohort_size`, `cohort_join_spread_hours`, `cohort_profile_similarity` — the model learns appropriate weights.

### Data Storage

`datasets/cross_channel/index.json` — user ID to channel mapping. Gitignored, never distributed. Published models only see numeric features, never raw mappings.

### Privacy Constraint

Cross-channel data contains user IDs mapped to channels. The exported feature vectors (Section 7) only include the numeric `cohort_*` features, never raw IDs or channel names.

## Section 6: Statistical Sampling Framework

**New module:** `tg_purge/statistics.py`

### Core Functions

```python
def estimate_bot_rate(scored_users, total_subscribers, threshold=3):
    """Returns: point_estimate, confidence_interval_95, margin_of_error"""

def sample_quality_report(enumerated, total, query_stats):
    """Returns: coverage_pct, estimated_bias, representativeness_score"""
```

### Capabilities

1. **Confidence intervals** — Wilson score interval for true bot proportion given sample size
2. **Bias estimation** — compare name-script distribution of sample vs expected (based on channel language/region or uniform prior)
3. **Stratified extrapolation** — per-score-bucket extrapolation instead of naive ratio, reducing variance from the score-2 cliff

### CLI Integration

All analysis commands gain `--stats` flag:
```
Bot rate estimate: 34.2% (95% CI: 31.8% - 36.6%)
Sample coverage: 5,142 / 48,000 (10.7%)
Sampling bias: moderate (Latin name over-representation detected)
```

No new required dependencies. Confidence intervals (Wilson score) and stratified extrapolation use stdlib `math` only. Bias estimation uses a simplified chi-squared approximation implemented inline — no scipy needed. If the `[ml]` extras are installed, `numpy` is used for faster computation but is not required.

## Section 7: Model Distribution

(W3 is a brainstorming option label — see Section 4 note.)

### Publishing

1. Train on @leviathan_news (and future channels)
2. Artifact: `.model` file + metadata JSON
3. Published via GitHub releases: `tg-bot-detector-model-v<N>.tar.gz`
4. Users: `tg-purge ml download` or manual placement in `~/.tg_purge/models/`

### CLI Commands

```
tg-purge ml download          # fetch latest from GitHub releases
tg-purge ml info              # show loaded model metadata
tg-purge ml train             # train from local labels
tg-purge ml export-features   # export anonymized vectors for contribution
```

### Federated Contribution

`tg-purge ml export-features --channel @foo --output features.json` exports:
- Numeric feature vectors only (no user IDs, names, usernames)
- Weak labels (heuristic score bucket: 0-1, 2-3, 4+)
- Channel hash (salted, not actual channel name)

**Not exported:** user IDs, names, usernames, PII, raw channel identifiers, session data, join timestamps (only relative features like `days_since_join`).

Manual trust-based contribution (email/PR). No automated telemetry or phone-home.

### Versioning

Models versioned independently from package. `predict()` checks feature compatibility and warns on mismatch.

## New Dependencies

**Optional** (`pip install tg-bot-detector[ml]`):
- `lightgbm`
- `xgboost`
- `scikit-learn`
- `numpy`
- `pandas`

Core heuristic workflow remains dependency-free (Telethon only).

## Data Directories (All Gitignored)

- `datasets/<channel>/` — labels, cached features, user data
- `datasets/cross_channel/` — cohort co-occurrence mapping
- `models/` — trained model artifacts

**Implementation prerequisite:** Add `datasets/`, `models/` to `.gitignore` before creating any of these directories. These entries do not currently exist in the project's `.gitignore`.

**File permissions:** All files in `datasets/` and `models/` are created with `chmod 600` (owner read+write only), directories with `chmod 700` (owner only), consistent with existing session file handling in `client.py`.

## What Stays Untouched

- `score_user()` — remains as-is, becomes one ML feature
- All existing 154 tests — nothing breaks
- Heuristic-only workflow — remains default, no ML deps required
- Session/credential security model — unchanged
