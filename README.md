# tg-bot-detector

Heuristic + ML bot detection and purge toolkit for Telegram broadcast channel subscribers.

Analyzes your channel's subscriber list using profile-based scoring heuristics, temporal clustering, and a LightGBM classifier to identify likely bot accounts. Built on Telethon (MTProto) for full subscriber enumeration and Telegram Bot API for high-speed purging.

**Proven on @leviathan_news**: detected ~60,000 fake subscribers out of 86,000 (subscriber count dropped from 86K to 43K through natural bot exodus + active purging).

## Quick Start

```bash
# Install
pip install -e .

# Install with ML dependencies
pip install -e ".[ml]"

# Set credentials (get from https://my.telegram.org)
export TG_PURGE_API_ID="your_api_id"
export TG_PURGE_API_HASH="your_api_hash"

# First run will prompt for phone number + verification code
tg-purge analyze --channel @your_channel --strategy minimal
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/leviathan-news/tg-bot-detector.git
cd tg-bot-detector
pip install -e .

# For TOML config support on Python <3.11:
pip install -e ".[toml]"

# For ML pipeline (scikit-learn, LightGBM, XGBoost):
pip install -e ".[ml]"
```

## Configuration

Set credentials via environment variables:

```bash
export TG_PURGE_API_ID="your_api_id"
export TG_PURGE_API_HASH="your_api_hash"
```

Or use a TOML config file (see `examples/config.example.toml`):

```bash
cp examples/config.example.toml config.toml
# Edit config.toml with your values
```

## CLI Reference

### analyze

Multi-round subscriber analysis. Three phases: self-identified bots, recently active users, and search-based sampling.

```bash
tg-purge analyze --channel @foo
tg-purge analyze --channel @foo --strategy minimal  # faster, fewer queries
```

### candidates

Generate scored candidate lists for offline review. Supports heuristic, ML, or hybrid scoring modes.

```bash
# Heuristic only (default)
tg-purge candidates --channel @foo --threshold 4 --output candidates.csv

# Hybrid: heuristic + ML (requires trained model in models/)
tg-purge candidates --channel @foo --threshold 0 --scoring hybrid --output hybrid.csv

# With safelist protection
tg-purge candidates --channel @foo --threshold 4 --safelist registry/safelist.csv
```

Ctrl+C during enumeration saves partial results automatically. Also handles SIGHUP/SIGTERM for tmux compatibility.

### join-dates

Join date clustering and spike detection. Identifies bulk-subscription events.

```bash
tg-purge join-dates --channel @foo
tg-purge join-dates --channel @foo --top-days 50
```

### spike

Deep-dive into a specific time window. Compares spike subscribers against a control group.

```bash
tg-purge spike --channel @foo --start "2025-11-09T06:00Z" --end "2025-11-09T07:00Z"
```

### validate

Score known-good users to measure the false positive rate at each threshold.

```bash
tg-purge validate --channel @foo --known-users my_users.csv
```

### label

Bootstrap weak labels from heuristic scores or view label stats.

```bash
tg-purge label --bootstrap --channel @foo
tg-purge label --stats
```

### ml

ML model management: train, inspect, or export features.

```bash
tg-purge ml train --channel @foo
tg-purge ml info
tg-purge ml export-features --channel @foo
```

### registry

Local bot registry management. Generate, add to, or query a local registry of flagged user IDs.

```bash
tg-purge registry generate --channel @foo --threshold 4
tg-purge registry add --ids-file flagged.txt
tg-purge registry check --user-id 123456789
```

## Detection Layers

### Layer 1: Heuristic Scoring

Each subscriber is scored based on Telegram profile attributes. Higher = more likely bot.

| Signal | Weight | Description |
|--------|--------|-------------|
| Deleted account | +5 | Account has been deleted |
| Scam/fake flag | +5 | Telegram-applied flag |
| Bot API account | +3 | Self-identified bot |
| Restricted | +2 | Telegram-restricted account |
| No activity status | +2 | Never been seen online |
| Offline >1 year | +2 | Abandoned account |
| Airdrop farmer | +2 | 2+ airdrop project tokens in display name |
| Spike join | +2 | Joined during a detected bulk-subscription window |
| No photo/username | +1 each | Sparse profile |
| Photo DC 1 | +1 | Profile photo on bot-heavy datacenter |
| Short/digit/mixed name | +1 each | Generated-looking name |
| Premium subscriber | -2 | Legitimacy signal (but gameable — bot farms buy Premium) |
| Emoji status | -1 | Premium feature usage |
| Photo DC 5 | -1 | Human-dominant datacenter |
| Video avatar | -1 | 0% prevalence on bots |
| Custom color | -1 | 0.1% on bots vs 14.1% on humans |

### Layer 2: Temporal Clustering

Sliding-window algorithm detects bulk-subscription spikes from join-date history. Users who joined during a detected spike get +2 to their heuristic score.

### Layer 3: Machine Learning

LightGBM classifier trained on 35K+ labeled accounts. Extracts 51 features per user including profile attributes, name analysis, photo metadata, and join timing. The ML model catches bots that evade individual heuristic signals.

Training uses ground-truth labels from:
- Departed bot accounts (captured via exit monitoring during mass exodus events)
- Human-reviewed profiles (1,132 manual reviews)
- High-confidence heuristic labels (bootstrap)

Status features are neutralized during training to prevent data leakage from exit monitoring (bots show "online" when the farm activates them to leave).

## Purge Scripts

### Bot API (recommended)

```bash
python scripts/purge_bot_api.py \
    --input output/candidates.csv \
    --chat-id -100XXXXXXXXXX \
    --bot-token "YOUR_BOT_TOKEN"
```

Requires a bot that is admin in the channel with ban permissions. Much faster than user API (~20 bans/s vs 300-then-flood-wait).

### User MTProto API (fallback)

```bash
python scripts/purge_batch.py \
    --input output/candidates.csv \
    --channel @your_channel \
    --chunk-size 200 --chunk-cooldown 300
```

Chunked banning with configurable cooldowns to avoid FloodWait. Both scripts support:
- **Resume**: JSONL progress files for interrupted runs
- **Graceful shutdown**: Ctrl+C / SIGHUP / SIGTERM save progress
- **Safelist**: Exclude protected user IDs

## Limitations

- **200-result cap**: Each Telegram query returns at most 200 results
- **Sampling bias**: Search-based enumeration misses accounts with no name or unusual names
- **Privacy false positives**: Users who hide online status score +2 automatically
- **Score-0 bots**: Sophisticated bots with full profiles are invisible to profile-based detection
- **Premium is gameable**: Bot farms buy Telegram Premium to evade the -2 bonus
- **Channel-specific**: Developed for broadcast channels, not groups

## Testing

```bash
python -m pytest tests/ -v
```

453 unit tests covering scoring logic, CLI parsing, config loading, enumeration, clustering, ML features, labeling, statistics, cross-channel detection, and collectors. No Telegram API calls in tests.

## License

MIT
