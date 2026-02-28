# tg-bot-detector

Heuristic bot detection toolkit for Telegram broadcast channel subscribers.

Analyzes your channel's subscriber list using profile-based scoring heuristics to identify likely bot accounts. Built on Telethon (MTProto) for full subscriber enumeration.

**v1 scope**: Analysis and identification only. No destructive operations (banning/purging). The CLI generates scored candidate lists for offline human review.

## Quick Start

```bash
# Install
pip install -e .

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
tg-purge validate --channel @foo --known-users my_users.json
```

Input format: CSV with `user_id` or `username` columns, JSON list of objects, or plain text (one ID per line).

### candidates

Generate scored candidate lists for offline review.

```bash
tg-purge candidates --channel @foo --threshold 4 --output candidates.csv
tg-purge candidates --channel @foo --threshold 3 --safelist protected.csv
```

### registry

Local bot registry management. Generate, add to, or query a local registry of flagged user IDs.

```bash
tg-purge registry generate --channel @foo --threshold 4 --output registry/known_bots.json
tg-purge registry add --ids-file flagged.txt
tg-purge registry check --user-id 123456789
```

## Scoring

Each subscriber is scored based on Telegram profile attributes. Higher = more likely bot.

| Signal | Weight | Description |
|--------|--------|-------------|
| Deleted account | +5 | Account has been deleted |
| Scam/fake flag | +5 | Telegram-applied flag |
| Bot API account | +3 | Self-identified bot |
| Restricted | +2 | Telegram-restricted account |
| No activity status | +2 | Never been seen online |
| Offline >1 year | +2 | Abandoned account |
| No photo/username | +1 each | Sparse profile |
| Short/digit/mixed name | +1 each | Generated-looking name |
| Premium subscriber | -2 | Strong legitimacy signal |
| Emoji status | -1 | Premium feature usage |

See `docs/scoring-methodology.md` for the full breakdown, rationale, and recommended thresholds.

## Limitations

- **Heuristic-only**: Scores are probabilistic signals, not definitive classifications
- **200-result cap**: Each Telegram query returns at most 200 results
- **~10K ceiling**: Server-side enumeration tops out around 10K participants
- **Privacy false positives**: Users who hide their online status score +2 automatically
- **Channel-specific**: Developed for broadcast channels, not groups

## Roadmap

### v2 (planned)

- **Purge execution**: `ban_users` admin permission, `EditBannedRequest` with timed unban, checkpoint/resume for partial failures, audit logging
- **Safelist integration**: Protected user IDs never banned regardless of score
- **Confirmation gate**: Typed channel name + threshold to prevent accidental execution

## License

MIT
