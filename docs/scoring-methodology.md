# Scoring Methodology

## Overview

Each Telegram user is scored based on heuristic signals extracted from their public profile. Higher scores indicate greater likelihood of being a bot or fake account. All scores are **probabilistic signals, not definitive classifications**.

## Reconciliation from Source Scripts

The scoring function was reconciled from four analysis scripts originally written for `@leviathan_news` channel analysis:

1. `channel-subscriber-analysis.py` — Initial scoring implementation
2. `spike-hour-analysis.py` — Identical copy of scoring from #1
3. `known-user-bot-score.py` — Most complete version (includes premium/emoji_status bonuses)
4. `join-date-analysis.py` — No scoring (join date analysis only)

**Differences reconciled:**
- Scripts #1 and #2 used slightly different reason label formatting (e.g., `"deleted_account(+5)"` vs `"deleted(+5)"`). Standardized to descriptive format.
- Script #3 included `premium(-2)` and `emoji_status(-1)` bonuses that were also in #1 and #2. All three agreed on these weights.
- All three scoring implementations agreed on all weight values. No substantive differences were found.

The canonical implementation in `tg_purge/scoring.py` uses the `known-user-bot-score.py` version as its base, with standardized reason labels.

## Scoring Table

| Signal | Weight | Rationale |
|--------|--------|-----------|
| `deleted_account` | +5 | Deleted accounts are definitionally inactive. Early return — no other checks needed. |
| `scam_flag` | +5 | Telegram's own scam detection flagged this account. |
| `fake_flag` | +5 | Telegram's own fake detection flagged this account. |
| `is_bot` | +3 | Self-identified Bot API account (has a bot token). Legitimate bots exist, so not +5. |
| `restricted` | +2 | Telegram restricted this account (e.g., spam behavior). |
| `no_status_ever` | +2 | User has never been seen online. Could be privacy-conscious human, but strong bot signal. |
| `offline_>365d` | +2 | Last seen more than a year ago. Account is effectively abandoned. |
| `offline_>180d` | +1 | Last seen 6-12 months ago. Moderately stale. |
| `last_month` | +1 | Last seen within the past month but not recently. Mild staleness signal. |
| `no_photo` | +1 | No profile photo. Common for bots and throwaway accounts. |
| `no_username` | +1 | No @username set. Common for mass-created accounts. |
| `short_name` | +1 | First name is 0-1 characters. Minimal effort profile. |
| `no_last+no_user` | +1 | No last name AND no username. Compound signal — very sparse profile. |
| `digit_name` | +1 | First name is >30% digits (e.g., "User38291"). Generated-looking. |
| `mixed_scripts` | +1 | First name mixes Latin + Cyrillic or Arabic scripts. Unusual for real users. |
| `premium` | -2 | Telegram Premium subscriber. Strong signal of a real, paying user. |
| `emoji_status` | -1 | Has a custom emoji status (Premium feature). Additional legitimacy signal. |

## The Score-2 Cliff

Score 2 is the default threshold for flagging accounts as "likely bot." This is significant because it's very easy to reach score 2 with just two mild signals:

- `no_username(+1)` + `no_photo(+1)` = score 2
- `no_status_ever(+2)` alone = score 2

This means **any** user who hides their online status and has a sparse profile gets flagged. When we validated against known contributors on `@leviathan_news`, approximately 22% of confirmed real users scored >= 2.

**Recommendation**: Use threshold 3 or 4 for any automated actions. Threshold 2 is best used for analysis only — to understand the population, not to take action.

## Recommended Thresholds

| Threshold | Risk Profile | Use Case |
|-----------|-------------|----------|
| >= 1 | Very aggressive | Analysis only. High false positive rate. |
| >= 2 | Aggressive | Default for analysis. ~22% false positive rate on known contributors. |
| >= 3 | Moderate | Reasonable for candidate lists. Lower false positive rate. |
| >= 4 | Conservative | Recommended for any action. Most flagged accounts are clearly suspicious. |
| >= 5 | Very conservative | Deleted/scam/fake accounts only. Minimal false positives. |

## Three-Round Validation Methodology

When evaluating the heuristics against a real channel:

1. **Phase 1 — Recent subscribers**: Analyze the 200 most recently active users. These skew toward real, engaged users and establish a baseline false positive rate.

2. **Phase 2 — Search sampling**: Use 22-67 search queries to sample across the broader subscriber base. These reach inactive/dormant accounts and bot-heavy populations.

3. **Phase 3 — Known-good validation**: Score a set of known real users (e.g., from your own database) and measure how many get falsely flagged. This is the ground truth for false positive rate.

## Limitations

### Heuristic-only (no ground truth)
These scores are based on publicly visible profile attributes. There is no way to definitively determine if a Telegram account is a bot from its profile alone. Sophisticated bot farms create accounts with photos, usernames, and activity patterns indistinguishable from real users.

### Telegram broadcast channel specific
The scoring system was developed for **broadcast channels** (one-way communication). Signals like "no activity status" may have different meaning in groups or supergroups where users actively participate.

### 200-result query cap
Each `GetParticipantsRequest` returns at most 200 results. On channels with >10K subscribers, this means large populations are only partially sampled. The "full" strategy (67 queries) mitigates this but does not eliminate it.

### ~10K enumeration ceiling
Telegram's server-side enumeration has a practical ceiling around 10K unique participants. Channels with significantly more subscribers will have incomplete coverage regardless of query strategy.

### False positive risks at each threshold
No threshold eliminates false positives entirely. Privacy-conscious users who hide their online status (`no_status_ever`) and use minimal profiles will be flagged regardless of threshold. The `premium(-2)` bonus helps but only applies to paying users.

### Activity status privacy
Users can hide their "last seen" status in Telegram privacy settings. These users appear as `no_status_ever` (+2), the single strongest bot signal. This is the largest source of false positives in the system.
