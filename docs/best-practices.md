# Best Practices

## Telegram API Rate Limits

Telegram enforces strict rate limits on all API calls. The default delay of 1.5 seconds between queries is conservative and should avoid `FloodWaitError` in most cases.

If you hit a `FloodWaitError`:
- The error message includes a `seconds` field telling you how long to wait
- Telethon handles this automatically in most cases, but long waits (>60s) may indicate you should slow down your query rate
- Increase the `delay` setting in your config
- Consider using the `minimal` strategy (22 queries) instead of `full` (67 queries)

### Recommended batch sizes

| Operation | Recommended delay | Notes |
|-----------|------------------|-------|
| Search queries | 1.5s | Default, safe for sustained use |
| Single user lookups | 0.5s | Lower delay OK for small batches |
| Bulk operations | 2.0-3.0s | If doing hundreds of queries |

## MTProto vs Bot API

This toolkit uses **Telethon** (MTProto client), not the Telegram Bot API. Key differences:

- **MTProto**: Full user account access, can enumerate channel subscribers, see user profiles. Requires phone number authentication.
- **Bot API**: Bot token authentication, cannot enumerate subscribers of channels the bot isn't admin of, limited user profile access.

We use MTProto because subscriber enumeration requires `GetParticipantsRequest`, which is not available through the Bot API.

## Enumeration Strategies and Biases

### ChannelParticipantsRecent
- Returns ~200 most recently **active** users (not most recently joined)
- Skews toward engaged, real users
- Underrepresents dormant/bot accounts
- Good baseline for "what does a healthy subscriber look like"

### ChannelParticipantsSearch
- Searches by name/username prefix
- Returns at most 200 results per query, even if more match
- Server-side enumeration ceiling is ~10K total unique participants
- Users with no name or very unusual names may be missed entirely
- Single-letter queries hit the 200 cap on large channels, missing many matches

### Coverage estimates
- **Minimal** (22 queries): Typically reaches 2,000-4,000 unique users
- **Full** (67 queries): Typically reaches 5,000-8,000 unique users
- Neither achieves full coverage on channels with >10K subscribers

## Session File Security

**Session files are equivalent to login credentials.**

A Telethon `.session` file contains your authentication tokens. Anyone with access to this file can:
- Send messages as you
- Read your private messages
- Join/leave channels
- Perform any action your account can

### Protection measures
1. **Never commit session files to git** (already in `.gitignore`)
2. **Never share session files** with anyone
3. **Store on encrypted filesystem** if on shared systems
4. **Enable 2FA** on your Telegram account (adds an additional layer)
5. **Use a dedicated account** for bot detection if possible
6. **Revoke sessions** periodically via Telegram Settings > Devices

### Non-interactive auth for automated pipelines
For CI/CD or cron usage:
1. Run `tg-purge analyze --channel @test` once interactively to create the session
2. Copy the `.session` file to your automation environment
3. Secure it with filesystem permissions (`chmod 600`)
4. The session will be reused without prompting for phone/code

## Admin Requirements

| Operation | Required permissions |
|-----------|---------------------|
| Analysis (all commands) | Channel subscriber (read access) |
| Purge (v2, not yet implemented) | `ban_users` admin permission |

For analysis, you just need to be subscribed to the channel. No admin rights required.

## FloodWaitError Handling

Telethon's built-in retry logic handles most `FloodWaitError` cases automatically. If you encounter persistent flood errors:

1. **Increase delay**: Set `delay = 3.0` or higher in config
2. **Use minimal strategy**: Fewer queries = less API pressure
3. **Wait and retry**: Telegram flood limits typically reset within minutes
4. **Check other sessions**: Rate limits are per-account, not per-session. Other tools using the same Telegram account contribute to the limit.
