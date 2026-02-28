# Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all issues identified in the systematic code review of tg-bot-detector v0.1.0.

**Architecture:** 10 fixes across enumeration, scoring, CLI, docs, and security. Each fix is independent. TDD where applicable — write failing tests first, then implement. Fixes are ordered by dependency: enumeration changes first (other code depends on them), then scoring, then CLI, then docs/security.

**Tech Stack:** Python 3.9+, Telethon, pytest

---

### Task 1: Add CJK characters to FULL_QUERIES

FULL_QUERIES is missing CJK chars (李, 王) that MINIMAL_QUERIES has. Running `full` strategy should be a superset of `minimal`.

**Files:**
- Modify: `tg_purge/enumeration.py:50-65`
- Test: `tests/test_enumeration.py` (create)

**Step 1: Write failing test**

```python
# tests/test_enumeration.py
"""Tests for enumeration query sets."""

from tg_purge.enumeration import MINIMAL_QUERIES, FULL_QUERIES


class TestQuerySets:
    """Verify query set coverage properties."""

    def test_full_queries_is_superset_of_minimal_unique_chars(self):
        """Every unique character in MINIMAL_QUERIES should appear in at least
        one FULL_QUERIES entry (either as the exact query or as a substring)."""
        # Extract single-char queries from minimal (exclude multi-char combos like "zq")
        minimal_single_chars = {q for q in MINIMAL_QUERIES if len(q) == 1}
        full_single_chars = {q for q in FULL_QUERIES if len(q) == 1}
        missing = minimal_single_chars - full_single_chars
        assert missing == set(), f"FULL_QUERIES missing chars from MINIMAL: {missing}"

    def test_full_contains_cjk(self):
        """FULL_QUERIES should contain CJK characters."""
        cjk_chars = {q for q in FULL_QUERIES if any('\u4e00' <= c <= '\u9fff' for c in q)}
        assert len(cjk_chars) >= 2, f"Expected at least 2 CJK queries, got {len(cjk_chars)}"

    def test_full_queries_has_more_than_minimal(self):
        assert len(FULL_QUERIES) > len(MINIMAL_QUERIES)

    def test_both_include_empty_string(self):
        assert "" in MINIMAL_QUERIES
        assert "" in FULL_QUERIES
```

**Step 2:** Run `python -m pytest tests/test_enumeration.py -v`, expect FAIL on `test_full_queries_is_superset` and `test_full_contains_cjk`.

**Step 3: Add CJK chars to FULL_QUERIES**

In `tg_purge/enumeration.py`, add CJK entries to FULL_QUERIES after the Arabic section:

```python
    # CJK (common surnames — covers Chinese/Japanese/Korean bot names)
    "\u674e", "\u738b",  # 李, 王
```

**Step 4:** Run `python -m pytest tests/test_enumeration.py -v`, expect all PASS.

**Step 5:** Commit: `fix: add CJK characters to FULL_QUERIES for superset coverage of MINIMAL`

---

### Task 2: Fix silent exception swallowing in enumeration

`enumerate_subscribers` catches exceptions but never logs the error message. The variable `e` is captured but unused.

**Files:**
- Modify: `tg_purge/enumeration.py:188-191`

**Step 1: Write failing test**

```python
# Add to tests/test_enumeration.py

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

class TestEnumerateSubscribers:
    """Test enumerate_subscribers behavior."""

    def test_error_query_records_error_message(self):
        """When a query fails, the error message should be captured in query_stats."""
        mock_client = AsyncMock()
        mock_channel = MagicMock()

        # Make fetch_by_search raise on every call
        with patch("tg_purge.enumeration.fetch_by_search",
                    side_effect=Exception("test error")):
            result = asyncio.run(
                _enumerate_single_query(mock_client, mock_channel, "a")
            )
        # Verify error is captured (not silently swallowed)
        # We test this indirectly by checking the query_stats tuple has 3 elements
        # The fix adds error info to stderr, which we verify doesn't crash
```

Actually, the simplest test for this is to verify the function doesn't crash on errors and still produces stats. The real fix is adding `print(f"  Search {display_q}: ERROR — {e}", file=sys.stderr)` to the except block.

**Step 1 (revised): Fix directly — this is a one-line stderr log addition, not logic change**

In `tg_purge/enumeration.py:188-191`, change:

```python
        except Exception as e:
            query_stats.append((display_q, 0, 0))
```

to:

```python
        except Exception as e:
            import sys
            print(f"  Query {display_q}: error — {e}", file=sys.stderr)
            query_stats.append((display_q, 0, 0))
```

Move the `import sys` to the top of the file with other imports.

**Step 2:** Run `python -m pytest tests/ -v`, expect all 101+ tests still PASS.

**Step 3:** Commit: `fix: log enumeration query errors to stderr instead of silently swallowing`

---

### Task 3: Implement recursive prefix expansion in enumeration

When a search query returns 200 results (the API cap), automatically drill deeper with prefix combinations up to max_depth=3.

**Files:**
- Modify: `tg_purge/enumeration.py` (add `_expand_query` helper, modify `enumerate_subscribers`)
- Test: `tests/test_enumeration.py` (add recursive expansion tests)

**Step 1: Write failing tests**

```python
# Add to tests/test_enumeration.py

class TestRecursiveExpansion:
    """Test recursive prefix expansion when queries hit the 200-result cap."""

    def test_no_expansion_under_cap(self):
        """Queries returning <200 results should not expand."""
        mock_client = AsyncMock()
        mock_channel = MagicMock()

        # Return 50 users (under cap)
        mock_result = MagicMock()
        mock_result.users = [MagicMock(id=i) for i in range(50)]
        mock_result.participants = []
        mock_client.return_value = mock_result

        result = asyncio.run(enumerate_subscribers(
            mock_client, mock_channel,
            strategy="minimal", delay=0, max_depth=3
        ))
        # Should not have made more calls than the query count
        assert mock_client.call_count == len(MINIMAL_QUERIES)

    def test_expansion_at_cap(self):
        """Queries returning exactly 200 results should trigger expansion."""
        call_count = [0]
        mock_client = AsyncMock()
        mock_channel = MagicMock()

        async def mock_fetch(client, channel, query, limit=200):
            call_count[0] += 1
            # First call returns 200 (hits cap), subsequent calls return <200
            if len(query) <= 1 and query in ["a"]:
                users = [MagicMock(id=1000 + i) for i in range(200)]
            else:
                users = [MagicMock(id=2000 + call_count[0]) for _ in range(5)]
            participants = []
            return users, participants

        with patch("tg_purge.enumeration.fetch_by_search", side_effect=mock_fetch):
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=2
            ))
        # Should have made more calls than base query count due to expansion of "a"
        assert call_count[0] > len(MINIMAL_QUERIES)

    def test_max_depth_zero_disables_expansion(self):
        """max_depth=0 should disable recursive expansion entirely."""
        mock_client = AsyncMock()
        mock_channel = MagicMock()

        mock_result = MagicMock()
        mock_result.users = [MagicMock(id=i) for i in range(200)]
        mock_result.participants = []
        mock_client.return_value = mock_result

        with patch("tg_purge.enumeration.fetch_by_search") as mock_fetch:
            mock_fetch.return_value = ([MagicMock(id=i) for i in range(200)], [])
            result = asyncio.run(enumerate_subscribers(
                mock_client, mock_channel,
                strategy="minimal", delay=0, max_depth=0
            ))
        # Exactly as many calls as queries — no expansion
        assert mock_fetch.call_count == len(MINIMAL_QUERIES)
```

**Step 2:** Run tests, expect FAIL (max_depth parameter doesn't exist yet).

**Step 3: Implement recursive expansion**

Add to `enumerate_subscribers`:
- New `max_depth` parameter (default=3)
- `EXPANSION_CHARS` constant: lowercase Latin a-z for expanding prefixes
- When a query returns >= 200 results AND depth < max_depth, generate sub-queries by appending each expansion char
- Track depth per query to enforce max_depth

**Step 4:** Run tests, expect PASS.

**Step 5:** Commit: `feat: recursive prefix expansion when search hits 200-result cap`

---

### Task 4: Add join-date spike signal to scoring

Add optional `join_date` and `spike_windows` parameters to `score_user()`. If a user's join date falls within a spike window, add +2. Scoring still works without these params.

**Files:**
- Modify: `tg_purge/scoring.py` (add params to `score_user`, add `spike_join` to `ScoringConfig`)
- Modify: `tests/test_scoring.py` (add join-date scoring tests)
- Modify: `docs/scoring-methodology.md` (document new signal)

**Step 1: Write failing tests**

```python
# Add to tests/test_scoring.py
from datetime import datetime, timezone

class TestJoinDateScoring:
    """Test join-date spike signal integration."""

    def test_no_join_date_no_change(self, clean_user):
        """Without join_date param, scoring is unchanged."""
        score, reasons = score_user(clean_user)
        assert score == 0

    def test_join_date_outside_spike_no_penalty(self, clean_user):
        """Join date outside spike windows adds no penalty."""
        spike_windows = [
            (datetime(2025, 1, 1, tzinfo=timezone.utc),
             datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]
        score, reasons = score_user(
            clean_user,
            join_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            spike_windows=spike_windows,
        )
        assert not any("spike_join" in r for r in reasons)

    def test_join_date_inside_spike_adds_penalty(self, clean_user):
        """Join date inside a spike window adds penalty."""
        spike_windows = [
            (datetime(2025, 1, 1, tzinfo=timezone.utc),
             datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]
        score, reasons = score_user(
            clean_user,
            join_date=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            spike_windows=spike_windows,
        )
        assert any("spike_join" in r for r in reasons)
        assert score >= 2  # default spike_join weight is +2

    def test_spike_windows_none_no_effect(self, clean_user):
        """Passing join_date without spike_windows has no effect."""
        score, reasons = score_user(
            clean_user,
            join_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert score == 0

    def test_spike_join_weight_configurable(self, clean_user):
        """Spike join weight is configurable via ScoringConfig."""
        config = ScoringConfig(spike_join=3)
        spike_windows = [
            (datetime(2025, 1, 1, tzinfo=timezone.utc),
             datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ]
        score, reasons = score_user(
            clean_user, config=config,
            join_date=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            spike_windows=spike_windows,
        )
        assert any("+3" in r for r in reasons)
```

**Step 2:** Run tests, expect FAIL.

**Step 3: Implement**

- Add `spike_join: int = 2` to `ScoringConfig`
- Add `join_date=None, spike_windows=None` params to `score_user()`
- After all existing checks, if both params provided, check if join_date falls within any (start, end) window

**Step 4:** Run tests, expect PASS.

**Step 5:** Update `docs/scoring-methodology.md` scoring table with new signal.

**Step 6:** Commit: `feat: add join-date spike signal to scoring engine`

---

### Task 5: Move PII output to stderr in client.py

`client.py` prints the connected user's first name and channel title to stdout. For scripting, stdout should be data-only. Move informational output to stderr.

**Files:**
- Modify: `tg_purge/client.py:66,89-91`

**Step 1: Fix directly**

Change all `print(...)` calls in `client.py` to `print(..., file=sys.stderr)`. Add `import sys` if not present (it's already imported).

**Step 2:** Run `python -m pytest tests/ -v`, expect all PASS.

**Step 3:** Commit: `fix: move informational output to stderr in client.py`

---

### Task 6: Enforce session file permissions

After creating the session directory, set restrictive permissions on the directory.

**Files:**
- Modify: `tg_purge/client.py:48-49`

**Step 1: Implement**

After `session_path.parent.mkdir(parents=True, exist_ok=True)`, add:
```python
import stat
session_path.parent.chmod(stat.S_IRWXU)  # 700 — owner only
```

On first run, after the session file is created, also set its permissions:
```python
session_file = session_path.with_suffix(".session")
if session_file.exists():
    session_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600 — owner read/write only
```

**Step 2:** Run tests, expect PASS.

**Step 3:** Commit: `fix: enforce restrictive permissions on session directory and file`

---

### Task 7: Add --delay and --strategy to spike command, add --delay globally

`spike` hardcodes `strategy="full"`. No CLI command exposes `--delay`.

**Files:**
- Modify: `tg_purge/cli.py` (add --delay to common args, add --strategy to spike)
- Modify: `tg_purge/commands/spike.py:155` (use args.strategy)
- Modify: All command `run()` functions to respect `args.delay` if present
- Test: `tests/test_cli.py` (add tests for new args)

**Step 1: Write failing tests**

```python
# Add to tests/test_cli.py::TestBuildParser

    def test_delay_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--delay", "3.0"])
        assert args.delay == 3.0

    def test_spike_strategy_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "spike", "--channel", "@test",
            "--start", "2025-01-01", "--end", "2025-01-02",
            "--strategy", "minimal",
        ])
        assert args.strategy == "minimal"
```

**Step 2:** Run tests, expect FAIL.

**Step 3: Implement**

- Add `--delay` to `_add_common_args()` with `type=float, default=None`
- Add `--strategy` to spike subparser
- In each command's `run()`, apply: `if args.delay is not None: config.delay = args.delay`

**Step 4:** Run tests, expect PASS.

**Step 5:** Commit: `feat: add --delay global flag and --strategy to spike command`

---

### Task 8: Fix registry check int parsing

`registry.py:181` does `int(args.user_id)` without error handling.

**Files:**
- Modify: `tg_purge/commands/registry.py:181`
- Test: `tests/test_registry.py` (add test)

**Step 1: Write failing test** — test that run_check handles non-numeric user_id gracefully (prints error, doesn't crash).

**Step 2: Wrap in try/except**

```python
try:
    user_id = int(args.user_id)
except ValueError:
    print(f"Error: invalid user ID: {args.user_id!r} (must be numeric)")
    return
```

**Step 3:** Run tests, expect PASS.

**Step 4:** Commit: `fix: handle non-numeric user ID in registry check`

---

### Task 9: Fix 22% false positive figure in docs

`docs/scoring-methodology.md:52` claims "approximately 22% of confirmed real users scored >= 2" but squid-bot#77 data shows 1.9% (2/104) for known-good users validated against the channel.

**Files:**
- Modify: `docs/scoring-methodology.md:52`

**Step 1: Fix directly**

Replace the line with data from the actual validation (squid-bot#77 Round 3):
"When we validated against 104 known contributors on `@leviathan_news`, approximately 2.9% scored >= 2 (3/104). However, this sample represents the most engaged users; the false positive rate for passive real subscribers is likely higher."

**Step 2:** Commit: `docs: correct false positive rate to match validated data from squid-bot#77`

---

### Task 10: Document short_name limitation in scoring methodology

The `short_name` signal catching legitimate single-initial users (e.g., PaperImperium2's "P") is a known limitation discussed in squid-bot#77 but not documented in scoring-methodology.md.

**Files:**
- Modify: `docs/scoring-methodology.md` (add note under `short_name` row or in Limitations section)

**Step 1: Add note to Limitations section**

Add after the "Activity status privacy" subsection:
```markdown
### Short name false positives
Users who abbreviate their first name to a single character (e.g., "P" for a well-known contributor) trigger the `short_name(+1)` signal. Combined with other mild signals like `no_photo`, this can push legitimate accounts to score 2. The `candidates --safelist` flag is the recommended mitigation for protecting known contributors.
```

**Step 2:** Commit: `docs: document short_name false positive limitation`

---

## Execution Order

Tasks 1-3 (enumeration) → Task 4 (scoring) → Tasks 5-7 (CLI/security) → Task 8 (registry) → Tasks 9-10 (docs)

Dependencies: Task 3 depends on Task 1 (CJK chars). All others are independent.
