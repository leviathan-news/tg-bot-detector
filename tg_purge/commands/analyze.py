"""
Multi-round subscriber analysis command.

Ported from channel-subscriber-analysis.py. Three phases:
  1. Self-identified bots (Bot API accounts)
  2. Recently active subscribers
  3. Search-based sampling across name patterns

Produces score distributions, signal frequency, threshold analysis,
and extrapolation estimates.
"""

import asyncio

from ..client import create_client, resolve_channel
from ..config import load_config
from ..enumeration import fetch_bots, fetch_recent, fetch_by_search, MINIMAL_QUERIES, FULL_QUERIES
from ..scoring import score_user, format_name, status_label
from ..formatters import (
    print_section,
    print_score_distribution,
    print_signal_frequency,
    print_threshold_analysis,
    print_comparison_table,
)


async def run(args):
    """Execute the analyze command."""
    config = load_config(getattr(args, "config", None))
    if args.session_path:
        config.session_path = args.session_path
    channel_name = config.resolve_channel(args.channel)

    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)
        sub_count = getattr(channel, "participants_count", None)
        delay = config.delay

        # ── Phase 0: Self-identified bots ─────────────────────────
        print(f"\n{'=' * 80}")
        print("PHASE 0: Self-identified bots (Bot API accounts subscribed to channel)")
        print(f"{'=' * 80}")

        try:
            actual_bots = await fetch_bots(client, channel)
            print(f"Found {len(actual_bots)} Bot API accounts:")
            for b in actual_bots:
                username = f"@{b.username}" if b.username else "(no username)"
                print(f"  * {b.first_name or '(unnamed)'} {username}")
            if not actual_bots:
                print("  (none)")
        except Exception as e:
            print(f"  ERROR fetching bots: {e}")
            actual_bots = []

        await asyncio.sleep(delay)

        # ── Phase 1: Recent subscribers ───────────────────────────
        print(f"\n{'=' * 80}")
        print("PHASE 1: Recently active subscribers (200 max)")
        print(f"{'=' * 80}")

        recent_users, _ = await fetch_recent(client, channel)
        seen_ids = {u.id for u in recent_users}
        recent_scored = [(u, *score_user(u)) for u in recent_users]
        recent_scored.sort(key=lambda x: -x[1])

        print(f"Fetched: {len(recent_users)} users")
        bot_count = sum(1 for _, s, _ in recent_scored if s >= 2)
        total_recent = len(recent_scored)
        if total_recent > 0:
            print(f"Likely bots (\u22652): {bot_count} ({bot_count/total_recent*100:.0f}%)")

        bots_recent = [(u, s, r) for u, s, r in recent_scored if s >= 2]
        print_section("Recent \u2014 flagged as likely bot (\u22652)", bots_recent)

        # ── Phase 2: Search-based sampling ────────────────────────
        print(f"\n\n{'=' * 80}")
        print("PHASE 2: Search-based sampling across name patterns")
        print(f"{'=' * 80}")

        queries = FULL_QUERIES if args.strategy == "full" else MINIMAL_QUERIES
        search_users = {}  # id -> (user, source_query)

        for query in queries:
            display_q = repr(query) if query else '""'
            try:
                users, _ = await fetch_by_search(client, channel, query)
                new_users = [u for u in users if u.id not in seen_ids]

                for u in new_users:
                    if u.id not in search_users:
                        search_users[u.id] = (u, display_q)
                    seen_ids.add(u.id)

                hit_cap = " >> HIT 200 CAP" if len(users) >= 200 else ""
                print(f"  Search {display_q:6s}: {len(users):3d} results, {len(new_users):3d} new{hit_cap}")

            except Exception as e:
                print(f"  Search {display_q:6s}: ERROR \u2014 {e}")

            await asyncio.sleep(delay)

        # Score search-discovered users
        search_scored = []
        for uid, (user, source) in search_users.items():
            s, reasons = score_user(user)
            search_scored.append((user, s, reasons))

        search_scored.sort(key=lambda x: -x[1])

        total_search = len(search_scored)
        if total_search > 0:
            bots_search = [(u, s, r) for u, s, r in search_scored if s >= 2]
            borderline_search = [(u, s, r) for u, s, r in search_scored if s == 1]
            clean_search = [(u, s, r) for u, s, r in search_scored if s <= 0]

            print_section("Search \u2014 flagged as likely bot (\u22652)", bots_search, max_display=50)
            print_section("Search \u2014 borderline (score 1) [sample]", borderline_search, max_display=20)
            print_section("Search \u2014 likely human (score \u22640) [sample]", clean_search, max_display=20)

        # ── Combined Summary ──────────────────────────────────────
        all_scored = list(recent_scored) + list(search_scored)
        total = len(all_scored)

        bot_likely = sum(1 for _, s, _ in all_scored if s >= 2)
        bot_maybe = sum(1 for _, s, _ in all_scored if s == 1)
        human_likely = sum(1 for _, s, _ in all_scored if s <= 0)

        print(f"\n\n{'=' * 80}")
        print("COMBINED SUMMARY")
        print(f"{'=' * 80}")
        print(f"Self-identified Bot accounts: {len(actual_bots)}")
        print(f"Total unique users analyzed:  {total}")
        print(f"  From 'Recent' filter:      {len(recent_scored)}")
        print(f"  From search sampling:      {total_search}")
        print()
        if total > 0:
            print(f"Likely bots (score \u22652):      {bot_likely:4d}  ({bot_likely/total*100:.1f}%)")
            print(f"Borderline (score 1):        {bot_maybe:4d}  ({bot_maybe/total*100:.1f}%)")
            print(f"Likely human (score \u22640):     {human_likely:4d}  ({human_likely/total*100:.1f}%)")

        # Comparison
        print_comparison_table([
            ("Recent", recent_scored),
            ("Search", search_scored),
        ], title="COMPARISON: Recent (active) vs Search (broad)")

        # Distribution and signals
        print_score_distribution(all_scored, "SCORE DISTRIBUTION (all users)")
        print_signal_frequency(all_scored, "SIGNAL FREQUENCY (all users)")
        print_threshold_analysis(all_scored)

        # Extrapolation
        if sub_count and total > 0:
            print(f"\n{'─' * 80}")
            print("EXTRAPOLATION (if sample is representative)")
            print(f"{'─' * 80}")
            bot_pct = bot_likely / total
            border_pct = bot_maybe / total
            human_pct = human_likely / total
            print(f"Channel subscribers:         {sub_count:,d}")
            print(f"Estimated bots (\u22652):         ~{int(sub_count * bot_pct):,d}")
            print(f"Estimated borderline (1):    ~{int(sub_count * border_pct):,d}")
            print(f"Estimated real humans (\u22640):  ~{int(sub_count * human_pct):,d}")
            print()
            print("IMPORTANT CAVEATS:")
            print("  - Search-based sampling is NOT truly random")
            print("  - Accounts with no name/unusual names may be under-represented")
            print("  - The 200-result cap per query means large pools are only partially sampled")
            print("  - 'no_status_ever' is the strongest bot signal but may include")
            print("    privacy-conscious humans who hide their online status")

    finally:
        await client.disconnect()

    print(f"\n{'=' * 80}")
    print("Done. No changes were made \u2014 read-only analysis.")
    print(f"{'=' * 80}")
