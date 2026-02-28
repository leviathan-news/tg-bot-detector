"""
Deep-dive analysis of specific time windows.

Ported from spike-hour-analysis.py. Takes arbitrary --start and --end
timestamps, enumerates subscribers, filters to the window, scores them,
and compares against a control group (all non-spike subscribers).

Produces: score distribution, threshold analysis, activity status breakdown,
profile characteristics, sample human-like and bot-like users.
"""

import asyncio
from collections import Counter
from datetime import datetime, timezone

from ..client import create_client, resolve_channel
from ..config import load_config
from ..enumeration import enumerate_subscribers
from ..scoring import score_user, format_name, status_label
from ..formatters import print_score_distribution, print_threshold_analysis
from ..clustering import detect_spike_windows, merge_windows


def _analyze_group(name, users, join_dates):
    """Analyze and print results for a group of users."""
    print(f"\n{'=' * 90}")
    print(f"GROUP: {name}")
    print(f"{'=' * 90}")
    print(f"Users in this group: {len(users)}")

    if not users:
        print("  (no users found)")
        return []

    scored = []
    for uid, user in users.items():
        s, reasons = score_user(user)
        join_time = join_dates.get(uid)
        scored.append((user, s, reasons, join_time))

    scored.sort(key=lambda x: -x[1])
    total = len(scored)

    # Score distribution
    scored_simple = [(u, s, r) for u, s, r, _ in scored]
    print_score_distribution(scored_simple, "SCORE DISTRIBUTION")
    print_threshold_analysis(scored_simple)

    # Activity status breakdown
    print(f"\n{'─' * 90}")
    print("ACTIVITY STATUS")
    print(f"{'─' * 90}")
    status_counts = Counter()
    for user, _, _, _ in scored:
        status_counts[status_label(user)] += 1
    for status, count in status_counts.most_common():
        pct = count / total * 100
        print(f"  {status:20s}: {count:5d} ({pct:5.1f}%)")

    # Profile characteristics
    print(f"\n{'─' * 90}")
    print("PROFILE CHARACTERISTICS")
    print(f"{'─' * 90}")
    has_photo = sum(1 for u, _, _, _ in scored if not u.deleted and getattr(u, "photo", None))
    has_username = sum(1 for u, _, _, _ in scored if not u.deleted and u.username)
    has_last = sum(1 for u, _, _, _ in scored if not u.deleted and u.last_name)
    has_premium = sum(1 for u, _, _, _ in scored if not u.deleted and getattr(u, "premium", False))
    is_deleted = sum(1 for u, _, _, _ in scored if u.deleted)

    print(f"  Has profile photo:   {has_photo:5d} ({has_photo/total*100:5.1f}%)")
    print(f"  Has username:        {has_username:5d} ({has_username/total*100:5.1f}%)")
    print(f"  Has last name:       {has_last:5d} ({has_last/total*100:5.1f}%)")
    print(f"  Has premium:         {has_premium:5d} ({has_premium/total*100:5.1f}%)")
    print(f"  Deleted accounts:    {is_deleted:5d} ({is_deleted/total*100:5.1f}%)")

    # Sample human-like users
    human_like = [(u, s, r, t) for u, s, r, t in scored if s <= 1]
    if human_like:
        human_like.sort(key=lambda x: x[1])
        sample_size = min(20, len(human_like))
        print(f"\n{'─' * 90}")
        print(f"MOST HUMAN-LOOKING USERS (score \u22641) \u2014 sample of {sample_size}")
        print(f"{'─' * 90}")
        for user, s, reasons, jt in human_like[:sample_size]:
            name = format_name(user)[:40]
            status = status_label(user)
            reason_str = ", ".join(reasons) if reasons else "clean"
            jt_str = jt.strftime('%H:%M') if jt else "?"
            print(f"  Score {s:2d} | {name:40s} | {status:12s} | joined {jt_str} | {reason_str}")

    # Sample bot-like users
    bot_like = [(u, s, r, t) for u, s, r, t in scored if s >= 3]
    if bot_like:
        bot_like.sort(key=lambda x: -x[1])
        sample_size = min(20, len(bot_like))
        print(f"\n{'─' * 90}")
        print(f"MOST BOT-LIKE USERS (score \u22653) \u2014 sample of {sample_size}")
        print(f"{'─' * 90}")
        for user, s, reasons, jt in bot_like[:sample_size]:
            name = format_name(user)[:40]
            status = status_label(user)
            reason_str = ", ".join(reasons) if reasons else "clean"
            jt_str = jt.strftime('%H:%M') if jt else "?"
            print(f"  Score {s:2d} | {name:40s} | {status:12s} | joined {jt_str} | {reason_str}")

    return scored_simple


def _parse_timestamp(ts_str):
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime."""
    # Handle common formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%H:%M%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str!r}")


async def run(args):
    """Execute the spike command."""
    config = load_config(getattr(args, "config", None))
    if args.session_path:
        config.session_path = args.session_path
    if getattr(args, "delay", None) is not None:
        config.delay = args.delay
    channel_name = config.resolve_channel(args.channel)

    spike_start = _parse_timestamp(args.start)
    spike_end = _parse_timestamp(args.end)

    print(f"Spike window: {spike_start.isoformat()} to {spike_end.isoformat()}")

    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        # Enumerate all subscribers
        def progress(i, total, found):
            if i % 10 == 0:
                print(f"  ...{i}/{total} queries, {found} participants", flush=True)

        print(f"\nEnumerating subscribers...", flush=True)
        result = await enumerate_subscribers(
            client, channel,
            strategy=getattr(args, "strategy", "full"),
            delay=config.delay,
            progress_callback=progress,
        )

        all_users = result["users"]
        join_dates = result["join_dates"]

        print(f"\nTotal participants enumerated: {len(all_users)}")

        # Auto-detect additional spike windows from join dates
        auto_windows = []
        if join_dates:
            auto_windows = detect_spike_windows(join_dates)
            if auto_windows:
                print(f"Auto-detected {len(auto_windows)} additional spike window(s)")

        # Merge manual window with auto-detected ones
        all_spike_windows = merge_windows([(spike_start, spike_end)] + auto_windows)

        # Filter to spike window
        spike_users = {}
        spike_dates = {}
        for uid, join_date in join_dates.items():
            if spike_start <= join_date < spike_end:
                if uid in all_users:
                    spike_users[uid] = all_users[uid]
                    spike_dates[uid] = join_date

        # Control group: everyone else
        normal_users = {}
        normal_dates = {}
        for uid, user in all_users.items():
            if uid not in spike_users:
                normal_users[uid] = user
                if uid in join_dates:
                    normal_dates[uid] = join_dates[uid]

        # Analyze each group
        spike_label = f"Spike window ({spike_start.strftime('%Y-%m-%d %H:%M')} to {spike_end.strftime('%Y-%m-%d %H:%M')} UTC)"
        spike_scored = _analyze_group(spike_label, spike_users, spike_dates)
        control_scored = _analyze_group("Control group (all other dates)", normal_users, normal_dates)

        # Comparison summary
        if spike_scored and control_scored:
            from ..formatters import print_comparison_table
            print_comparison_table([
                ("Spike", spike_scored),
                ("Control", control_scored),
            ], title="COMPARISON: SPIKE vs CONTROL")

    finally:
        await client.disconnect()

    print(f"\n{'=' * 90}")
    print("Done. No changes were made \u2014 read-only analysis.")
    print(f"{'=' * 90}")
