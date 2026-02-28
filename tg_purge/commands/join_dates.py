"""
Join date clustering and spike detection command.

Ported from join-date-analysis.py. Enumerates subscribers, extracts join
dates from ChannelParticipant records, and produces:
  - Monthly join distribution
  - Top spike days by volume
  - Hourly breakdown of spike days
  - 48-hour sliding window burst detection
"""

import asyncio
from collections import Counter
from datetime import datetime, timezone, timedelta

from ..client import create_client, resolve_channel
from ..config import load_config
from ..enumeration import enumerate_subscribers


async def run(args):
    """Execute the join-dates command."""
    config = load_config(getattr(args, "config", None))
    if args.session_path:
        config.session_path = args.session_path
    channel_name = config.resolve_channel(args.channel)

    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        # Enumerate subscribers
        def progress(i, total, found):
            if i % 10 == 0:
                print(f"  ...{i}/{total} queries, {found} participants found", flush=True)

        print(f"\nEnumerating subscribers...", flush=True)
        result = await enumerate_subscribers(
            client, channel,
            strategy=getattr(args, "strategy", "full"),
            delay=config.delay,
            progress_callback=progress,
        )

        join_dates = result["join_dates"]
        print(f"\nTotal participants with join dates: {len(join_dates)}", flush=True)

        if not join_dates:
            print("No join dates found! The API may not return dates for this channel type.")
            return

        # ── Analysis ──────────────────────────────────────────────
        dates = sorted(join_dates.values())
        earliest = dates[0]
        latest = dates[-1]

        print(f"\n{'=' * 80}")
        print("JOIN DATE ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Earliest join: {earliest.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Latest join:   {latest.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Span:          {(latest - earliest).days} days")

        # ── Monthly distribution ──────────────────────────────────
        monthly = Counter()
        for d in dates:
            monthly[d.strftime('%Y-%m')] += 1

        print(f"\n{'─' * 80}")
        print("MONTHLY JOIN DISTRIBUTION")
        print(f"{'─' * 80}")

        max_monthly = max(monthly.values()) if monthly else 1
        for month in sorted(monthly.keys()):
            count = monthly[month]
            bar_len = int(count / max_monthly * 50)
            bar = "\u2588" * bar_len
            print(f"  {month}: {count:5d} {bar}")

        # ── Daily distribution ────────────────────────────────────
        daily = Counter()
        for d in dates:
            daily[d.strftime('%Y-%m-%d')] += 1

        top_days_count = getattr(args, "top_days", 30)
        top_days = daily.most_common(top_days_count)

        print(f"\n{'─' * 80}")
        print(f"TOP {top_days_count} DAYS BY JOIN VOLUME (potential bot farm waves)")
        print(f"{'─' * 80}")

        for day, count in top_days:
            bar = "\u2588" * min(count, 60)
            print(f"  {day}: {count:5d} {bar}")

        # ── Hourly breakdown of top spike days ────────────────────
        print(f"\n{'─' * 80}")
        print("HOURLY BREAKDOWN OF TOP 5 SPIKE DAYS")
        print(f"{'─' * 80}")

        top_5_days = [day for day, _ in top_days[:5]]
        for spike_day in top_5_days:
            hourly = Counter()
            for d in dates:
                if d.strftime('%Y-%m-%d') == spike_day:
                    hourly[d.hour] += 1

            print(f"\n  {spike_day} ({daily[spike_day]} joins):")
            for hour in sorted(hourly.keys()):
                count = hourly[hour]
                bar = "\u2588" * min(count, 40)
                print(f"    {hour:02d}:00 \u2014 {count:3d} {bar}")

        # ── 48-hour burst detection ───────────────────────────────
        print(f"\n{'─' * 80}")
        print("48-HOUR WINDOW ANALYSIS (sliding)")
        print(f"{'─' * 80}")

        sorted_dates = sorted(dates)
        max_window_count = 0
        max_window_start = None
        windows = []

        unique_days = sorted(set(d.date() for d in sorted_dates))
        for start_date in unique_days:
            start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
            end_dt = start_dt + timedelta(hours=48)
            count = sum(1 for d in sorted_dates if start_dt <= d < end_dt)
            if count >= 50:
                windows.append((start_date, count))
            if count > max_window_count:
                max_window_count = count
                max_window_start = start_date

        if windows:
            print(f"\n  Windows with 50+ joins in 48 hours:")
            windows.sort(key=lambda x: -x[1])
            shown = set()
            display_count = 0
            for start, count in windows:
                if any(abs((start - s).days) <= 2 for s in shown):
                    continue
                shown.add(start)
                bar = "\u2588" * min(count, 50)
                print(f"    {start} \u2014 {start + timedelta(days=2)}: {count:5d} joins {bar}")
                display_count += 1
                if display_count >= 20:
                    break

            print(f"\n  Largest 48-hour window: {max_window_start} ({max_window_count} joins)")
        else:
            print(f"  No 48-hour windows with 50+ joins found.")

        # ── Summary ───────────────────────────────────────────────
        total = len(join_dates)
        avg_daily = total / max((latest - earliest).days, 1)
        median_date = dates[len(dates) // 2]

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total subscribers with join dates:  {total}")
        print(f"Average joins per day:              {avg_daily:.1f}")
        print(f"Median join date:                   {median_date.strftime('%Y-%m-%d')}")
        if top_days:
            print(f"Largest single-day spike:           {top_days[0][0]} ({top_days[0][1]} joins)")
        if max_window_start:
            print(f"Largest 48-hour window:             {max_window_start} ({max_window_count} joins)")

        top5_total = sum(count for _, count in top_days[:5])
        print(f"Joins in top 5 days:                {top5_total} ({top5_total/total*100:.1f}% of sample)")

        top10_total = sum(count for _, count in top_days[:10])
        print(f"Joins in top 10 days:               {top10_total} ({top10_total/total*100:.1f}% of sample)")

    finally:
        await client.disconnect()

    print(f"\n{'=' * 80}")
    print("Done. No changes were made \u2014 read-only analysis.")
    print(f"{'=' * 80}")
