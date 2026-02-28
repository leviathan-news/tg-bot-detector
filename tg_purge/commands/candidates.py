"""
Generate scored candidate lists for offline review.

Enumerates subscribers, scores them, and outputs a list of candidates
above a given threshold. Supports safelist exclusion and CSV/JSON export.

This is a pure analysis tool — no destructive capability.
The output is intended for human review before any action is taken.
"""

import asyncio
import csv
import json
from pathlib import Path

from ..client import create_client, resolve_channel
from ..config import load_config
from ..enumeration import enumerate_subscribers
from ..scoring import score_user, format_name, status_label
from ..formatters import (
    print_score_distribution,
    print_signal_frequency,
    print_threshold_analysis,
    export_csv,
    export_json,
)


def _load_safelist(path):
    """Load a safelist of user IDs that should be excluded from output.

    Accepts CSV, JSON, or plain text (one ID per line).

    Returns:
        Set of user IDs (ints).
    """
    if not path:
        return set()

    path = Path(path)
    if not path.exists():
        print(f"Warning: safelist file not found: {path}")
        return set()

    ids = set()
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    uid = item.get("user_id")
                    if uid:
                        ids.add(int(uid))
                else:
                    ids.add(int(item))
        return ids

    elif suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id")
                if uid:
                    try:
                        ids.add(int(uid))
                    except ValueError:
                        pass
        return ids

    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        ids.add(int(line))
                    except ValueError:
                        pass
        return ids


async def run(args):
    """Execute the candidates command."""
    config = load_config(getattr(args, "config", None))
    if args.session_path:
        config.session_path = args.session_path
    if getattr(args, "delay", None) is not None:
        config.delay = args.delay
    channel_name = config.resolve_channel(args.channel)
    threshold = getattr(args, "threshold", None) or config.threshold

    # Load safelist
    safelist = _load_safelist(getattr(args, "safelist", None))
    if safelist:
        print(f"Safelist: {len(safelist)} protected user IDs loaded")

    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)
        sub_count = getattr(channel, "participants_count", None)

        # Enumerate
        def progress(i, total, found):
            if i % 10 == 0:
                print(f"  ...{i}/{total} queries, {found} users found", flush=True)

        print(f"\nEnumerating subscribers...", flush=True)
        result = await enumerate_subscribers(
            client, channel,
            strategy=getattr(args, "strategy", "full"),
            delay=config.delay,
            progress_callback=progress,
        )

        all_users = result["users"]
        print(f"\nTotal users enumerated: {len(all_users)}")

        # Score everyone
        all_scored = []
        for uid, user in all_users.items():
            s, reasons = score_user(user)
            all_scored.append((user, s, reasons))

        all_scored.sort(key=lambda x: -x[1])

        # Filter to candidates above threshold, excluding safelist
        candidates = [
            (u, s, r) for u, s, r in all_scored
            if s >= threshold and u.id not in safelist
        ]

        safelisted_count = sum(
            1 for u, s, r in all_scored
            if s >= threshold and u.id in safelist
        )

        # ── Summary ───────────────────────────────────────────────
        print(f"\n{'=' * 80}")
        print(f"CANDIDATE ANALYSIS (threshold \u2265{threshold})")
        print(f"{'=' * 80}")
        print(f"Total users analyzed:       {len(all_scored)}")
        print(f"Candidates above threshold: {len(candidates)}")
        if safelisted_count:
            print(f"Safelisted (excluded):      {safelisted_count}")
        if sub_count:
            print(f"Channel total subscribers:  {sub_count:,}")
            sample_pct = len(all_scored) / sub_count * 100
            print(f"Sample coverage:            {sample_pct:.1f}%")

        print_score_distribution(all_scored, "SCORE DISTRIBUTION (all analyzed)")
        print_threshold_analysis(all_scored)
        print_signal_frequency(candidates, f"SIGNAL FREQUENCY (candidates \u2265{threshold})")

        # Export
        output_path = getattr(args, "output", None)
        if output_path:
            if output_path.endswith(".json"):
                export_json(candidates, output_path)
            else:
                export_csv(candidates, output_path)
        else:
            # Print candidates to stdout
            print(f"\n{'─' * 80}")
            print(f"CANDIDATES (score \u2265{threshold}) \u2014 {len(candidates)} users")
            print(f"{'─' * 80}")
            for user, s, reasons in candidates[:100]:
                name = format_name(user)[:40]
                status = status_label(user)
                reason_str = ", ".join(reasons) if reasons else "clean"
                print(f"  Score {s:2d} | ID {user.id:>12d} | {name:40s} | {status:15s} | {reason_str}")
            if len(candidates) > 100:
                print(f"  ... and {len(candidates) - 100} more (use --output to export all)")

    finally:
        await client.disconnect()

    print(f"\n{'=' * 80}")
    print("Done. No changes were made \u2014 analysis only.")
    print(f"{'=' * 80}")
