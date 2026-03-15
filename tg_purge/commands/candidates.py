"""
Generate scored candidate lists for offline review.

Enumerates subscribers, scores them, and outputs a list of candidates
above a given threshold. Supports safelist exclusion and CSV/JSON export.

This is a pure analysis tool — no destructive capability.
The output is intended for human review before any action is taken.

Ctrl+C handling: On interrupt, enumeration stops and all results collected
so far are scored and saved to a "-partial" file. A second Ctrl+C during
the save phase is blocked to prevent data loss.
"""

import asyncio
import csv
import json
import signal
import sys
from pathlib import Path

from ..client import create_client, resolve_channel
from ..config import load_config
from ..enumeration import enumerate_subscribers
from ..scoring import score_user, format_name, status_label
from ..clustering import detect_spike_windows
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


def _score_and_export(all_users, join_dates, spike_windows, threshold, safelist,
                      sub_count, output_path, interrupted):
    """Score all users and export results. Runs synchronously so it cannot
    be cancelled by asyncio. This ensures partial results are always saved
    even if the event loop is shutting down.

    Args:
        all_users: Dict of {user_id: User} from enumeration.
        join_dates: Dict of {user_id: datetime} from enumeration.
        spike_windows: List of (start, end) datetime tuples.
        threshold: Minimum score for candidate inclusion.
        safelist: Set of user IDs to exclude from candidates.
        sub_count: Total channel subscriber count (for stats).
        output_path: File path for CSV/JSON export, or None for stdout.
        interrupted: Whether enumeration was interrupted (triggers partial save).
    """
    # Block Ctrl+C during scoring and export so partial results are never lost.
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        # Score everyone
        all_scored = []
        for uid, user in all_users.items():
            s, reasons = score_user(
                user,
                join_date=join_dates.get(uid),
                spike_windows=spike_windows,
            )
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

        # ── Export results ──
        if interrupted:
            # On interrupt, ALWAYS save ALL scored users (not just
            # above-threshold candidates) so nothing from the enumeration
            # is lost. If --output was provided, use that with a "-partial"
            # suffix. Otherwise, auto-generate a timestamped path in output/.
            if output_path:
                base, ext = output_path.rsplit(".", 1) if "." in output_path else (output_path, "csv")
                partial_path = f"{base}-partial.{ext}"
            else:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                partial_path = f"output/candidates-partial-{ts}.csv"
                ext = "csv"

            Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
            if ext == "json":
                export_json(all_scored, partial_path)
            else:
                export_csv(all_scored, partial_path)
            print(f"\n  Partial results saved: {partial_path} ({len(all_scored)} users)")
        elif output_path:
            if output_path.endswith(".json"):
                export_json(candidates, output_path)
            else:
                export_csv(candidates, output_path)

        # ── Summary ──
        print(f"\n{'=' * 80}")
        label = "PARTIAL CANDIDATE ANALYSIS" if interrupted else "CANDIDATE ANALYSIS"
        print(f"{label} (threshold \u2265{threshold})")
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

        if not output_path:
            # Print candidates to stdout when no output file specified
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

        status_msg = "Partial results saved" if interrupted else "No changes were made"
        print(f"\n{'=' * 80}")
        print(f"Done. {status_msg} \u2014 analysis only.")
        print(f"{'=' * 80}")

    finally:
        # Restore original SIGINT handler
        signal.signal(signal.SIGINT, original_handler)


async def run(args):
    """Execute the candidates command.

    Ctrl+C during enumeration triggers a graceful shutdown: enumeration
    stops, all users collected so far are scored and exported. When no
    --output is provided, a timestamped partial file is auto-generated
    in output/ so data is never lost.

    The client is disconnected BEFORE scoring/export to ensure all file
    I/O runs outside the async context, immune to CancelledError.
    """
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

    # Phase 1: Enumerate (async, Ctrl+C handled by enumerate_subscribers).
    # Client is disconnected in the finally block BEFORE any file I/O,
    # matching the pattern in label.py that's proven reliable.
    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)
        sub_count = getattr(channel, "participants_count", None)

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
    finally:
        # Disconnect first — all subsequent work is sync and safe from
        # asyncio CancelledError propagation.
        await client.disconnect()

    # Phase 2: Score and export (sync, outside async context).
    all_users = result["users"]
    join_dates = result["join_dates"]
    interrupted = result.get("interrupted", False)

    if interrupted:
        print(
            f"\n  Interrupted — scoring and saving {len(all_users)} partial results...",
            flush=True,
        )
    print(f"\nTotal users enumerated: {len(all_users)}")

    if not all_users:
        print("No users enumerated — nothing to export.")
        return

    # Auto-detect spike windows from join dates
    auto_cluster = not getattr(args, "no_auto_cluster", False)
    spike_windows = []
    if auto_cluster and join_dates:
        spike_windows = detect_spike_windows(join_dates)
        if spike_windows:
            print(f"Auto-detected {len(spike_windows)} spike window(s):")
            for start, end in spike_windows:
                print(f"  {start.strftime('%Y-%m-%d %H:%M')} — {end.strftime('%Y-%m-%d %H:%M')} UTC")

    # Score and export — runs synchronously with SIGINT blocked so
    # a second Ctrl+C cannot kill the save.
    output_path = getattr(args, "output", None)
    _score_and_export(
        all_users, join_dates, spike_windows, threshold, safelist,
        sub_count, output_path, interrupted,
    )
