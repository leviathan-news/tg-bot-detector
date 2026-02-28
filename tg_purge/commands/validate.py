"""
Score known-good users to measure false positive rate.

Generic version of known-user-bot-score.py (no Django dependency).
Accepts CSV or JSON of known-good user IDs or usernames, searches for each
on the target channel, scores them, and reports the false positive rate
at each threshold.

CSV format: one column named 'user_id' or 'username' (or first column used)
JSON format: list of objects with 'user_id' or 'username' keys, or plain list of IDs
"""

import asyncio
import csv
import json
from collections import Counter
from pathlib import Path

from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import ChannelParticipantsSearch

from ..client import create_client, resolve_channel
from ..config import load_config
from ..scoring import score_user, format_name, status_label
from ..formatters import print_score_distribution, print_signal_frequency


def _load_known_users(path):
    """Load known user identifiers from CSV or JSON.

    Returns:
        List of dicts with 'user_id' and/or 'username' keys.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return data
            else:
                # Plain list of IDs
                return [{"user_id": uid} for uid in data]
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

    elif suffix == ".csv":
        entries = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                # No header — treat as single column of IDs
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append({"user_id": int(line)})
                        except ValueError:
                            entries.append({"username": line})
                return entries

            for row in reader:
                entry = {}
                if "user_id" in row:
                    try:
                        entry["user_id"] = int(row["user_id"])
                    except (ValueError, TypeError):
                        pass
                if "username" in row:
                    entry["username"] = row["username"]
                # Fallback: use first column
                if not entry and reader.fieldnames:
                    val = row[reader.fieldnames[0]]
                    try:
                        entry["user_id"] = int(val)
                    except ValueError:
                        entry["username"] = val
                if entry:
                    entries.append(entry)
        return entries

    else:
        # Try plain text: one ID/username per line
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        entries.append({"user_id": int(line)})
                    except ValueError:
                        entries.append({"username": line.lstrip("@")})
        return entries


async def run(args):
    """Execute the validate command."""
    config = load_config(getattr(args, "config", None))
    if args.session_path:
        config.session_path = args.session_path
    if getattr(args, "delay", None) is not None:
        config.delay = args.delay
    channel_name = config.resolve_channel(args.channel)

    known_users = _load_known_users(args.known_users)
    print(f"Loaded {len(known_users)} known-good users from {args.known_users}")

    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        results = []    # (entry, tg_user, score, reasons)
        not_found = []
        errors = []

        searchable = [e for e in known_users if e.get("username")]
        id_only = [e for e in known_users if not e.get("username") and e.get("user_id")]

        total = len(searchable) + len(id_only)
        print(f"Searchable by username: {len(searchable)}")
        print(f"ID-only (will attempt search): {len(id_only)}")
        print(f"Looking up each on {channel_name}...", flush=True)

        # Search by username
        for i, entry in enumerate(searchable):
            username = entry["username"]
            target_id = entry.get("user_id")
            try:
                result = await client(GetParticipantsRequest(
                    channel=channel,
                    filter=ChannelParticipantsSearch(username),
                    offset=0,
                    limit=10,
                    hash=0,
                ))

                tg_user = None
                for u in result.users:
                    if target_id and u.id == target_id:
                        tg_user = u
                        break
                    if u.username and u.username.lower() == username.lower():
                        tg_user = u
                        break

                if tg_user:
                    s, reasons = score_user(tg_user)
                    results.append((entry, tg_user, s, reasons))
                else:
                    not_found.append(entry)

            except Exception as e:
                errors.append((entry, str(e)))

            if (i + 1) % 25 == 0:
                print(f"  ...checked {i + 1}/{total}", flush=True)

            await asyncio.sleep(config.delay)

        # Sort by score descending
        results.sort(key=lambda x: -x[2])

        # ── Print Results ─────────────────────────────────────────
        print(f"\n{'=' * 100}")
        print("KNOWN USERS SCORED BY BOT HEURISTICS")
        print(f"{'=' * 100}")
        print(f"{'Verdict':6s} {'Score':5s} | {'Identifier':20s} | {'TG Profile':35s} | {'Status':12s} | Signals")
        print(f"{'─' * 100}")

        for entry, tg_user, s, reasons in results:
            verdict = "BOT?" if s >= 2 else "OK"
            identifier = (entry.get("username") or str(entry.get("user_id", "?")))[:20]
            tg_name = format_name(tg_user)[:35]
            status = status_label(tg_user)
            reason_str = ", ".join(reasons) if reasons else "clean"
            print(f"[{verdict:4s}] {s:3d}   | {identifier:20s} | {tg_name:35s} | {status:12s} | {reason_str}")

        # ── Summary ───────────────────────────────────────────────
        total_found = len(results)
        flagged = sum(1 for _, _, s, _ in results if s >= 2)
        borderline = sum(1 for _, _, s, _ in results if s == 1)
        clean = sum(1 for _, _, s, _ in results if s <= 0)

        print(f"\n{'=' * 100}")
        print("SUMMARY")
        print(f"{'=' * 100}")
        print(f"Known-good users loaded:         {len(known_users)}")
        print(f"Found as channel subscribers:    {total_found}")
        print(f"Not subscribed / not found:      {len(not_found)}")
        print(f"Errors:                          {len(errors)}")

        if total_found > 0:
            print(f"\nOf the {total_found} subscribed known users:")
            print(f"  Would be FLAGGED (score \u22652): {flagged:4d}  ({flagged/total_found*100:.1f}%)")
            print(f"  Borderline (score 1):        {borderline:4d}  ({borderline/total_found*100:.1f}%)")
            print(f"  Safe (score \u22640):             {clean:4d}  ({clean/total_found*100:.1f}%)")

        # False positive rate at each threshold
        print(f"\n{'─' * 100}")
        print("FALSE POSITIVE RATE AT EACH THRESHOLD")
        print(f"{'─' * 100}")
        if total_found > 0:
            for threshold in [1, 2, 3, 4, 5]:
                fp = sum(1 for _, _, s, _ in results if s >= threshold)
                print(f"  Threshold \u2265{threshold}: {fp:4d} / {total_found} known-good users flagged ({fp/total_found*100:.1f}%)")

        # Score distribution and signals
        scored_simple = [(u, s, r) for _, u, s, r in results]
        print_score_distribution(scored_simple, "SCORE DISTRIBUTION (known users)")
        print_signal_frequency(scored_simple, "SIGNALS TRIGGERING FALSE POSITIVES", top_n=15)

        if errors:
            print(f"\n{'─' * 100}")
            print(f"ERRORS ({len(errors)})")
            print(f"{'─' * 100}")
            for entry, err in errors[:10]:
                identifier = entry.get("username") or str(entry.get("user_id", "?"))
                print(f"  {identifier}: {err}")

    finally:
        await client.disconnect()

    print(f"\n{'=' * 100}")
    print("Done. No changes were made \u2014 read-only analysis.")
    print(f"{'=' * 100}")
