#!/usr/bin/env python3
"""
Purge bot accounts via Telegram Bot API (banChatMember).

Much faster than user MTProto API — bots have higher rate limits (~30/s).
Requires a bot token for a bot that is admin in the target channel with
ban permissions.

Resumes from progress file automatically. Ctrl+C saves progress cleanly.

Usage:
    python scripts/purge_bot_api.py \
        --input output/purge-batch-1-fresh.csv \
        --chat-id -1001923526882 \
        --bot-token "TOKEN"
"""

import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Purge bots via Bot API")
    parser.add_argument("--input", required=True, help="CSV with user_id column")
    parser.add_argument("--chat-id", required=True, type=int, help="Channel chat ID")
    parser.add_argument("--bot-token", required=True, help="Bot API token")
    parser.add_argument("--progress", default=None, help="JSONL progress file")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds between requests (default 0.05)")
    parser.add_argument("--mtproto-progress", default=None,
                        help="Previous MTProto progress file to skip already-banned users")
    args = parser.parse_args()

    api = f"https://api.telegram.org/bot{args.bot_token}"

    # Load user IDs from batch CSV.
    ids = []
    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["user_id"]))
    print(f"Loaded {len(ids)} user IDs from {args.input}")

    # Progress file for resume.
    if not args.progress:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.progress = f"output/purge-bot-progress-{ts}.jsonl"
    Path(args.progress).parent.mkdir(parents=True, exist_ok=True)

    # Load already-processed IDs from all progress sources.
    done = set()
    for ppath in [args.progress, args.mtproto_progress]:
        if ppath and Path(ppath).exists():
            with open(ppath) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["user_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    remaining = [uid for uid in ids if uid not in done]
    print(f"Already done: {len(done)}, Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    # Graceful shutdown on Ctrl+C / SIGHUP / SIGTERM.
    interrupted = False
    def _shutdown(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGHUP, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    banned = 0
    skipped = 0
    errors = 0
    start = time.time()

    with open(args.progress, "a") as pf:
        for i, uid in enumerate(remaining):
            if interrupted:
                print(f"\n  Interrupted at {i}/{len(remaining)}. Progress saved.")
                break

            try:
                # Ban then immediately unban = kick (removes from channel
                # without leaving a permanent ban entry).
                resp = requests.post(f"{api}/banChatMember", json={
                    "chat_id": args.chat_id,
                    "user_id": uid,
                }, timeout=10)
                r = resp.json()

                if r.get("ok"):
                    # Unban to clear the ban list entry.
                    requests.post(f"{api}/unbanChatMember", json={
                        "chat_id": args.chat_id,
                        "user_id": uid,
                        "only_if_banned": True,
                    }, timeout=10)
                    banned += 1
                    status = "banned"
                elif "Too Many Requests" in r.get("description", ""):
                    # Rate limited — wait and retry.
                    retry = r.get("parameters", {}).get("retry_after", 30)
                    print(f"\n  Rate limited at {i+1}/{len(remaining)}, "
                          f"waiting {retry}s...", flush=True)
                    time.sleep(retry + 1)
                    resp = requests.post(f"{api}/banChatMember", json={
                        "chat_id": args.chat_id, "user_id": uid,
                    }, timeout=10)
                    r2 = resp.json()
                    if r2.get("ok"):
                        requests.post(f"{api}/unbanChatMember", json={
                            "chat_id": args.chat_id, "user_id": uid,
                            "only_if_banned": True,
                        }, timeout=10)
                        banned += 1
                        status = "banned"
                    else:
                        errors += 1
                        status = "error"
                elif "USER_NOT_PARTICIPANT" in r.get("description", "") or \
                     "user not found" in r.get("description", "").lower():
                    skipped += 1
                    status = "not_participant"
                elif "not enough rights" in r.get("description", "").lower():
                    print(f"\n  ERROR: Bot lacks ban rights. Stopping.")
                    break
                else:
                    errors += 1
                    status = "error"
                    if errors <= 5:
                        print(f"\n  Error on {uid}: {r.get('description', 'unknown')}",
                              file=sys.stderr)

                entry = {"user_id": uid, "status": status,
                         "ts": datetime.now(timezone.utc).isoformat()}
                pf.write(json.dumps(entry) + "\n")

                # Progress report every 500.
                if (i + 1) % 500 == 0:
                    pf.flush()
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                    print(f"  ...{i+1}/{len(remaining)} "
                          f"({banned} banned, {skipped} left, {errors} err) "
                          f"[{rate:.1f}/s, ETA {eta:.0f}s]", flush=True)

            except Exception as e:
                errors += 1
                entry = {"user_id": uid, "status": "error", "error": str(e),
                         "ts": datetime.now(timezone.utc).isoformat()}
                pf.write(json.dumps(entry) + "\n")

            time.sleep(args.delay)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"PURGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Banned:       {banned}")
    print(f"Already left: {skipped}")
    print(f"Errors:       {errors}")
    print(f"Time:         {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Progress:     {args.progress}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
