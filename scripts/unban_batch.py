#!/usr/bin/env python3
"""
Unban previously banned users via Telegram Bot API.

After banChatMember removes a user from a channel, calling
unbanChatMember lifts the permanent ban so they could theoretically
rejoin (but bots won't). This keeps the channel's ban list clean.

Usage:
    python scripts/unban_batch.py \
        --input output/banned-ids-all.txt \
        --chat-id -1001923526882 \
        --bot-token "TOKEN"
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Unban users via Bot API")
    parser.add_argument("--input", required=True,
                        help="Text file with one user_id per line")
    parser.add_argument("--chat-id", required=True, type=int)
    parser.add_argument("--bot-token", required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    api = f"https://api.telegram.org/bot{args.bot_token}"

    # Load user IDs.
    with open(args.input) as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(ids)} user IDs to unban")

    interrupted = False
    def _stop(s, f):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGHUP, _stop)
    signal.signal(signal.SIGTERM, _stop)

    unbanned = 0
    errors = 0
    start = time.time()

    for i, uid in enumerate(ids):
        if interrupted:
            print(f"\nInterrupted at {i}/{len(ids)}")
            break

        try:
            resp = requests.post(f"{api}/unbanChatMember", json={
                "chat_id": args.chat_id,
                "user_id": uid,
                "only_if_banned": True,
            }, timeout=10)
            r = resp.json()

            if r.get("ok"):
                unbanned += 1
            elif "Too Many Requests" in r.get("description", ""):
                retry = r.get("parameters", {}).get("retry_after", 30)
                print(f"\n  Rate limit at {i+1}, wait {retry}s...", flush=True)
                time.sleep(retry + 1)
                resp = requests.post(f"{api}/unbanChatMember", json={
                    "chat_id": args.chat_id, "user_id": uid,
                    "only_if_banned": True,
                }, timeout=10)
                if resp.json().get("ok"):
                    unbanned += 1
                else:
                    errors += 1
            else:
                errors += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  ...{i+1}/{len(ids)} ({unbanned} unbanned, {errors} err) "
                      f"[{rate:.1f}/s]", flush=True)

        except Exception as e:
            errors += 1

        time.sleep(args.delay)

    elapsed = time.time() - start
    print(f"\nUnbanned: {unbanned} | Errors: {errors} | Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
