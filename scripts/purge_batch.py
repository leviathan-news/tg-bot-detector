#!/usr/bin/env python3
"""
Purge (ban+kick) a batch of users from a Telegram channel.

Reads user IDs from a CSV file and bans each user from the specified
channel using EditBannedRequest. Banned users are immediately removed
from the channel and cannot rejoin.

Progress is logged to a JSONL file so interrupted runs can be resumed.
On Ctrl+C, saves progress and exits cleanly.

Usage:
    python scripts/purge_batch.py \
        --input output/purge-batch.csv \
        --channel @leviathan_news \
        --delay 0.5

Resume after interruption (skips already-processed users):
    python scripts/purge_batch.py \
        --input output/purge-batch.csv \
        --channel @leviathan_news \
        --delay 0.5 \
        --progress output/purge-progress.jsonl
"""

import argparse
import asyncio
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_user_ids(csv_path):
    """Read user IDs from a candidates CSV file."""
    ids = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["user_id"]))
    return ids


def load_progress(progress_path):
    """Load already-processed user IDs from a progress JSONL file."""
    done = set()
    if progress_path and Path(progress_path).exists():
        with open(progress_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry["user_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


async def main():
    parser = argparse.ArgumentParser(description="Purge bot accounts from channel")
    parser.add_argument("--input", required=True, help="CSV file with user_id column")
    parser.add_argument("--channel", required=True, help="Channel username or ID")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between ban requests (default 0.3)")
    parser.add_argument("--progress", default=None,
                        help="JSONL progress file (for resume). Auto-generated if omitted.")
    parser.add_argument("--session-path", default=None,
                        help="Override session file path")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Ban this many users per chunk (default 200)")
    parser.add_argument("--chunk-cooldown", type=float, default=300,
                        help="Seconds to wait between chunks (default 300 = 5min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without banning")
    args = parser.parse_args()

    # Load batch.
    user_ids = load_user_ids(args.input)
    print(f"Loaded {len(user_ids)} user IDs from {args.input}")

    # Progress file for resume capability.
    if not args.progress:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.progress = f"output/purge-progress-{ts}.jsonl"
    Path(args.progress).parent.mkdir(parents=True, exist_ok=True)

    already_done = load_progress(args.progress)
    remaining = [uid for uid in user_ids if uid not in already_done]
    if already_done:
        print(f"Resuming: {len(already_done)} already processed, {len(remaining)} remaining")
    else:
        print(f"Starting fresh: {len(remaining)} users to ban")

    if args.dry_run:
        print(f"DRY RUN — would ban {len(remaining)} users. Exiting.")
        return

    # Connect to Telegram.
    from tg_purge.config import load_config
    from tg_purge.client import create_client, resolve_channel
    from telethon.tl.functions.channels import EditBannedRequest
    from telethon.tl.types import ChatBannedRights, InputPeerUser, InputPeerChannel
    from telethon.errors import (
        FloodWaitError, UserNotParticipantError, ChatAdminRequiredError,
    )

    config = load_config(args.config)
    if args.session_path:
        config.session_path = args.session_path

    client = await create_client(config)
    channel = await resolve_channel(client, args.channel)
    print(f"Channel: {channel.title} (ID: {channel.id})")

    # Ban rights: revoke everything, no expiry.
    ban_rights = ChatBannedRights(
        until_date=None,  # Permanent ban.
        view_messages=True,
        send_messages=True,
        send_media=True,
        send_stickers=True,
        send_gifs=True,
        send_games=True,
        send_inline=True,
        embed_links=True,
    )

    interrupted = False

    # Graceful shutdown handler.
    def _shutdown(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGHUP, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Open progress file for appending.
    progress_f = open(args.progress, "a")
    banned = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    # Process in chunks to avoid FloodWait. Telegram allows ~300 actions
    # before throttling — we use 200 per chunk with a 5-min cooldown.
    chunk_size = args.chunk_size
    cooldown = args.chunk_cooldown
    total_chunks = (len(remaining) + chunk_size - 1) // chunk_size

    try:
        for chunk_idx in range(total_chunks):
            if interrupted:
                break

            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(remaining))
            chunk = remaining[chunk_start:chunk_end]
            chunk_num = chunk_idx + 1

            print(f"\n  Chunk {chunk_num}/{total_chunks}: "
                  f"banning {len(chunk)} users ({chunk_start+1}-{chunk_end}/{len(remaining)})",
                  flush=True)

            for j, uid in enumerate(chunk):
                if interrupted:
                    print(f"\n  Interrupted at chunk {chunk_num}, "
                          f"user {j}/{len(chunk)}. Progress saved.")
                    break

                try:
                    await client(EditBannedRequest(
                        channel=channel,
                        participant=uid,
                        banned_rights=ban_rights,
                    ))
                    banned += 1
                    entry = {"user_id": uid, "status": "banned",
                             "ts": datetime.now(timezone.utc).isoformat()}
                except UserNotParticipantError:
                    skipped += 1
                    entry = {"user_id": uid, "status": "not_participant",
                             "ts": datetime.now(timezone.utc).isoformat()}
                except FloodWaitError as e:
                    # Shouldn't happen with chunking, but handle gracefully.
                    wait = e.seconds + 2
                    print(f"\n  FloodWait {e.seconds}s — sleeping {wait}s...",
                          flush=True)
                    await asyncio.sleep(wait)
                    try:
                        await client(EditBannedRequest(
                            channel=channel,
                            participant=uid,
                            banned_rights=ban_rights,
                        ))
                        banned += 1
                        entry = {"user_id": uid, "status": "banned",
                                 "ts": datetime.now(timezone.utc).isoformat()}
                    except Exception as e2:
                        errors += 1
                        entry = {"user_id": uid, "status": "error",
                                 "error": str(e2),
                                 "ts": datetime.now(timezone.utc).isoformat()}
                except ChatAdminRequiredError:
                    print(f"\n  ERROR: Admin rights required. Cannot ban users.")
                    interrupted = True
                    break
                except Exception as e:
                    errors += 1
                    entry = {"user_id": uid, "status": "error",
                             "error": str(e),
                             "ts": datetime.now(timezone.utc).isoformat()}

                progress_f.write(json.dumps(entry) + "\n")
                progress_f.flush()

                if args.delay > 0:
                    await asyncio.sleep(args.delay)

            # Chunk summary.
            elapsed = time.time() - start_time
            print(f"  Chunk {chunk_num} done: {banned} banned, "
                  f"{skipped} already left, {errors} errors "
                  f"[{elapsed:.0f}s elapsed]", flush=True)

            # Cooldown between chunks (skip after the last chunk).
            if chunk_idx < total_chunks - 1 and not interrupted:
                print(f"  Cooling down {cooldown:.0f}s before next chunk...",
                      flush=True)
                # Sleep in small increments so Ctrl+C is responsive.
                for _ in range(int(cooldown)):
                    if interrupted:
                        break
                    await asyncio.sleep(1)

    finally:
        progress_f.close()
        await client.disconnect()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PURGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processed: {banned + skipped + errors}")
    print(f"Banned:          {banned}")
    print(f"Already left:    {skipped}")
    print(f"Errors:          {errors}")
    print(f"Time:            {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Progress file:   {args.progress}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
