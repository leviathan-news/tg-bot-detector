"""
Extract feature vectors for labeled users without re-enumerating.

Reads datasets/<channel>/labels.json (produced by label --bootstrap),
fetches each labeled user's profile via Telethon GetUsersRequest in
batches of 200, extracts features, and saves features.json.

Usage:
    .venv/bin/python scripts/extract_features_from_labels.py --channel @leviathan_news

This is much faster than re-running label --bootstrap because it skips
the search-based enumeration (11K+ queries) and directly fetches users
by their known IDs.
"""

import asyncio
import json
import os
import stat
import sys
from pathlib import Path

# Add project root to path so tg_purge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.config import load_config
from tg_purge.client import create_client, resolve_channel
from tg_purge.clustering import detect_spike_windows
from tg_purge.features import extract_features
from tg_purge.scoring import score_user
from tg_purge.utils import channel_slug


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract features for labeled users")
    parser.add_argument("--channel", required=True, help="Channel identifier")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    channel_name = args.channel
    slug = channel_slug(channel_name)

    # Load labels to get the user IDs we need.
    labels_path = Path("datasets") / slug / "labels.json"
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        print("Run 'tg-purge label --bootstrap' first.", file=sys.stderr)
        sys.exit(1)

    with open(labels_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    labels = raw.get("labels", {})
    user_ids = [int(uid) for uid in labels.keys()]
    print(f"Loaded {len(user_ids)} user IDs from {labels_path}", file=sys.stderr)

    # Connect to Telegram and resolve channel for join dates.
    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        # Fetch user objects in batches of 200 via GetUsersRequest.
        # This is orders of magnitude faster than search-based enumeration.
        from telethon.tl.functions.users import GetUsersRequest
        from telethon.tl.types import InputUser, InputPeerUser

        all_users = {}
        batch_size = 200
        for i in range(0, len(user_ids), batch_size):
            batch_ids = user_ids[i:i + batch_size]
            # Use InputPeerUser with access_hash=0; Telethon resolves from cache
            # or falls back to a minimal lookup.
            try:
                input_users = [InputUser(uid, 0) for uid in batch_ids]
                users = await client(GetUsersRequest(input_users))
                for u in users:
                    if hasattr(u, "id"):
                        all_users[u.id] = u
            except Exception as e:
                # Some IDs may fail (deleted accounts with no cached access_hash).
                # Fall back to one-by-one for this batch.
                print(f"  Batch {i//batch_size} partial fail: {e}", file=sys.stderr)
                for uid in batch_ids:
                    try:
                        u = await client.get_entity(uid)
                        all_users[u.id] = u
                    except Exception:
                        pass  # User no longer accessible

            if (i // batch_size) % 10 == 0:
                print(
                    f"  Fetched {len(all_users)}/{len(user_ids)} users...",
                    file=sys.stderr,
                )
            await asyncio.sleep(0.5)

        print(f"Fetched {len(all_users)} user profiles.", file=sys.stderr)

        # Fetch join dates from channel participants (for spike detection).
        # Use the same search approach but we only need dates, not discovery.
        from telethon.tl.functions.channels import GetParticipantsRequest
        from telethon.tl.types import ChannelParticipantsRecent

        join_dates = {}
        try:
            result = await client(GetParticipantsRequest(
                channel, ChannelParticipantsRecent(), offset=0, limit=200, hash=0
            ))
            for p in result.participants:
                if hasattr(p, "user_id") and hasattr(p, "date") and p.date:
                    join_dates[p.user_id] = p.date
        except Exception:
            pass

        # Also check if labels.json was produced alongside join date data.
        # The bootstrap run collected join dates; we can only get a subset here.
        print(f"Join dates available: {len(join_dates)}", file=sys.stderr)

    finally:
        await client.disconnect()

    # Detect spike windows.
    spike_windows = []
    if len(join_dates) >= 10:
        spike_windows = detect_spike_windows(join_dates)
        print(f"Spike windows detected: {len(spike_windows)}", file=sys.stderr)

    # Extract features for every fetched user.
    print("Extracting features...", file=sys.stderr)
    feature_cache = {}
    for uid, user in all_users.items():
        features = extract_features(
            user,
            join_date=join_dates.get(uid),
            spike_windows=spike_windows,
        )
        feature_cache[str(uid)] = features

    # Save features.json.
    features_path = str(Path("datasets") / slug / "features.json")
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"channel": channel_name, "features": feature_cache},
            fh,
            indent=2,
        )
    try:
        os.chmod(features_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    print(
        f"\nDone! Saved {len(feature_cache)} feature vectors to {features_path}",
        file=sys.stderr,
    )
    print(
        f"Now run:  .venv/bin/python -m tg_purge ml train --channel {channel_name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
