"""Fetch Telegram profiles for the most uncertain ML predictions.

Active learning helper: identifies unlabeled users where the model is most
uncertain (probability closest to 0.5), then fetches their live Telegram
profiles via GetParticipantRequest for human review. Outputs JSON to stdout.

Usage:
    python scripts/fetch_uncertain_profiles.py --channel @leviathan_news
    python scripts/fetch_uncertain_profiles.py --channel @leviathan_news --top 60 --delay 0.2
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.config import load_config
from tg_purge.client import create_client, resolve_channel
from tg_purge.ml import predict


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch profiles for the model's most uncertain predictions."
    )
    parser.add_argument(
        "--channel", required=True,
        help="Channel @username or numeric ID to fetch participants from.",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of most uncertain users to fetch (default: 50).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="Delay between API calls in seconds (default: 0.15).",
    )
    args = parser.parse_args()

    # Derive dataset paths from channel name (strip @ prefix)
    channel_name = args.channel.lstrip("@")
    base = Path("datasets") / channel_name
    labels_path = base / "labels.json"
    features_path = base / "features.json"
    model_path = Path("models") / f"{channel_name}_sklearn_rf.joblib"

    for path, desc in [
        (labels_path, "Labels"),
        (features_path, "Features"),
        (model_path, "Model"),
    ]:
        if not path.exists():
            print("%s not found: %s" % (desc, path), file=sys.stderr)
            sys.exit(1)

    with open(labels_path) as f:
        raw = json.load(f)
    with open(features_path) as f:
        features = json.load(f)["features"]

    # Collect unlabeled users with feature vectors
    unlabeled = []
    for uid_str, info in raw["labels"].items():
        if info["label"] == "unlabeled":
            feat = features.get(uid_str, {})
            if feat:
                unlabeled.append({"uid": uid_str, "features": feat})

    if not unlabeled:
        print("No unlabeled users with features found.", file=sys.stderr)
        sys.exit(0)

    # Run predictions and rank by uncertainty
    feature_dicts = [u["features"] for u in unlabeled]
    results = predict(feature_dicts, str(model_path))

    uncertain = []
    for u, r in zip(unlabeled, results):
        uncertain.append((u["uid"], r["probability"], r["label"]))
    uncertain.sort(key=lambda x: abs(x[1] - 0.5))

    # Take top N most uncertain user IDs
    top_entries = uncertain[: args.top]
    top_uids = [int(uid) for uid, _, _ in top_entries]
    uid_to_prob = {int(uid): (prob, pred) for uid, prob, pred in top_entries}

    # Connect to Telegram and fetch profiles
    config = load_config()
    client = await create_client(config)
    channel = await resolve_channel(client, args.channel)

    from telethon.tl.functions.channels import GetParticipantRequest

    output = []
    for i, uid in enumerate(top_uids):
        try:
            p = await client(GetParticipantRequest(channel, uid))
            u = p.users[0] if p.users else None
            if u:
                feat = features.get(str(u.id), {})
                status = type(u.status).__name__ if u.status else "None"
                prob, pred = uid_to_prob[u.id]
                output.append({
                    "id": u.id,
                    "name": (
                        (u.first_name or "") + " " + (u.last_name or "")
                    ).strip(),
                    "username": u.username or "",
                    "photo": bool(u.photo),
                    "premium": bool(u.premium),
                    "status": status,
                    "h_score": int(feat.get("heuristic_score", 0)),
                    "spike": bool(feat.get("is_spike_join")),
                    "days": int(feat.get("days_since_join", -1)),
                    "ml_prob": round(prob, 3),
                    "ml_pred": pred,
                })
        except Exception:
            pass  # user may have left channel or be inaccessible

        await asyncio.sleep(args.delay)
        if (i + 1) % 10 == 0:
            print("  ...%d/%d fetched" % (i + 1, len(top_uids)), file=sys.stderr)

    await client.disconnect()

    # Output JSON to stdout for piping/processing
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(
        "\nFetched %d/%d profiles." % (len(output), len(top_uids)),
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
