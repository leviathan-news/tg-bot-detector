"""Active learning review: ML-guided interactive label review.

Combines model uncertainty ranking with live profile fetching and
interactive review, saving results to a separate human_reviews.json
that is never touched by `label --bootstrap`.

Workflow:
  1. Load model predictions for unlabeled users
  2. Rank by uncertainty (probability closest to 0.5)
  3. Fetch live Telegram profiles for the most uncertain users
  4. Present them for interactive human review
  5. Save decisions to datasets/<channel>/human_reviews.json

The separate file prevents bootstrap from overwriting human work.
Reviews are merged into training via train_ground_truth.py.

Usage:
    python scripts/active_review.py --channel @leviathan_news
    python scripts/active_review.py --channel @leviathan_news --top 100 --batch-size 20
    python scripts/active_review.py --channel @leviathan_news --mode heuristic

Controls during review:
    b  = bot
    h  = human
    s  = skip (keep unlabeled)
    q  = quit and save progress
"""

import argparse
import asyncio
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.config import load_config
from tg_purge.client import create_client, resolve_channel
from tg_purge.utils import channel_slug


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def _format_user(profile: dict, index: int, total: int) -> str:
    """Format a user profile for terminal display.

    Renders a compact block showing all relevant signals so the reviewer
    can make a quick bot/human decision. Includes ML probability and
    prediction alongside heuristic score for comparison.

    Args:
        profile: Dict with user profile fields from fetch_profiles().
        index:   1-based position in the current review batch.
        total:   Total number of users in the current batch.

    Returns:
        Formatted multi-line string for terminal output.
    """
    # Build name display.
    name_parts = [profile.get("first_name", "")]
    if profile.get("last_name"):
        name_parts.append(profile["last_name"])
    name = " ".join(name_parts).strip() or "(no name)"

    username = profile.get("username", "")
    username_display = f"@{username}" if username else "—"

    # Collect signal indicators.
    signals = []
    if profile.get("photo"):
        signals.append("photo")
    else:
        signals.append("NO photo")

    if profile.get("premium"):
        signals.append("PREMIUM")

    if profile.get("deleted"):
        signals.append("DELETED")

    if profile.get("bot_flag"):
        signals.append("BOT API")

    if profile.get("spike"):
        signals.append("spike join")

    status = profile.get("status", "unknown")
    h_score = profile.get("h_score", 0)
    ml_prob = profile.get("ml_prob", -1)
    ml_pred = profile.get("ml_pred", "?")

    # ML confidence indicator: how far from 0.5 the prediction is.
    # Closer to 0.5 = more uncertain, which is why it's selected.
    if ml_prob >= 0:
        confidence = abs(ml_prob - 0.5) * 200  # 0-100 scale
        ml_line = (
            f"  ML:       prob={ml_prob:.3f}  pred={ml_pred}  "
            f"confidence={confidence:.0f}%"
        )
    else:
        ml_line = f"  ML:       (no prediction)"

    return (
        f"\n{'─' * 60}\n"
        f"  [{index}/{total}]  ID: {profile['id']}\n"
        f"  Name:     {name}\n"
        f"  Username: {username_display}\n"
        f"  Status:   {status}  |  Score: {h_score}\n"
        f"  Signals:  {', '.join(signals)}\n"
        f"{ml_line}\n"
        f"{'─' * 60}"
    )


# ---------------------------------------------------------------------------
# Profile fetching
# ---------------------------------------------------------------------------

async def fetch_profiles(client, channel, user_ids: list, features: dict,
                         delay: float = 0.15) -> list:
    """Fetch live Telegram profiles for a list of user IDs.

    Uses GetParticipantRequest (one at a time) since these are channel
    participants we haven't DMed. Includes rate limiting to avoid flood.

    Args:
        client:   Connected TelegramClient.
        channel:  Resolved channel entity.
        user_ids: List of integer user IDs to fetch.
        features: Dict of uid_str -> feature dict for score/signal lookup.
        delay:    Seconds between API calls (default 0.15).

    Returns:
        List of dicts with user profile fields ready for display.
    """
    from telethon.tl.functions.channels import GetParticipantRequest

    results = []
    for i, uid in enumerate(user_ids):
        try:
            p = await client(GetParticipantRequest(channel, uid))
            u = p.users[0] if p.users else None
            if u:
                feat = features.get(str(u.id), {})
                status_type = type(u.status).__name__ if u.status else "None"
                results.append({
                    "id": u.id,
                    "first_name": u.first_name or "",
                    "last_name": u.last_name or "",
                    "username": u.username or "",
                    "photo": bool(u.photo),
                    "premium": bool(u.premium),
                    "deleted": bool(u.deleted),
                    "bot_flag": bool(u.bot),
                    "status": status_type,
                    "h_score": int(feat.get("heuristic_score", 0)),
                    "spike": bool(feat.get("is_spike_join")),
                    # ML prediction data is injected by caller after fetch.
                    "ml_prob": -1,
                    "ml_pred": "?",
                })
        except Exception as e:
            # User may have left the channel or been deleted since scoring.
            print(f"  Could not fetch {uid}: {e}", file=sys.stderr)

        # Rate limit: heavier throttle every 30 requests.
        if (i + 1) % 30 == 0:
            await asyncio.sleep(1.0)
        else:
            await asyncio.sleep(delay)

        # Progress indicator.
        if (i + 1) % 10 == 0:
            print(
                f"  Fetching profiles... {i + 1}/{len(user_ids)}",
                file=sys.stderr,
            )

    return results


# ---------------------------------------------------------------------------
# Human reviews file I/O
# ---------------------------------------------------------------------------

def load_human_reviews(path: Path) -> dict:
    """Load existing human reviews from the dedicated reviews file.

    Args:
        path: Path to human_reviews.json.

    Returns:
        Dict with "channel" (str) and "reviews" (dict uid_str -> info).
    """
    if not path.exists():
        return {"channel": "", "reviews": {}}

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return {
        "channel": data.get("channel", ""),
        "reviews": data.get("reviews", {}),
    }


def save_human_reviews(data: dict, path: Path) -> None:
    """Write human reviews to the dedicated file with PII-safe permissions.

    Args:
        data: Dict with "channel" and "reviews" keys.
        path: Path to human_reviews.json.
    """
    # Ensure parent directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write with indentation for readability.
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Restrict permissions — contains Telegram user IDs (PII).
    try:
        os.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Uncertainty ranking
# ---------------------------------------------------------------------------

def rank_by_ml_uncertainty(labels_raw: dict, features: dict,
                           model_path: str, score_filter: int = None,
                           label_filter: str = "unlabeled") -> list:
    """Rank unlabeled users by ML model uncertainty.

    Loads the trained model, predicts probabilities for all unlabeled
    users, and sorts by distance from 0.5 (ascending = most uncertain
    first).

    Args:
        labels_raw:  Raw labels dict from labels.json (string keys).
        features:    Dict of uid_str -> feature dict.
        model_path:  Path to the trained model file.

    Returns:
        List of (uid_str, probability, predicted_label) sorted by
        uncertainty (most uncertain first).
    """
    from tg_purge.ml import predict

    # Collect users matching the label filter that have feature vectors.
    unlabeled = []
    for uid_str, info in labels_raw.get("labels", {}).items():
        if label_filter != "all" and info.get("label") != label_filter:
            continue
        feat = features.get(uid_str)
        if not feat:
            continue
        # Filter by exact heuristic score if specified.
        if score_filter is not None:
            h_score = int(feat.get("heuristic_score", 0))
            if h_score != score_filter:
                continue
        unlabeled.append((uid_str, feat))

    if not unlabeled:
        return []

    # Run batch prediction.
    feature_dicts = [feat for _, feat in unlabeled]
    results = predict(feature_dicts, model_path)

    # Pair with UIDs and sort by uncertainty.
    ranked = []
    for (uid_str, _), result in zip(unlabeled, results):
        ranked.append((uid_str, result["probability"], result["label"]))

    # Sort: closest to 0.5 first (most uncertain).
    ranked.sort(key=lambda x: abs(x[1] - 0.5))

    return ranked


def rank_by_heuristic_score(labels_raw: dict, features: dict,
                           score_filter: int = None,
                           label_filter: str = "unlabeled") -> list:
    """Rank users by heuristic score.

    Supports filtering by specific score value and bootstrap label type.
    Without filters, returns unlabeled users sorted by score descending.

    Args:
        labels_raw:    Raw labels dict from labels.json (string keys).
        features:      Dict of uid_str -> feature dict.
        score_filter:  If set, only include users with this exact score.
        label_filter:  Bootstrap label to filter: "unlabeled", "human",
                       "bot", or "all" (default: "unlabeled").

    Returns:
        List of (uid_str, heuristic_score, "?") sorted by score desc.
    """
    candidates = []
    for uid_str, info in labels_raw.get("labels", {}).items():
        # Filter by bootstrap label.
        if label_filter != "all" and info.get("label") != label_filter:
            continue
        feat = features.get(uid_str, {})
        h_score = int(feat.get("heuristic_score", 0))
        # Filter by exact score if specified.
        if score_filter is not None and h_score != score_filter:
            continue
        candidates.append((uid_str, h_score, "?"))

    # Sort: highest heuristic score first (most likely bots → most informative).
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Interactive review loop
# ---------------------------------------------------------------------------

def review_loop(profiles: list, reviews_data: dict, reviews_path: Path,
                batch_size: int) -> tuple:
    """Run the interactive review loop on fetched profiles.

    Presents users in batches, collects bot/human/skip decisions,
    and saves progress after each batch to human_reviews.json.

    Args:
        profiles:     List of profile dicts (with ml_prob/ml_pred injected).
        reviews_data: Current human_reviews data (loaded from file).
        reviews_path: Path to human_reviews.json for incremental saving.
        batch_size:   Number of profiles per batch.

    Returns:
        Tuple of (total_reviewed, total_labeled) counts.
    """
    total_reviewed = 0
    total_labeled = 0
    quit_requested = False

    batch_start = 0
    while batch_start < len(profiles) and not quit_requested:
        batch = profiles[batch_start:batch_start + batch_size]

        print(
            f"\n{'=' * 60}\n"
            f"  Batch {batch_start // batch_size + 1} "
            f"({len(batch)} users, #{batch_start + 1}-"
            f"#{batch_start + len(batch)} of {len(profiles)})\n"
            f"{'=' * 60}",
            file=sys.stderr,
        )

        for i, profile in enumerate(batch):
            print(_format_user(profile, i + 1, len(batch)))
            print("  Label: [b]ot  [h]uman  [s]kip  [q]uit+save")

            while True:
                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    # Ctrl+C or Ctrl+D — treat as quit.
                    choice = "q"

                if choice in ("b", "bot"):
                    new_label = "bot"
                    break
                elif choice in ("h", "human"):
                    new_label = "human"
                    break
                elif choice in ("s", "skip", ""):
                    new_label = None
                    break
                elif choice in ("q", "quit"):
                    quit_requested = True
                    new_label = None
                    break
                else:
                    print("  Invalid input. Use: b, h, s, or q")

            if quit_requested:
                break

            total_reviewed += 1

            if new_label is not None:
                uid_str = str(profile["id"])
                reviews_data["reviews"][uid_str] = {
                    "label": new_label,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                total_labeled += 1
                print(f"  -> Labeled as {new_label.upper()}")

        batch_start += batch_size

        # Save after each batch to avoid losing progress on crash/disconnect.
        if total_labeled > 0:
            save_human_reviews(reviews_data, reviews_path)
            print(
                f"\n  Saved progress: {total_labeled} labeled, "
                f"{total_reviewed} reviewed so far.",
                file=sys.stderr,
            )

    return total_reviewed, total_labeled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Active learning: ML-guided interactive label review"
    )
    parser.add_argument(
        "--channel", required=True,
        help="Channel identifier (e.g. @leviathan_news)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.toml (optional)",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of most uncertain users to review (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, dest="batch_size",
        help="Users per review batch (default: 20)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="Delay between Telegram API calls in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--mode", choices=["ml", "heuristic"], default="ml",
        help=(
            "Ranking mode: 'ml' uses model uncertainty (default), "
            "'heuristic' uses raw heuristic scores (no model required)"
        ),
    )
    parser.add_argument(
        "--skip-reviewed", action="store_true", dest="skip_reviewed",
        help="Skip users that already have human reviews",
    )
    parser.add_argument(
        "--score", type=int, default=None,
        help="Only review users with this heuristic score (e.g. --score 0 for humans)",
    )
    parser.add_argument(
        "--label-filter", default="unlabeled", dest="label_filter",
        help=(
            "Which bootstrap label to review: 'unlabeled' (default), "
            "'human', 'bot', or 'all'"
        ),
    )
    args = parser.parse_args()

    slug = channel_slug(args.channel)
    base = Path("datasets") / slug

    labels_path = base / "labels.json"
    features_path = base / "features.json"
    reviews_path = base / "human_reviews.json"

    # Validate required files exist.
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)
    if not features_path.exists():
        print(f"Features file not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    # Load labels and features.
    with open(labels_path, "r", encoding="utf-8") as fh:
        labels_raw = json.load(fh)
    with open(features_path, "r", encoding="utf-8") as fh:
        features = json.load(fh).get("features", {})

    # Load existing human reviews (to append to, not overwrite).
    reviews_data = load_human_reviews(reviews_path)
    reviews_data["channel"] = args.channel
    existing_count = len(reviews_data["reviews"])

    if existing_count > 0:
        print(
            f"Loaded {existing_count} existing human reviews from {reviews_path}",
            file=sys.stderr,
        )

    # Rank unlabeled users by the chosen strategy.
    if args.mode == "ml":
        # Find the model file.
        model_candidates = [
            Path("models") / f"{slug}_sklearn_rf.joblib",
            Path("models") / f"{slug}_lightgbm.model",
        ]
        model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                model_path = str(candidate)
                break

        if model_path is None:
            print(
                "No trained model found. Use --mode heuristic or train first:\n"
                f"  python scripts/train_ground_truth.py\n\n"
                f"Checked: {', '.join(str(c) for c in model_candidates)}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Ranking by ML uncertainty (model: {model_path})...", file=sys.stderr)
        ranked = rank_by_ml_uncertainty(
            labels_raw, features, model_path,
            score_filter=args.score,
            label_filter=args.label_filter,
        )
    else:
        score_desc = f" (score={args.score})" if args.score is not None else ""
        label_desc = f" label={args.label_filter}" if args.label_filter != "unlabeled" else ""
        print(f"Ranking by heuristic score{score_desc}{label_desc}...", file=sys.stderr)
        ranked = rank_by_heuristic_score(
            labels_raw, features,
            score_filter=args.score,
            label_filter=args.label_filter,
        )

    if not ranked:
        print("No unlabeled users to review.", file=sys.stderr)
        sys.exit(0)

    # Filter out already-reviewed users if requested.
    if args.skip_reviewed:
        before = len(ranked)
        ranked = [
            (uid_str, prob, pred) for uid_str, prob, pred in ranked
            if uid_str not in reviews_data["reviews"]
        ]
        skipped = before - len(ranked)
        if skipped > 0:
            print(f"Skipping {skipped} already-reviewed users.", file=sys.stderr)

    # Take the top N most uncertain/informative users.
    selected = ranked[:args.top]
    selected_uids = [int(uid_str) for uid_str, _, _ in selected]

    # Build uid -> ML prediction lookup for injecting into profiles.
    uid_to_ml = {
        int(uid_str): (prob, pred) for uid_str, prob, pred in selected
    }

    print(
        f"\nSelected {len(selected)} users for review "
        f"(from {len(ranked)} unlabeled).",
        file=sys.stderr,
    )

    if args.mode == "ml":
        # Show uncertainty distribution of selected users.
        probs = [prob for _, prob, _ in selected]
        print(
            f"  Probability range: {min(probs):.3f} - {max(probs):.3f}\n"
            f"  Median uncertainty: {sorted(probs)[len(probs)//2]:.3f}",
            file=sys.stderr,
        )

    # Connect to Telegram and fetch live profiles.
    config = load_config(args.config)
    client = await create_client(config)
    try:
        channel = await resolve_channel(client, args.channel)

        print(
            f"\nFetching {len(selected_uids)} live profiles...",
            file=sys.stderr,
        )
        profiles = await fetch_profiles(
            client, channel, selected_uids, features, delay=args.delay,
        )
    finally:
        await client.disconnect()

    if not profiles:
        print("No profiles fetched. Users may have left the channel.", file=sys.stderr)
        sys.exit(1)

    # Inject ML prediction data into each profile for display.
    for profile in profiles:
        uid = profile["id"]
        if uid in uid_to_ml:
            prob, pred = uid_to_ml[uid]
            profile["ml_prob"] = round(prob, 3) if isinstance(prob, float) else prob
            profile["ml_pred"] = pred

    print(
        f"\nFetched {len(profiles)} profiles. Starting interactive review.\n"
        f"  Reviews will be saved to: {reviews_path}\n"
        f"  (This file is SEPARATE from labels.json — bootstrap won't touch it.)",
        file=sys.stderr,
    )

    # Run the interactive review loop.
    total_reviewed, total_labeled = review_loop(
        profiles, reviews_data, reviews_path, args.batch_size,
    )

    # Final save.
    if total_labeled > 0:
        save_human_reviews(reviews_data, reviews_path)

    # Print summary.
    total_reviews = len(reviews_data["reviews"])
    bot_count = sum(
        1 for info in reviews_data["reviews"].values()
        if info.get("label") == "bot"
    )
    human_count = sum(
        1 for info in reviews_data["reviews"].values()
        if info.get("label") == "human"
    )

    print(
        f"\n{'=' * 60}\n"
        f"  Review session complete!\n"
        f"  This session:  reviewed={total_reviewed}  labeled={total_labeled}\n\n"
        f"  Total human reviews: {total_reviews}\n"
        f"    Bot:   {bot_count}\n"
        f"    Human: {human_count}\n"
        f"{'=' * 60}\n\n"
        f"  To retrain with updated reviews:\n"
        f"    python scripts/train_ground_truth.py --save-dataset\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
