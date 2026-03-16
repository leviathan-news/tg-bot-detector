#!/usr/bin/env python3
"""
Re-score an existing candidates CSV with ML predictions.

Instead of re-fetching users from Telegram (rate-limited), this script
reads the partial candidates CSV and the fresh enumeration data from the
candidates command, then runs the ML model on the heuristic-scored users.

The trick: candidates --scoring hybrid already does this. But when we
have partial enumeration results saved by Ctrl+C, we can run ML on those
results offline using the cached feature vectors from the enumeration.

Simpler approach: just re-run candidates with --scoring hybrid and accept
the re-enumeration cost. But for 11K users already enumerated, we can
extract features from the CSV signals and run a lightweight prediction.

Usage:
    python scripts/ml_rescore_candidates.py \
        --input output/candidates-20260316-partial.csv \
        --output output/candidates-ml-20260316.csv \
        --model models/leviathan_news_lightgbm.model
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.ml import predict, ml_available, load_model_metadata


def parse_signals_to_features(row):
    """Convert a candidates CSV row into a partial feature vector.

    Extracts what we can from the heuristic signals string and the
    metadata columns. This won't have all 49 features the model expects
    but predict() pads missing keys with 0.0.
    """
    signals = row.get("signals", "")
    score = int(row["score"])
    name = row.get("name", "")
    username = row.get("username", "")
    status = row.get("status", "")

    features = {}

    # Heuristic score.
    features["heuristic_score"] = float(score)

    # Name features.
    features["first_name_length"] = float(len(name)) if name else 0.0
    features["has_last_name"] = 0.0  # CSV doesn't separate first/last
    features["has_username"] = 1.0 if username else 0.0
    features["name_digit_ratio"] = sum(c.isdigit() for c in name) / max(len(name), 1)
    features["name_emoji_count"] = float(len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F900-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U0000FE00-\U0000FE0F"
        r"\U0000200D\U00002640\U00002642]", name)))
    features["script_count"] = float(len(set(
        "latin" if c.isascii() and c.isalpha() else
        "cyrillic" if "\u0400" <= c <= "\u04ff" else
        "arabic" if "\u0600" <= c <= "\u06ff" else
        "cjk" if "\u4e00" <= c <= "\u9fff" else
        "other"
        for c in name if c.isalpha()
    )))

    # Signal-derived flags.
    features["is_deleted"] = 1.0 if "deleted_account" in signals else 0.0
    features["is_scam"] = 1.0 if "scam_flag" in signals else 0.0
    features["is_fake"] = 1.0 if "fake_flag" in signals else 0.0
    features["is_restricted"] = 1.0 if "restricted" in signals else 0.0
    features["is_bot"] = 1.0 if "bot_flag" in signals else 0.0
    features["is_premium"] = 1.0 if "premium" in signals else 0.0
    features["has_photo"] = 0.0 if "no_photo" in signals else 1.0
    features["has_emoji_status"] = 1.0 if "emoji_status" in signals else 0.0
    features["is_spike_join"] = 1.0 if "spike_join" in signals else 0.0

    # Status features — parse from the status column.
    features["status_online"] = 1.0 if status == "online" else 0.0
    features["status_recently"] = 1.0 if status == "recently" else 0.0
    features["status_empty"] = 1.0 if status in ("no status", "") else 0.0

    # Parse offline duration if present (e.g., "offline 410d").
    offline_match = re.match(r"offline (\d+)d", status)
    if offline_match:
        features["status_offline"] = 1.0
        features["days_since_last_seen"] = float(offline_match.group(1))
    else:
        features["status_offline"] = 0.0
        features["days_since_last_seen"] = 0.0

    features["status_last_week"] = 1.0 if status == "last week" else 0.0
    features["status_last_month"] = 1.0 if status == "last month" else 0.0

    return features


def main():
    parser = argparse.ArgumentParser(description="Re-score candidates with ML (offline)")
    parser.add_argument("--input", required=True, help="Input candidates CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="models/leviathan_news_lightgbm.model",
                        help="Path to model file")
    args = parser.parse_args()

    if not ml_available():
        print("ERROR: ML dependencies not installed.")
        sys.exit(1)

    # Load model metadata to see what features it expects.
    meta_path = args.model.rsplit(".", 1)[0] + ".json"
    meta = load_model_metadata(meta_path)
    print(f"Model: {meta['algorithm']}, {len(meta['feature_names'])} features")

    # Read candidates and build feature vectors.
    rows = []
    features_list = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            features_list.append(parse_signals_to_features(row))

    print(f"Loaded {len(rows)} candidates from {args.input}")

    # Check feature coverage.
    available = set(features_list[0].keys()) if features_list else set()
    expected = set(meta["feature_names"])
    covered = available & expected
    missing = expected - available
    print(f"Feature coverage: {len(covered)}/{len(expected)} "
          f"({len(covered)/len(expected)*100:.0f}%)")
    if missing:
        print(f"Missing features (will default to 0.0): {sorted(missing)}")

    # Run ML inference.
    print(f"\nRunning inference...")
    predictions = predict(features_list, args.model)

    # Merge results.
    output_rows = []
    for row, pred in zip(rows, predictions):
        output_rows.append({
            "user_id": row["user_id"],
            "name": row["name"],
            "username": row["username"],
            "heuristic_score": row["score"],
            "ml_probability": f"{pred['probability']:.4f}",
            "ml_label": pred["label"],
            "status": row["status"],
            "signals": row["signals"],
        })

    # Sort by ML probability descending.
    output_rows.sort(key=lambda r: -float(r["ml_probability"]))

    # Write output.
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id", "name", "username", "heuristic_score",
            "ml_probability", "ml_label", "status", "signals",
        ])
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)
    print(f"Saved to {args.output}")

    # Summary.
    ml_bots = sum(1 for r in output_rows if r["ml_label"] == "bot")
    h_bots = sum(1 for r in output_rows if int(r["heuristic_score"]) >= 4)
    ml_only = sum(1 for r in output_rows
                  if r["ml_label"] == "bot" and int(r["heuristic_score"]) < 4)
    h_only = sum(1 for r in output_rows
                 if r["ml_label"] != "bot" and int(r["heuristic_score"]) >= 4)

    print(f"\n{'=' * 60}")
    print(f"Total users:              {len(output_rows)}")
    print(f"Heuristic bots (>= 4):    {h_bots}")
    print(f"ML bots (p >= 0.5):       {ml_bots}")
    print(f"ML-only catches (h < 4):  {ml_only}  <-- new detections")
    print(f"Heuristic-only (ML safe): {h_only}")
    print(f"Both agree bot:           {ml_bots - ml_only}")
    print(f"{'=' * 60}")

    # Show the ML-only catches — these are the interesting ones.
    ml_only_rows = [r for r in output_rows
                    if r["ml_label"] == "bot" and int(r["heuristic_score"]) < 4]
    if ml_only_rows:
        print(f"\nML-only bot detections (heuristic score < 4):")
        print(f"{'Score':>5} {'ML Prob':>7} {'Name':30} {'Status':15} {'Signals'}")
        print("-" * 90)
        for r in ml_only_rows[:30]:
            print(f"{r['heuristic_score']:>5} {r['ml_probability']:>7} "
                  f"{r['name'][:30]:30} {r['status'][:15]:15} {r['signals'][:40]}")
        if len(ml_only_rows) > 30:
            print(f"  ... and {len(ml_only_rows) - 30} more")


if __name__ == "__main__":
    main()
