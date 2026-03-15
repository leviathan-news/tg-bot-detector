"""Predict labels for remaining unlabeled users and find edge cases.

Active learning helper: runs the trained ML model on all unlabeled users,
reports prediction distribution, and lists the most uncertain predictions
(probability closest to 0.5) for human review prioritization.

Usage:
    python scripts/predict_unlabeled.py --channel leviathan_news
    python scripts/predict_unlabeled.py --channel leviathan_news --top 50 --threshold 0.4 0.6
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tg_purge.ml import predict


def main():
    parser = argparse.ArgumentParser(
        description="Predict labels for unlabeled users and surface edge cases."
    )
    parser.add_argument(
        "--channel", required=True,
        help="Channel name (matches datasets/<channel>/ directory).",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of most uncertain predictions to display (default: 30).",
    )
    parser.add_argument(
        "--threshold", type=float, nargs=2, default=[0.3, 0.7],
        metavar=("LOW", "HIGH"),
        help="Probability range for edge cases (default: 0.3 0.7).",
    )
    args = parser.parse_args()

    # Load labels and features
    base = Path("datasets") / args.channel
    labels_path = base / "labels.json"
    features_path = base / "features.json"
    model_path = Path("models") / f"{args.channel}_sklearn_rf.joblib"

    if not labels_path.exists():
        print("Labels file not found: %s" % labels_path, file=sys.stderr)
        sys.exit(1)
    if not features_path.exists():
        print("Features file not found: %s" % features_path, file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print("Model not found: %s" % model_path, file=sys.stderr)
        sys.exit(1)

    with open(labels_path) as f:
        raw = json.load(f)
    with open(features_path) as f:
        features = json.load(f)["features"]

    # Collect unlabeled users that have feature vectors
    unlabeled = []
    for uid_str, info in raw["labels"].items():
        if info["label"] == "unlabeled":
            feat = features.get(uid_str, {})
            if feat:
                unlabeled.append({"uid": uid_str, "features": feat})

    print("Unlabeled users: %d" % len(unlabeled), file=sys.stderr)

    if not unlabeled:
        print("No unlabeled users with features found.", file=sys.stderr)
        sys.exit(0)

    # Run predictions
    feature_dicts = [u["features"] for u in unlabeled]
    results = predict(feature_dicts, str(model_path))

    # Analyze prediction distribution and find edge cases
    pred_counts = Counter()
    edge_cases = []
    low, high = args.threshold

    for u, r in zip(unlabeled, results):
        pred_counts[r["label"]] += 1

        # Edge cases: model is uncertain (probability within threshold range)
        if low <= r["probability"] <= high:
            edge_cases.append({
                "uid": u["uid"],
                "prob": round(r["probability"], 3),
                "pred": r["label"],
                "h_score": int(u["features"].get("heuristic_score", 0)),
            })

    # Output summary
    print("ML predictions on unlabeled:")
    print("  Bot:   %d" % pred_counts.get("bot", 0))
    print("  Human: %d" % pred_counts.get("human", 0))
    print("")
    print(
        "Edge cases (probability %.1f-%.1f, model uncertain): %d"
        % (low, high, len(edge_cases))
    )

    # Sort by uncertainty (closest to 0.5 first) and display top N
    edge_cases.sort(key=lambda x: abs(x["prob"] - 0.5))
    for ec in edge_cases[: args.top]:
        print(
            "  UID %s: prob=%.3f pred=%s h_score=%d"
            % (ec["uid"], ec["prob"], ec["pred"], ec["h_score"])
        )


if __name__ == "__main__":
    main()
