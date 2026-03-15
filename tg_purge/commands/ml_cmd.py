"""
ML model management command for tg-bot-detector.

Sub-actions dispatched from cli.main() via args.ml_action:

  train
      Locate the label file for a channel (from --labels-path or derived from
      --channel), check that ML dependencies are available via ml_available(),
      load the labels, and print a stub message explaining that cached feature
      vectors are required before full training can proceed.

  info
      Find the latest model metadata JSON file in the models/ directory (or at
      --model-path), deserialise it, and print its fields to stdout.

  export-features
      Print a stub message noting that feature export is not yet implemented.
      (The infrastructure — extract_features() in features.py — exists; this
      command will be wired up in a future task.)

All network operations would require a connected Telethon client (relevant for
future sub-actions).  The current stubs do not open a Telegram connection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from ..ml import ml_available, load_model_metadata, train_model
from ..utils import channel_slug as _channel_slug


def _default_labels_path(channel: str) -> str:
    """Return the canonical label file path for a channel.

    Mirrors the convention used by label.py so that 'ml train' can locate
    labels produced by 'label --bootstrap' without the user needing to supply
    an explicit path.

    Args:
        channel: Channel identifier string.

    Returns:
        Relative path string: datasets/<slug>/labels.json
    """
    slug = _channel_slug(channel)
    return str(Path("datasets") / slug / "labels.json")


def _find_latest_model_json(models_dir: str = "models") -> Optional[str]:
    """Search models_dir for the most recently modified .json file.

    JSON files in the models directory are metadata files produced by
    train_model().  When --model-path is not specified, we pick the newest
    one so that 'ml info' is useful without arguments.

    Args:
        models_dir: Directory to search (default: "models").

    Returns:
        Absolute path to the most recently modified .json file, or None if
        no .json files exist in models_dir.
    """
    models_path = Path(models_dir)
    if not models_path.is_dir():
        return None

    json_files = list(models_path.glob("*.json"))
    if not json_files:
        return None

    # Sort by modification time, descending.  Return the newest file.
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(json_files[0].resolve())


def _run_train(args) -> None:
    """Stub: locate labels and report training prerequisites.

    In the full implementation this will:
      1. Load the label file.
      2. Enumerate subscribers and extract feature vectors (or load a cached
         feature file produced by 'ml export-features').
      3. Call train_model() with the aligned (features, labels) lists.
      4. Print the resulting metrics.

    For now the function validates dependencies and input files, then prints
    a clear message explaining the next step.

    Args:
        args: Parsed argparse namespace from build_parser().
    """
    # Check that ML dependencies are installed before doing any file I/O.
    if not ml_available():
        print(
            "ML dependencies are not installed.\n"
            "Install scikit-learn (and optionally lightgbm / xgboost):\n\n"
            "  pip install scikit-learn",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve the labels file path: --labels-path takes priority; otherwise
    # derive from --channel using the same convention as 'label --bootstrap'.
    labels_path = getattr(args, "labels_path", None)
    if not labels_path:
        channel = getattr(args, "channel", None)
        if not channel:
            print(
                "Error: supply either --labels-path or --channel so the label "
                "file can be located.",
                file=sys.stderr,
            )
            sys.exit(1)
        labels_path = _default_labels_path(channel)

    labels_file = Path(labels_path)
    if not labels_file.exists():
        print(
            f"Label file not found: {labels_path}\n"
            "Run 'tg-purge label --channel <channel> --bootstrap' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load labels.
    with open(labels_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    labels_data = raw.get("labels", {})
    channel_name = raw.get("channel", getattr(args, "channel", None))

    # Look for cached feature vectors (produced by label --bootstrap).
    features_path = str(labels_file.parent / "features.json")
    if not Path(features_path).exists():
        print(
            f"Feature vectors not found: {features_path}\n"
            "Re-run 'tg-purge label --channel <channel> --bootstrap' to generate them.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(features_path, "r", encoding="utf-8") as fh:
        features_raw = json.load(fh)
    feature_cache = features_raw.get("features", {})

    # Align features and labels — only include users with both a feature vector
    # and a bot/human label (skip "unlabeled").
    train_features = []
    train_labels = []
    for uid_str, label_info in labels_data.items():
        label = label_info.get("label", "unlabeled")
        if label not in ("bot", "human"):
            continue
        if uid_str not in feature_cache:
            continue
        train_features.append(feature_cache[uid_str])
        train_labels.append(label)

    n_bot = sum(1 for l in train_labels if l == "bot")
    n_human = sum(1 for l in train_labels if l == "human")
    print(
        f"Labels file:   {labels_path} ({len(labels_data)} entries)\n"
        f"Features file: {features_path} ({len(feature_cache)} vectors)\n"
        f"Training set:  {len(train_features)} samples ({n_bot} bot, {n_human} human)",
        file=sys.stderr,
    )

    if len(train_features) < 10:
        print("Error: fewer than 10 labeled samples. Need more data.", file=sys.stderr)
        sys.exit(1)
    if n_bot == 0 or n_human == 0:
        print("Error: both bot and human labels are required.", file=sys.stderr)
        sys.exit(1)

    # Train the model.
    output_dir = getattr(args, "output_dir", "models")
    print(f"\nTraining...", file=sys.stderr)
    result = train_model(
        train_features,
        train_labels,
        output_dir=output_dir,
        channel=channel_name,
    )

    if not result.get("success"):
        print(f"Training failed: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    # Print results.
    metrics = result["metrics"]
    print(
        f"\nTraining complete!\n"
        f"  Algorithm:  {result['algorithm']}\n"
        f"  F1:         {metrics['f1']:.3f}\n"
        f"  Precision:  {metrics['precision']:.3f}\n"
        f"  Recall:     {metrics['recall']:.3f}\n"
        f"  AUC-ROC:    {metrics['auc_roc']:.3f}\n"
        f"\n"
        f"  Model:      {result['model_file']}\n"
        f"  Metadata:   {result['metadata_file']}",
        file=sys.stderr,
    )


def _run_info(args) -> None:
    """Print metadata for a saved model JSON file.

    When --model-path is not provided, the latest .json in models/ is used.
    All metadata fields are printed to stdout as formatted key-value pairs.

    Args:
        args: Parsed argparse namespace from build_parser().
    """
    model_path = getattr(args, "model_path", None)

    if not model_path:
        # Auto-discover the newest metadata file in the default models/ dir.
        model_path = _find_latest_model_json("models")

    if not model_path:
        print(
            "No model metadata file found.\n"
            "Train a model first with 'tg-purge ml train', or specify "
            "--model-path <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    meta_path = Path(model_path)
    if not meta_path.exists():
        print(f"Model metadata file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # load_model_metadata raises FileNotFoundError / JSONDecodeError on bad
    # files — let them propagate as unhandled exceptions for a clear traceback.
    metadata = load_model_metadata(str(meta_path))

    # Pretty-print metadata fields to stdout.
    print(f"Model metadata: {meta_path}")
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Nested dicts (e.g. "metrics") printed as a sub-block.
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        elif isinstance(value, list):
            # Lists (e.g. "feature_names") joined for readability.
            print(f"  {key}: [{', '.join(str(v) for v in value)}]")
        else:
            print(f"  {key}: {value}")


def _run_export_features(args) -> None:
    """Stub: feature vector export is not yet implemented.

    The underlying extract_features() function in features.py is complete;
    this command will enumerate subscribers, call extract_features() for each
    one, and write the resulting list to the --output JSON file.

    Args:
        args: Parsed argparse namespace from build_parser().
    """
    output = getattr(args, "output", None)
    print(
        f"export-features to {output!r}: not yet implemented.\n"
        "Feature extraction infrastructure exists in tg_purge/features.py;\n"
        "this sub-command will be wired up in a future task.",
        file=sys.stderr,
    )


async def run(args) -> None:
    """Entry point dispatched from cli.main() for the 'ml' subcommand.

    Dispatches to the appropriate sub-action handler based on args.ml_action.
    All sub-actions are synchronous stubs at this stage, so the coroutine
    wrapper is thin — it simply awaits nothing and calls the sync helper.

    Args:
        args: Parsed argparse namespace from build_parser().
    """
    action = getattr(args, "ml_action", None)

    if action == "train":
        _run_train(args)
    elif action == "info":
        _run_info(args)
    elif action == "export-features":
        _run_export_features(args)
    else:
        # No sub-action specified — print usage hint.
        print(
            "Usage: tg-purge ml <action>\n"
            "Available actions: train, info, export-features\n\n"
            "Run 'tg-purge ml --help' for details.",
            file=sys.stderr,
        )
        sys.exit(1)
