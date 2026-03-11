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

import json
import os
import sys
from pathlib import Path

from ..ml import ml_available, load_model_metadata


def _channel_slug(channel: str) -> str:
    """Derive a filesystem-safe slug from a channel identifier.

    Strips a leading '@' and replaces every non-alphanumeric, non-underscore
    character with '_'.  Truncates to 64 characters.

    Args:
        channel: Channel identifier string, e.g. "@leviathan_news".

    Returns:
        Slug string safe to use as a directory name component.
    """
    slug = channel.lstrip("@")
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
    return slug[:64]


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


def _find_latest_model_json(models_dir: str = "models") -> str | None:
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

    # Load labels to count them and show the user what is available.
    with open(labels_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    n_labels = len(raw.get("labels", {}))

    output_dir = getattr(args, "output_dir", "models")

    print(
        f"Labels file:  {labels_path} ({n_labels} entries)\n"
        f"Output dir:   {output_dir}\n\n"
        "Training requires cached feature vectors (not yet implemented).\n"
        "Use 'tg-purge ml export-features --channel <channel> --output <path>'\n"
        "to generate feature vectors, then re-run 'ml train'.",
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
