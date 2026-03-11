"""
Label management command for tg-bot-detector.

Two modes of operation:

  --bootstrap
      Enumerates channel subscribers using the standard work-queue enumeration,
      scores every user with the heuristic scorer, derives weak labels via
      bootstrap_labels(), persists them to datasets/<channel_slug>/labels.json,
      and prints aggregate statistics to stderr.  This is the recommended entry
      point before any ML training.

  (default, no --bootstrap)
      Loads an existing labels file for the channel (if present) and prints
      label statistics.  If the file does not exist, prints a helpful message
      directing the user to run with --bootstrap.

Security
--------
  Label files contain real Telegram user IDs (PII).  save_labels() sets
  directory permissions to 700 and file permissions to 600 (best-effort).

  The channel slug is derived by stripping '@' and replacing non-alphanumeric
  characters with '_', so it is safe to use as a filesystem path component.
"""

import json
import os
import stat
import sys
from pathlib import Path

from ..config import load_config
from ..client import create_client, resolve_channel
from ..clustering import detect_spike_windows
from ..enumeration import enumerate_subscribers
from ..features import extract_features
from ..scoring import score_user
from ..labeling import bootstrap_labels, save_labels, load_labels, label_stats
from ..utils import channel_slug as _channel_slug


def _labels_path(channel: str) -> str:
    """Return the canonical path for a channel's labels JSON file.

    Uses a predictable layout:  datasets/<slug>/labels.json

    Args:
        channel: Channel identifier string.

    Returns:
        Relative path string (resolved to cwd at runtime).
    """
    slug = _channel_slug(channel)
    return str(Path("datasets") / slug / "labels.json")


async def _run_bootstrap(args, config, channel_name: str) -> None:
    """Enumerate, score, and persist weak heuristic labels for a channel.

    Connects to Telegram via the Telethon client, enumerates subscribers using
    the configured search strategy, scores each user, derives labels, saves the
    label file, and prints statistics to stderr.

    Args:
        args:         Parsed argparse namespace.
        config:       Loaded Config object.
        channel_name: Resolved channel identifier string.
    """
    client = await create_client(config)
    try:
        channel = await resolve_channel(client, channel_name)

        # Show a lightweight progress indicator while queries run.
        def progress(i, total, found):
            if i % 10 == 0:
                print(
                    f"  ...{i}/{total} queries, {found} users found",
                    flush=True,
                    file=sys.stderr,
                )

        print(f"\nBootstrapping labels for {channel_name!r}...", file=sys.stderr)
        result = await enumerate_subscribers(
            client,
            channel,
            strategy=getattr(args, "strategy", "full"),
            delay=config.delay,
            progress_callback=progress,
        )
    finally:
        # Always disconnect even if enumeration raises.
        await client.disconnect()

    all_users = result["users"]
    join_dates = result["join_dates"]
    print(f"Enumerated {len(all_users)} subscribers.", file=sys.stderr)

    # Auto-detect spike windows from join dates for spike_join scoring.
    spike_windows = []
    if len(join_dates) >= 10:
        spike_windows = detect_spike_windows(join_dates)
        if spike_windows:
            print(
                f"Auto-detected {len(spike_windows)} spike window(s).",
                file=sys.stderr,
            )

    # Score every user with the default heuristic config.
    scored = {}
    for uid, user in all_users.items():
        score, _reasons = score_user(
            user,
            join_date=join_dates.get(uid),
            spike_windows=spike_windows,
        )
        scored[uid] = (score, _reasons)

    # Extract feature vectors for every user (cached to disk for ml train).
    print("Extracting feature vectors...", file=sys.stderr)
    feature_cache = {}  # uid (str) -> feature dict
    for uid, user in all_users.items():
        features = extract_features(
            user,
            join_date=join_dates.get(uid),
            spike_windows=spike_windows,
        )
        feature_cache[str(uid)] = features

    # Derive weak labels from heuristic scores.
    labels = bootstrap_labels(all_users, scored)

    # Persist labels to disk.
    path = _labels_path(channel_name)
    save_labels(labels, channel_name, path)
    print(f"Labels saved to {path}", file=sys.stderr)

    # Persist feature vectors alongside labels (same directory).
    # This file is loaded by 'ml train' to avoid re-enumerating.
    features_path = str(Path(path).parent / "features.json")
    features_dir = Path(features_path).parent
    features_dir.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"channel": channel_name, "features": feature_cache},
            fh,
            indent=2,
        )
    # Restrict permissions — feature vectors contain heuristic scores tied to user IDs.
    try:
        os.chmod(features_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    print(f"Feature vectors saved to {features_path}", file=sys.stderr)

    # Print aggregate statistics.
    stats = label_stats(labels)
    print(
        f"\nLabel statistics:\n"
        f"  Total:      {stats['total']}\n"
        f"  Bot:        {stats['bot']}\n"
        f"  Human:      {stats['human']}\n"
        f"  Unlabeled:  {stats['unlabeled']}\n"
        f"  Human-reviewed: {stats['human_labeled']}",
        file=sys.stderr,
    )


def _run_inspect(channel_name: str) -> None:
    """Load and display existing label statistics for a channel.

    If no label file exists yet, directs the user to run with --bootstrap.

    Args:
        channel_name: Resolved channel identifier string.
    """
    path = _labels_path(channel_name)
    data = load_labels(path)

    if not data["labels"]:
        # Either file doesn't exist or it is empty.
        print(
            f"No labels found for {channel_name!r}.\n"
            f"Run with --bootstrap to generate weak labels from heuristic scores:\n\n"
            f"  tg-purge label --channel {channel_name} --bootstrap",
            file=sys.stderr,
        )
        return

    stats = label_stats(data["labels"])
    print(
        f"Labels for {data['channel']!r} (version {data['version']}):\n"
        f"  Total:          {stats['total']}\n"
        f"  Bot:            {stats['bot']}\n"
        f"  Human:          {stats['human']}\n"
        f"  Unlabeled:      {stats['unlabeled']}\n"
        f"  Human-reviewed: {stats['human_labeled']}\n\n"
        "Interactive labeling not yet implemented. "
        "Use --bootstrap to (re-)generate heuristic labels.",
        file=sys.stderr,
    )


async def run(args) -> None:
    """Entry point dispatched from cli.main() for the 'label' subcommand.

    Loads configuration, resolves the target channel name, then delegates to
    either _run_bootstrap (when --bootstrap is set) or _run_inspect.

    Args:
        args: Parsed argparse namespace from build_parser().
    """
    config = load_config(getattr(args, "config", None))

    # Apply CLI overrides.
    if getattr(args, "session_path", None):
        config.session_path = args.session_path
    if getattr(args, "delay", None) is not None:
        config.delay = args.delay

    channel_name = config.resolve_channel(args.channel)

    if getattr(args, "bootstrap", False):
        await _run_bootstrap(args, config, channel_name)
    else:
        _run_inspect(channel_name)
