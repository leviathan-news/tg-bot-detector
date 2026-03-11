"""
CLI entry point and subcommand router for tg-purge.

Usage:
    tg-purge analyze [--channel @foo] [--strategy full|minimal]
    tg-purge join-dates [--channel @foo] [--top-days 30]
    tg-purge spike --start "2025-11-09T06:00Z" --end "2025-11-09T07:00Z"
    tg-purge validate --known-users users.csv [--channel @foo]
    tg-purge candidates [--channel @foo] [--threshold 4] [--output candidates.csv]
    tg-purge registry generate [--channel @foo] [--threshold 4]
    tg-purge registry add --ids-file flagged.txt
    tg-purge registry check --user-id 123456789
"""

import argparse
import asyncio
import sys


def _add_common_args(parser):
    """Add arguments shared across all subcommands."""
    parser.add_argument(
        "--channel",
        help="Target channel username (e.g., @foo) or numeric ID. "
             "Required unless default_channel is set in config.toml.",
    )
    parser.add_argument(
        "--session-path",
        dest="session_path",
        default=None,
        help="Override session file path (default: ~/.tg_purge/session).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config file (default: config.toml in current dir).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between API queries (overrides config, default: 1.5).",
    )
    parser.add_argument(
        "--scoring",
        choices=["heuristic", "ml", "hybrid"],
        default="heuristic",
        help="Scoring mode: 'heuristic' (default), 'ml' (requires model), "
             "'hybrid' (show both).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="Show statistical summary (confidence intervals, bias estimate).",
    )


def build_parser():
    """Build the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="tg-purge",
        description="Heuristic bot detection toolkit for Telegram broadcast channel subscribers.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── analyze ───────────────────────────────────────────────
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Multi-round subscriber analysis with scoring.",
    )
    _add_common_args(p_analyze)
    p_analyze.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy: 'full' (69 queries) or 'minimal' (22 queries). Default: full.",
    )
    p_analyze.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )

    # ── join-dates ────────────────────────────────────────────
    p_join = subparsers.add_parser(
        "join-dates",
        help="Join date clustering and spike detection.",
    )
    _add_common_args(p_join)
    p_join.add_argument(
        "--top-days",
        dest="top_days",
        type=int,
        default=30,
        help="Number of top spike days to display (default: 30).",
    )
    p_join.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy: 'full' or 'minimal'. Default: full.",
    )
    p_join.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )

    # ── spike ─────────────────────────────────────────────────
    p_spike = subparsers.add_parser(
        "spike",
        help="Deep-dive into a specific time window.",
    )
    _add_common_args(p_spike)
    p_spike.add_argument(
        "--start",
        required=True,
        help="Spike window start (ISO 8601, e.g., '2025-11-09T06:00Z').",
    )
    p_spike.add_argument(
        "--end",
        required=True,
        help="Spike window end (ISO 8601, e.g., '2025-11-09T07:00Z').",
    )
    p_spike.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy: 'full' or 'minimal'. Default: full.",
    )
    p_spike.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of additional spike clusters.",
    )

    # ── validate ──────────────────────────────────────────────
    p_validate = subparsers.add_parser(
        "validate",
        help="Score known-good users to measure false positive rate.",
    )
    _add_common_args(p_validate)
    p_validate.add_argument(
        "--known-users",
        dest="known_users",
        required=True,
        help="CSV or JSON file of known-good user IDs/usernames.",
    )

    # ── candidates ────────────────────────────────────────────
    p_candidates = subparsers.add_parser(
        "candidates",
        help="Generate scored candidate lists for offline review.",
    )
    _add_common_args(p_candidates)
    p_candidates.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Minimum score to include in candidates (default: from config, usually 2).",
    )
    p_candidates.add_argument(
        "--output",
        default=None,
        help="Output file path (CSV or JSON based on extension). Prints to stdout if omitted.",
    )
    p_candidates.add_argument(
        "--safelist",
        default=None,
        help="CSV/JSON/text file of user IDs to exclude from candidates.",
    )
    p_candidates.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy: 'full' or 'minimal'. Default: full.",
    )
    p_candidates.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )

    # ── registry ──────────────────────────────────────────────
    p_registry = subparsers.add_parser(
        "registry",
        help="Local bot registry management.",
    )
    reg_sub = p_registry.add_subparsers(dest="registry_action", help="Registry actions")

    # registry generate
    p_reg_gen = reg_sub.add_parser("generate", help="Generate registry from enumeration + scoring.")
    _add_common_args(p_reg_gen)
    p_reg_gen.add_argument("--threshold", type=int, default=None, help="Score threshold.")
    p_reg_gen.add_argument("--output", default=None, help="Output registry JSON path.")
    p_reg_gen.add_argument(
        "--no-auto-cluster",
        dest="no_auto_cluster",
        action="store_true",
        default=False,
        help="Disable auto-detection of join-date spike clusters.",
    )

    # registry add
    p_reg_add = reg_sub.add_parser("add", help="Add IDs from a file to the registry.")
    p_reg_add.add_argument("--ids-file", dest="ids_file", required=True, help="File of user IDs (one per line).")
    p_reg_add.add_argument("--registry-path", dest="registry_path", default=None, help="Registry file path.")
    p_reg_add.add_argument("--session-path", dest="session_path", default=None)
    p_reg_add.add_argument("--channel", default=None)
    p_reg_add.add_argument("--config", default=None)

    # registry check
    p_reg_check = reg_sub.add_parser("check", help="Look up a user ID in the registry.")
    p_reg_check.add_argument("--user-id", dest="user_id", required=True, help="User ID to look up.")
    p_reg_check.add_argument("--registry-path", dest="registry_path", default=None, help="Registry file path.")
    p_reg_check.add_argument("--session-path", dest="session_path", default=None)
    p_reg_check.add_argument("--channel", default=None)
    p_reg_check.add_argument("--config", default=None)

    # ── label ─────────────────────────────────────────────────
    p_label = subparsers.add_parser(
        "label",
        help="Manage ML training labels for subscribers.",
    )
    _add_common_args(p_label)
    p_label.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Enumerate channel, score users, and write weak heuristic labels to disk.",
    )
    p_label.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy when bootstrapping: 'full' or 'minimal'. Default: full.",
    )

    # ── ml ────────────────────────────────────────────────────
    p_ml = subparsers.add_parser(
        "ml",
        help="Machine-learning model management: train, inspect, or export features.",
    )
    ml_sub = p_ml.add_subparsers(dest="ml_action", help="ML sub-actions")

    # ml train
    p_ml_train = ml_sub.add_parser(
        "train",
        help="Train a bot-detection model from labelled data.",
    )
    _add_common_args(p_ml_train)
    p_ml_train.add_argument(
        "--labels-path",
        dest="labels_path",
        default=None,
        help="Path to labels JSON file (default: derived from --channel).",
    )
    p_ml_train.add_argument(
        "--output-dir",
        dest="output_dir",
        default="models",
        help="Directory to write trained model and metadata (default: models).",
    )

    # ml info
    p_ml_info = ml_sub.add_parser(
        "info",
        help="Print metadata for a saved model.",
    )
    p_ml_info.add_argument(
        "--model-path",
        dest="model_path",
        default=None,
        help="Path to a model metadata JSON file. "
             "When omitted, the latest .json in models/ is used.",
    )

    # ml export-features
    p_ml_export = ml_sub.add_parser(
        "export-features",
        help="Export feature vectors for subscribers to a JSON file.",
    )
    _add_common_args(p_ml_export)
    p_ml_export.add_argument(
        "--output",
        required=True,
        help="Output path for the feature vector JSON file.",
    )
    p_ml_export.add_argument(
        "--strategy",
        choices=["full", "minimal"],
        default="full",
        help="Search strategy: 'full' or 'minimal'. Default: full.",
    )

    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Import and run the appropriate command
    if args.command == "analyze":
        from .commands.analyze import run
        asyncio.run(run(args))
    elif args.command == "join-dates":
        from .commands.join_dates import run
        asyncio.run(run(args))
    elif args.command == "spike":
        from .commands.spike import run
        asyncio.run(run(args))
    elif args.command == "validate":
        from .commands.validate import run
        asyncio.run(run(args))
    elif args.command == "candidates":
        from .commands.candidates import run
        asyncio.run(run(args))
    elif args.command == "registry":
        from .commands.registry import run
        asyncio.run(run(args))
    elif args.command == "label":
        from .commands.label import run
        asyncio.run(run(args))
    elif args.command == "ml":
        from .commands.ml_cmd import run
        asyncio.run(run(args))
    else:
        parser.print_help()
        sys.exit(1)
