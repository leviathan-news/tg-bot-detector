"""Tests for CLI argument routing and subcommand dispatch."""

import pytest
from unittest.mock import patch, AsyncMock

from tg_purge.cli import build_parser, main


class TestBuildParser:
    """Test argparse configuration."""

    def test_no_args_shows_help(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_analyze_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test"])
        assert args.command == "analyze"
        assert args.channel == "@test"

    def test_analyze_strategy(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--strategy", "minimal"])
        assert args.strategy == "minimal"

    def test_analyze_default_strategy(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test"])
        assert args.strategy == "full"

    def test_join_dates_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["join-dates", "--channel", "@test"])
        assert args.command == "join-dates"
        assert args.channel == "@test"

    def test_join_dates_top_days(self):
        parser = build_parser()
        args = parser.parse_args(["join-dates", "--channel", "@test", "--top-days", "50"])
        assert args.top_days == 50

    def test_spike_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "spike", "--channel", "@test",
            "--start", "2025-11-09T06:00Z",
            "--end", "2025-11-09T07:00Z",
        ])
        assert args.command == "spike"
        assert args.start == "2025-11-09T06:00Z"
        assert args.end == "2025-11-09T07:00Z"

    def test_spike_requires_start_end(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["spike", "--channel", "@test"])

    def test_validate_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "--channel", "@test", "--known-users", "users.csv"])
        assert args.command == "validate"
        assert args.known_users == "users.csv"

    def test_validate_requires_known_users(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["validate", "--channel", "@test"])

    def test_candidates_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "candidates", "--channel", "@test",
            "--threshold", "4",
            "--output", "out.csv",
            "--safelist", "safe.csv",
        ])
        assert args.command == "candidates"
        assert args.threshold == 4
        assert args.output == "out.csv"
        assert args.safelist == "safe.csv"

    def test_candidates_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@test"])
        assert args.threshold is None
        assert args.output is None
        assert args.safelist is None

    def test_registry_generate_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "registry", "generate",
            "--channel", "@test",
            "--threshold", "4",
        ])
        assert args.command == "registry"
        assert args.registry_action == "generate"
        assert args.threshold == 4

    def test_registry_add_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "registry", "add", "--ids-file", "ids.txt",
        ])
        assert args.command == "registry"
        assert args.registry_action == "add"
        assert args.ids_file == "ids.txt"

    def test_registry_check_subcommand(self):
        parser = build_parser()
        args = parser.parse_args([
            "registry", "check", "--user-id", "123456789",
        ])
        assert args.command == "registry"
        assert args.registry_action == "check"
        assert args.user_id == "123456789"

    def test_session_path_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "analyze", "--channel", "@test", "--session-path", "/custom/session",
        ])
        assert args.session_path == "/custom/session"

    def test_config_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "analyze", "--channel", "@test", "--config", "/custom/config.toml",
        ])
        assert args.config == "/custom/config.toml"

    def test_delay_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--delay", "3.0"])
        assert args.delay == 3.0

    def test_delay_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test"])
        assert args.delay is None

    def test_spike_strategy_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "spike", "--channel", "@test",
            "--start", "2025-01-01", "--end", "2025-01-02",
            "--strategy", "minimal",
        ])
        assert args.strategy == "minimal"

    def test_spike_default_strategy(self):
        parser = build_parser()
        args = parser.parse_args([
            "spike", "--channel", "@test",
            "--start", "2025-01-01", "--end", "2025-01-02",
        ])
        assert args.strategy == "full"

    def test_delay_on_all_subcommands(self):
        """--delay should be accepted by all subcommands that use _add_common_args."""
        parser = build_parser()
        for cmd_args in [
            ["analyze", "--channel", "@t", "--delay", "2.0"],
            ["join-dates", "--channel", "@t", "--delay", "2.0"],
            ["spike", "--channel", "@t", "--start", "2025-01-01", "--end", "2025-01-02", "--delay", "2.0"],
            ["validate", "--channel", "@t", "--known-users", "u.csv", "--delay", "2.0"],
            ["candidates", "--channel", "@t", "--delay", "2.0"],
            ["registry", "generate", "--channel", "@t", "--delay", "2.0"],
        ]:
            args = parser.parse_args(cmd_args)
            assert args.delay == 2.0, f"--delay not accepted for {cmd_args[0]}"

    def test_candidates_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_candidates_default_auto_cluster(self):
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@t"])
        assert args.no_auto_cluster is False

    def test_analyze_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_registry_generate_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["registry", "generate", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_join_dates_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args(["join-dates", "--channel", "@t", "--no-auto-cluster"])
        assert args.no_auto_cluster is True

    def test_spike_no_auto_cluster_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "spike", "--channel", "@t",
            "--start", "2025-01-01", "--end", "2025-01-02",
            "--no-auto-cluster",
        ])
        assert args.no_auto_cluster is True

    def test_help_output(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


class TestMainDispatch:
    """Test that main() dispatches to correct command handlers."""

    def test_no_command_exits(self):
        with patch("sys.argv", ["tg-purge"]):
            with pytest.raises(SystemExit):
                main()

    def test_analyze_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "analyze", "--channel", "@test"]):
            with patch("tg_purge.commands.analyze.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_join_dates_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "join-dates", "--channel", "@test"]):
            with patch("tg_purge.commands.join_dates.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_spike_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "spike", "--channel", "@test",
                    "--start", "2025-01-01", "--end", "2025-01-02"]):
            with patch("tg_purge.commands.spike.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_validate_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "validate", "--channel", "@test",
                    "--known-users", "users.csv"]):
            with patch("tg_purge.commands.validate.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_candidates_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "candidates", "--channel", "@test"]):
            with patch("tg_purge.commands.candidates.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_registry_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "registry", "generate", "--channel", "@test"]):
            with patch("tg_purge.commands.registry.run", mock_run):
                main()
        mock_run.assert_called_once()
