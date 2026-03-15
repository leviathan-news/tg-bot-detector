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

    def test_label_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "label", "--channel", "@test"]):
            with patch("tg_purge.commands.label.run", mock_run):
                main()
        mock_run.assert_called_once()

    def test_ml_dispatches(self):
        mock_run = AsyncMock()
        with patch("sys.argv", ["tg-purge", "ml", "info"]):
            with patch("tg_purge.commands.ml_cmd.run", mock_run):
                main()
        mock_run.assert_called_once()


class TestLabelCommand:
    """Test label subcommand argument parsing."""

    def test_label_requires_channel(self):
        parser = build_parser()
        args = parser.parse_args(["label", "--channel", "@test"])
        assert args.command == "label"
        assert args.channel == "@test"

    def test_label_bootstrap_flag(self):
        parser = build_parser()
        args = parser.parse_args(["label", "--channel", "@test", "--bootstrap"])
        assert args.bootstrap is True

    def test_label_default_no_bootstrap(self):
        parser = build_parser()
        args = parser.parse_args(["label", "--channel", "@test"])
        assert args.bootstrap is False

    def test_label_default_strategy(self):
        parser = build_parser()
        args = parser.parse_args(["label", "--channel", "@test"])
        assert args.strategy == "full"

    def test_label_minimal_strategy(self):
        parser = build_parser()
        args = parser.parse_args(["label", "--channel", "@test", "--strategy", "minimal"])
        assert args.strategy == "minimal"

    def test_label_accepts_common_args(self):
        """label subcommand should accept --delay, --session-path, --config."""
        parser = build_parser()
        args = parser.parse_args([
            "label", "--channel", "@test",
            "--delay", "2.5",
            "--session-path", "/tmp/sess",
            "--config", "/tmp/config.toml",
        ])
        assert args.delay == 2.5
        assert args.session_path == "/tmp/sess"
        assert args.config == "/tmp/config.toml"


class TestMLCommand:
    """Test ml subcommand and sub-action argument parsing."""

    def test_ml_train(self):
        parser = build_parser()
        args = parser.parse_args(["ml", "train", "--channel", "@test"])
        assert args.command == "ml"
        assert args.ml_action == "train"

    def test_ml_info(self):
        parser = build_parser()
        args = parser.parse_args(["ml", "info"])
        assert args.command == "ml"
        assert args.ml_action == "info"

    def test_ml_export_features(self):
        parser = build_parser()
        args = parser.parse_args([
            "ml", "export-features",
            "--channel", "@test",
            "--output", "out.json",
        ])
        assert args.ml_action == "export-features"
        assert args.output == "out.json"

    def test_ml_train_labels_path(self):
        parser = build_parser()
        args = parser.parse_args([
            "ml", "train",
            "--channel", "@test",
            "--labels-path", "datasets/test/labels.json",
        ])
        assert args.labels_path == "datasets/test/labels.json"

    def test_ml_train_output_dir_default(self):
        parser = build_parser()
        args = parser.parse_args(["ml", "train", "--channel", "@test"])
        assert args.output_dir == "models"

    def test_ml_train_output_dir_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "ml", "train", "--channel", "@test", "--output-dir", "my_models",
        ])
        assert args.output_dir == "my_models"

    def test_ml_info_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["ml", "info", "--model-path", "models/test.json"])
        assert args.model_path == "models/test.json"

    def test_ml_info_model_path_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["ml", "info"])
        assert args.model_path is None

    def test_ml_export_features_strategy(self):
        parser = build_parser()
        args = parser.parse_args([
            "ml", "export-features",
            "--channel", "@test",
            "--output", "out.json",
            "--strategy", "minimal",
        ])
        assert args.strategy == "minimal"

    def test_ml_train_accepts_common_args(self):
        """ml train should accept --delay, --session-path, --config via _add_common_args."""
        parser = build_parser()
        args = parser.parse_args([
            "ml", "train", "--channel", "@test", "--delay", "1.0",
        ])
        assert args.delay == 1.0

    def test_ml_export_features_requires_output(self):
        """ml export-features --output is required."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ml", "export-features", "--channel", "@test"])


class TestScoringFlag:
    """Test --scoring and --stats flags added to common args."""

    def test_scoring_default_heuristic(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test"])
        assert args.scoring == "heuristic"

    def test_scoring_ml(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--scoring", "ml"])
        assert args.scoring == "ml"

    def test_scoring_hybrid(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--scoring", "hybrid"])
        assert args.scoring == "hybrid"

    def test_scoring_invalid_choice_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze", "--channel", "@test", "--scoring", "invalid"])

    def test_stats_flag(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test", "--stats"])
        assert args.stats is True

    def test_stats_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--channel", "@test"])
        assert args.stats is False

    def test_scoring_flag_on_candidates(self):
        """--scoring should be accepted by candidates subcommand too."""
        parser = build_parser()
        args = parser.parse_args(["candidates", "--channel", "@test", "--scoring", "hybrid"])
        assert args.scoring == "hybrid"

    def test_stats_flag_on_join_dates(self):
        """--stats should be accepted by join-dates subcommand."""
        parser = build_parser()
        args = parser.parse_args(["join-dates", "--channel", "@test", "--stats"])
        assert args.stats is True


class TestPartialSave:
    """Test that Ctrl+C partial save works with and without --output."""

    def _make_mock_users(self, count=5):
        """Create a dict of mock users for scoring."""
        from tests.conftest import MockUser
        users = {}
        for i in range(1, count + 1):
            users[i] = MockUser(
                id=i, first_name=f"User{i}", username=f"user{i}",
                photo=False, status=None,
            )
        return users

    def test_partial_save_without_output_flag(self, tmp_path, monkeypatch):
        """When interrupted=True and no --output, a partial CSV is auto-generated."""
        from tg_purge.commands.candidates import _score_and_export

        # Redirect auto-generated output into tmp_path
        monkeypatch.chdir(tmp_path)

        users = self._make_mock_users(10)
        _score_and_export(
            all_users=users,
            join_dates={},
            spike_windows=[],
            threshold=2,
            safelist=set(),
            sub_count=100,
            output_path=None,  # No --output provided
            interrupted=True,
        )

        # A partial CSV should have been auto-generated in output/
        partial_files = list((tmp_path / "output").glob("candidates-partial-*.csv"))
        assert len(partial_files) == 1, f"Expected 1 partial file, found: {partial_files}"

        # Verify all 10 users are in the file (header + 10 data rows)
        lines = partial_files[0].read_text().strip().split("\n")
        assert len(lines) == 11, f"Expected 11 lines (header + 10), got {len(lines)}"

    def test_partial_save_with_output_flag(self, tmp_path):
        """When interrupted=True and --output is provided, a -partial suffixed file is created."""
        from tg_purge.commands.candidates import _score_and_export

        output_path = str(tmp_path / "results.csv")
        users = self._make_mock_users(5)

        _score_and_export(
            all_users=users,
            join_dates={},
            spike_windows=[],
            threshold=2,
            safelist=set(),
            sub_count=50,
            output_path=output_path,
            interrupted=True,
        )

        # Should create results-partial.csv, NOT results.csv
        partial = tmp_path / "results-partial.csv"
        assert partial.exists(), "Expected results-partial.csv to be created"
        assert not (tmp_path / "results.csv").exists(), "Original path should NOT be created on interrupt"

        lines = partial.read_text().strip().split("\n")
        assert len(lines) == 6  # header + 5 users

    def test_normal_run_no_partial_without_output(self, tmp_path, monkeypatch):
        """When not interrupted and no --output, no partial file is created."""
        from tg_purge.commands.candidates import _score_and_export

        monkeypatch.chdir(tmp_path)
        users = self._make_mock_users(3)

        _score_and_export(
            all_users=users,
            join_dates={},
            spike_windows=[],
            threshold=2,
            safelist=set(),
            sub_count=50,
            output_path=None,
            interrupted=False,
        )

        # No output directory should be created
        output_dir = tmp_path / "output"
        if output_dir.exists():
            assert list(output_dir.glob("*")) == []
