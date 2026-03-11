"""Tests for labeling.py — ML label management data layer.

Covers:
  - bootstrap_labels: heuristic score thresholds (bot/human/unlabeled)
  - save_labels / load_labels: JSON roundtrip, key type conversion, missing file, permissions
  - label_stats: correct counts of each category and human-labeled entries
"""

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from tg_purge.labeling import bootstrap_labels, load_labels, save_labels, label_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored(user_id, score):
    """Return a minimal (score, reasons) tuple for the given score."""
    return (score, [f"reason_a(+{score})"] if score else [])


# ---------------------------------------------------------------------------
# TestBootstrapLabels
# ---------------------------------------------------------------------------

class TestBootstrapLabels:
    """bootstrap_labels maps heuristic score thresholds to label strings."""

    def test_high_score_labelled_bot(self):
        """Score >= 4 must produce label 'bot'."""
        users = {1: object(), 2: object(), 3: object()}
        scored = {
            1: _make_scored(1, 4),
            2: _make_scored(2, 5),
            3: _make_scored(3, 10),
        }
        result = bootstrap_labels(users, scored)
        for uid in (1, 2, 3):
            assert result[uid]["label"] == "bot", f"uid={uid} score={scored[uid][0]}"

    def test_zero_score_labelled_human(self):
        """Score == 0 must produce label 'human'."""
        users = {10: object()}
        scored = {10: _make_scored(10, 0)}
        result = bootstrap_labels(users, scored)
        assert result[10]["label"] == "human"

    def test_mid_score_labelled_unlabeled(self):
        """Scores 1, 2, 3 must produce label 'unlabeled'."""
        users = {1: object(), 2: object(), 3: object()}
        scored = {
            1: _make_scored(1, 1),
            2: _make_scored(2, 2),
            3: _make_scored(3, 3),
        }
        result = bootstrap_labels(users, scored)
        for uid, expected_score in [(1, 1), (2, 2), (3, 3)]:
            assert result[uid]["label"] == "unlabeled", (
                f"uid={uid} score={expected_score}"
            )

    def test_source_is_heuristic_bootstrap(self):
        """All bootstrap entries must carry source='heuristic_bootstrap'."""
        users = {1: object(), 2: object()}
        scored = {1: (0, []), 2: (4, [])}
        result = bootstrap_labels(users, scored)
        for uid in (1, 2):
            assert result[uid]["source"] == "heuristic_bootstrap"

    def test_timestamp_is_present_and_non_empty(self):
        """Each entry must have a non-empty 'timestamp' string."""
        users = {1: object()}
        scored = {1: (0, [])}
        result = bootstrap_labels(users, scored)
        ts = result[1]["timestamp"]
        assert isinstance(ts, str) and len(ts) > 0

    def test_empty_inputs_return_empty_dict(self):
        """No users / no scored → empty output dict."""
        assert bootstrap_labels({}, {}) == {}

    def test_only_users_in_scored_are_included(self):
        """Users missing from scored are not included in output."""
        users = {1: object(), 2: object()}
        scored = {1: (0, [])}
        result = bootstrap_labels(users, scored)
        assert 1 in result
        assert 2 not in result

    def test_score_exactly_4_is_bot(self):
        """Boundary: score == 4 → 'bot'."""
        users = {99: object()}
        scored = {99: (4, [])}
        result = bootstrap_labels(users, scored)
        assert result[99]["label"] == "bot"

    def test_score_exactly_3_is_unlabeled(self):
        """Boundary: score == 3 → 'unlabeled', not 'bot'."""
        users = {99: object()}
        scored = {99: (3, [])}
        result = bootstrap_labels(users, scored)
        assert result[99]["label"] == "unlabeled"


# ---------------------------------------------------------------------------
# TestLabelPersistence
# ---------------------------------------------------------------------------

class TestLabelPersistence:
    """save_labels / load_labels roundtrip and edge cases."""

    def test_roundtrip_preserves_labels(self, tmp_path):
        """Saved labels must survive a load/reload cycle unchanged."""
        labels = {
            1: {"label": "bot", "source": "heuristic_bootstrap", "timestamp": "2026-01-01T00:00:00"},
            42: {"label": "human", "source": "human", "timestamp": "2026-01-02T00:00:00"},
        }
        path = tmp_path / "labels.json"
        save_labels(labels, "@testchan", str(path))
        loaded = load_labels(str(path))

        # Keys must be converted back to int after JSON serialization
        assert 1 in loaded["labels"]
        assert 42 in loaded["labels"]
        assert loaded["labels"][1]["label"] == "bot"
        assert loaded["labels"][42]["label"] == "human"
        assert loaded["channel"] == "@testchan"

    def test_string_keys_converted_to_int_on_load(self, tmp_path):
        """JSON stores string keys; load_labels must return int user IDs."""
        labels = {
            100: {"label": "unlabeled", "source": "heuristic_bootstrap", "timestamp": "t"},
        }
        path = tmp_path / "labels.json"
        save_labels(labels, "@ch", str(path))

        # Verify the raw file indeed has string keys
        raw = json.loads(path.read_text())
        assert "100" in raw["labels"]

        # load_labels must return int keys
        loaded = load_labels(str(path))
        assert 100 in loaded["labels"]
        assert "100" not in loaded["labels"]

    def test_load_nonexistent_returns_empty_structure(self):
        """Loading a path that doesn't exist must return a valid empty structure."""
        result = load_labels("/nonexistent/path/labels.json")
        assert result["channel"] == ""
        assert result["version"] == 1
        assert result["labels"] == {}

    def test_file_permissions_are_600(self, tmp_path):
        """Saved label files must be chmod 600 (owner read/write only)."""
        labels = {1: {"label": "bot", "source": "heuristic_bootstrap", "timestamp": "t"}}
        path = tmp_path / "labels.json"
        save_labels(labels, "@ch", str(path))

        file_mode = stat.S_IMODE(os.stat(str(path)).st_mode)
        assert file_mode == 0o600, (
            f"Expected 0o600 but got {oct(file_mode)}"
        )

    def test_parent_dirs_created_if_missing(self, tmp_path):
        """save_labels must create missing parent directories."""
        deep_path = tmp_path / "nested" / "deep" / "labels.json"
        labels = {1: {"label": "human", "source": "heuristic_bootstrap", "timestamp": "t"}}
        # Must not raise even though parents don't exist yet
        save_labels(labels, "@ch", str(deep_path))
        assert deep_path.exists()

    def test_version_field_is_1(self, tmp_path):
        """Saved JSON must include version=1."""
        path = tmp_path / "labels.json"
        save_labels({}, "@ch", str(path))
        raw = json.loads(path.read_text())
        assert raw["version"] == 1

    def test_channel_field_preserved(self, tmp_path):
        """Channel name must be saved and reloaded correctly."""
        path = tmp_path / "labels.json"
        save_labels({}, "@mychannel", str(path))
        loaded = load_labels(str(path))
        assert loaded["channel"] == "@mychannel"


# ---------------------------------------------------------------------------
# TestLabelStats
# ---------------------------------------------------------------------------

class TestLabelStats:
    """label_stats returns correct aggregated counts."""

    def test_empty_labels_all_zero(self):
        """No labels → all counts are zero."""
        stats = label_stats({})
        assert stats == {"bot": 0, "human": 0, "unlabeled": 0, "total": 0, "human_labeled": 0}

    def test_bot_count(self):
        labels = {
            1: {"label": "bot", "source": "heuristic_bootstrap"},
            2: {"label": "bot", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["bot"] == 2

    def test_human_count(self):
        labels = {
            1: {"label": "human", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["human"] == 1

    def test_unlabeled_count(self):
        labels = {
            1: {"label": "unlabeled", "source": "heuristic_bootstrap"},
            2: {"label": "unlabeled", "source": "heuristic_bootstrap"},
            3: {"label": "unlabeled", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["unlabeled"] == 3

    def test_total_equals_sum_of_categories(self):
        labels = {
            1: {"label": "bot", "source": "heuristic_bootstrap"},
            2: {"label": "human", "source": "human"},
            3: {"label": "unlabeled", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["total"] == 3
        assert stats["total"] == stats["bot"] + stats["human"] + stats["unlabeled"]

    def test_human_labeled_counts_only_human_source(self):
        """human_labeled must count entries where source == 'human' regardless of label."""
        labels = {
            1: {"label": "bot", "source": "human"},
            2: {"label": "human", "source": "human"},
            3: {"label": "unlabeled", "source": "heuristic_bootstrap"},
            4: {"label": "bot", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["human_labeled"] == 2  # IDs 1 and 2 have source="human"

    def test_human_labeled_zero_when_no_human_source(self):
        labels = {
            1: {"label": "bot", "source": "heuristic_bootstrap"},
            2: {"label": "human", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["human_labeled"] == 0

    def test_mixed_counts_are_correct(self):
        """Comprehensive check with all three categories and human-labeled entries."""
        labels = {
            1: {"label": "bot", "source": "human"},
            2: {"label": "bot", "source": "heuristic_bootstrap"},
            3: {"label": "human", "source": "human"},
            4: {"label": "human", "source": "heuristic_bootstrap"},
            5: {"label": "unlabeled", "source": "heuristic_bootstrap"},
        }
        stats = label_stats(labels)
        assert stats["bot"] == 2
        assert stats["human"] == 2
        assert stats["unlabeled"] == 1
        assert stats["total"] == 5
        assert stats["human_labeled"] == 2  # IDs 1 and 3
