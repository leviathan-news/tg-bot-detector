"""
Tests for tg_purge/ml.py — ML training and inference pipeline.

All tests are skipped when sklearn is not installed. Tests verify:
- ml_available() reflects real import availability.
- train_model() succeeds with sufficient balanced samples.
- train_model() returns structured failure dicts for edge cases.
- predict() returns probability/label dicts per sample.
- load_model_metadata() round-trips the JSON metadata.
"""

import os
import json
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Probe for sklearn before importing ml module to set the skip marker.
# ---------------------------------------------------------------------------
try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Skip the entire module if ML deps are absent.
pytestmark = pytest.mark.skipif(
    not HAS_SKLEARN,
    reason="ML deps not installed (sklearn missing)",
)

from tg_purge.ml import (  # noqa: E402 — import after skip guard
    ml_available,
    train_model,
    predict,
    load_model_metadata,
)
from tg_purge.features import FEATURE_KEYS  # noqa: E402


# ---------------------------------------------------------------------------
# Feature vector builders
# ---------------------------------------------------------------------------

def _bot_vector() -> dict:
    """
    Construct a feature vector that looks like a bot.

    Sets the strongest bot signals:
    - is_deleted=1: account has been deleted by Telegram.
    - status_empty=1: never seen online.
    - heuristic_score=5: high heuristic risk score.
    All other keys default to 0.0.
    """
    vec = {k: 0.0 for k in FEATURE_KEYS}
    vec["is_deleted"] = 1.0
    vec["status_empty"] = 1.0
    vec["heuristic_score"] = 5.0
    return vec


def _human_vector() -> dict:
    """
    Construct a feature vector that looks like a real human user.

    Sets the strongest human signals:
    - has_photo=1: profile photo is set.
    - has_username=1: public @handle is set.
    - status_recently=1: was online recently.
    - heuristic_score=0: no heuristic risk signals.
    All other keys default to 0.0.
    """
    vec = {k: 0.0 for k in FEATURE_KEYS}
    vec["has_photo"] = 1.0
    vec["has_username"] = 1.0
    vec["status_recently"] = 1.0
    vec["heuristic_score"] = 0.0
    return vec


def _make_dataset(n_bots: int = 30, n_humans: int = 30):
    """
    Build a balanced list of feature dicts and corresponding string labels.

    Returns (features: list[dict], labels: list[str]).
    """
    features = [_bot_vector() for _ in range(n_bots)] + \
               [_human_vector() for _ in range(n_humans)]
    labels = ["bot"] * n_bots + ["human"] * n_humans
    return features, labels


# ---------------------------------------------------------------------------
# TestMLAvailability
# ---------------------------------------------------------------------------

class TestMLAvailability:
    """ml_available() must return True when sklearn is importable."""

    def test_ml_available_returns_true(self):
        # Because the entire module is skipped when sklearn is missing,
        # reaching this line means sklearn IS installed, so we expect True.
        assert ml_available() is True


# ---------------------------------------------------------------------------
# TestTrainModel
# ---------------------------------------------------------------------------

class TestTrainModel:
    """train_model() — success and failure cases."""

    def test_train_balanced_dataset_succeeds(self, tmp_path):
        """
        30 bot + 30 human samples must produce a successful training result
        with model and metadata files on disk, and a positive F1 score.
        """
        features, labels = _make_dataset(30, 30)
        result = train_model(features, labels, output_dir=str(tmp_path))

        assert result["success"] is True, f"Expected success, got: {result}"
        assert os.path.isfile(result["model_file"]), "model_file path must exist on disk"
        assert os.path.isfile(result["metadata_file"]), "metadata_file path must exist on disk"
        assert "algorithm" in result
        assert "metrics" in result
        metrics = result["metrics"]
        # F1 must be a non-negative float; with a perfectly separable synthetic
        # dataset it should be > 0.
        assert metrics.get("f1", -1) > 0, f"Expected F1 > 0, got: {metrics}"

    def test_train_returns_auc_roc_in_metrics(self, tmp_path):
        """Metrics dict must contain auc_roc, precision, recall in addition to f1."""
        features, labels = _make_dataset(30, 30)
        result = train_model(features, labels, output_dir=str(tmp_path))

        assert result["success"] is True
        metrics = result["metrics"]
        for key in ("f1", "precision", "recall", "auc_roc"):
            assert key in metrics, f"metrics missing key '{key}'"

    def test_train_insufficient_samples_fails(self, tmp_path):
        """
        Fewer than 10 total samples must return success=False with 'insufficient'
        somewhere in the error string (case-insensitive).
        """
        features, labels = _make_dataset(3, 3)  # only 6 samples
        result = train_model(features, labels, output_dir=str(tmp_path))

        assert result["success"] is False
        assert "insufficient" in result.get("error", "").lower(), (
            f"Expected 'insufficient' in error, got: {result.get('error')}"
        )

    def test_train_single_class_fails(self, tmp_path):
        """
        All samples belonging to a single class must return success=False
        with 'class' somewhere in the error string (case-insensitive).
        """
        # 20 bots, 0 humans — only one class.
        features, labels = _make_dataset(20, 0)
        result = train_model(features, labels, output_dir=str(tmp_path))

        assert result["success"] is False
        assert "class" in result.get("error", "").lower(), (
            f"Expected 'class' in error, got: {result.get('error')}"
        )

    def test_train_creates_output_directory(self):
        """train_model must create output_dir if it does not exist."""
        with tempfile.TemporaryDirectory() as base:
            new_dir = os.path.join(base, "nested", "models")
            features, labels = _make_dataset(20, 20)
            result = train_model(features, labels, output_dir=new_dir)

            assert result["success"] is True
            assert os.path.isdir(new_dir), "output_dir should have been created"

    def test_train_channel_slug_in_filenames(self, tmp_path):
        """When channel= is provided, it should appear in the output filenames."""
        features, labels = _make_dataset(20, 20)
        result = train_model(
            features, labels,
            output_dir=str(tmp_path),
            channel="testchannel",
        )

        assert result["success"] is True
        assert "testchannel" in result["model_file"]


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:
    """predict() — must return probability/label per sample."""

    @pytest.fixture(scope="class")
    def trained_model(self, tmp_path_factory):
        """Train a model once and reuse the paths across all predict tests."""
        tmp = tmp_path_factory.mktemp("model")
        features, labels = _make_dataset(30, 30)
        result = train_model(features, labels, output_dir=str(tmp))
        assert result["success"] is True, f"Setup failed: {result}"
        return result

    def test_predict_returns_list_of_dicts(self, trained_model):
        """predict() must return a list with one dict per input sample."""
        sample = [_bot_vector()]
        results = predict(
            sample,
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert "probability" in results[0]
        assert "label" in results[0]
        assert "heuristic_score" in results[0]

    def test_predict_probability_in_range(self, trained_model):
        """Each probability must be a float in [0.0, 1.0]."""
        samples = [_bot_vector(), _human_vector()]
        results = predict(
            samples,
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        for r in results:
            assert 0.0 <= r["probability"] <= 1.0, (
                f"probability out of range: {r['probability']}"
            )

    def test_predict_label_is_valid(self, trained_model):
        """Each label must be either 'bot' or 'human'."""
        samples = [_bot_vector(), _human_vector()]
        results = predict(
            samples,
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        for r in results:
            assert r["label"] in ("bot", "human"), (
                f"Unexpected label: {r['label']}"
            )

    def test_predict_metadata_path_inferred_from_model_path(self, trained_model):
        """
        If metadata_path=None, predict() must derive it from model_path
        by replacing the model extension with .json.
        """
        results = predict(
            [_bot_vector()],
            model_path=trained_model["model_file"],
            metadata_path=None,  # explicit None — must be derived
        )
        assert len(results) == 1
        assert "probability" in results[0]

    def test_predict_bot_vector_higher_probability(self, trained_model):
        """
        A bot-like vector should have a higher bot probability than a
        human-like vector when the model is trained on clearly separable data.
        """
        bot_results = predict(
            [_bot_vector()],
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        human_results = predict(
            [_human_vector()],
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        assert bot_results[0]["probability"] >= human_results[0]["probability"], (
            f"Bot prob ({bot_results[0]['probability']}) should be >= "
            f"human prob ({human_results[0]['probability']})"
        )

    def test_predict_multiple_samples(self, trained_model):
        """predict() on a batch of N samples must return exactly N results."""
        samples = [_bot_vector()] * 10 + [_human_vector()] * 10
        results = predict(
            samples,
            model_path=trained_model["model_file"],
            metadata_path=trained_model["metadata_file"],
        )
        assert len(results) == 20


# ---------------------------------------------------------------------------
# TestModelMetadata
# ---------------------------------------------------------------------------

class TestModelMetadata:
    """load_model_metadata() and metadata content verification."""

    @pytest.fixture(scope="class")
    def meta(self, tmp_path_factory):
        """Train a model and return its loaded metadata dict."""
        tmp = tmp_path_factory.mktemp("meta_test")
        features, labels = _make_dataset(30, 30)
        result = train_model(features, labels, output_dir=str(tmp))
        assert result["success"] is True, f"Setup failed: {result}"
        return load_model_metadata(result["metadata_file"])

    def test_metadata_has_algorithm(self, meta):
        """Metadata must contain the 'algorithm' field (non-empty string)."""
        assert "algorithm" in meta
        assert isinstance(meta["algorithm"], str)
        assert meta["algorithm"]  # non-empty

    def test_metadata_has_metrics(self, meta):
        """Metadata must contain f1, precision, recall, auc_roc under 'metrics'."""
        assert "metrics" in meta
        for key in ("f1", "precision", "recall", "auc_roc"):
            assert key in meta["metrics"], f"metrics missing '{key}'"

    def test_metadata_has_feature_names(self, meta):
        """
        Metadata must contain 'feature_names' matching FEATURE_KEYS exactly
        (order may differ — compare as sets).
        """
        assert "feature_names" in meta
        assert set(meta["feature_names"]) == set(FEATURE_KEYS)

    def test_metadata_has_n_samples(self, meta):
        """Metadata must record per-class sample counts (30 bot + 30 human = 60)."""
        assert "n_samples" in meta
        assert isinstance(meta["n_samples"], dict)
        assert meta["n_samples"]["bot"] == 30
        assert meta["n_samples"]["human"] == 30

    def test_metadata_has_created_timestamp(self, meta):
        """Metadata must contain a 'created' field (ISO 8601 timestamp string)."""
        assert "created" in meta
        assert isinstance(meta["created"], str)
        assert meta["created"]  # non-empty

    def test_metadata_has_threshold(self, meta):
        """Metadata must record the decision threshold (0.5 by default)."""
        assert "threshold" in meta
        assert meta["threshold"] == 0.5

    def test_metadata_round_trips_from_json(self, tmp_path):
        """load_model_metadata() must faithfully deserialise the JSON on disk."""
        features, labels = _make_dataset(20, 20)
        result = train_model(features, labels, output_dir=str(tmp_path))
        assert result["success"] is True

        # Load via the public function.
        meta = load_model_metadata(result["metadata_file"])

        # Also load the raw JSON for comparison.
        with open(result["metadata_file"]) as fh:
            raw = json.load(fh)

        assert meta == raw
