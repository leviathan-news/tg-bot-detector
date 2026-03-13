"""
ML training and inference pipeline for Telegram bot detection.

This module sits on top of features.py and provides three main capabilities:
  1. Training an ensemble classifier (LightGBM > XGBoost > sklearn RF, in order
     of preference) from labelled feature vectors.
  2. Persisting the trained model and a JSON metadata file to disk.
  3. Loading a saved model + metadata and running batch inference.

Design constraints:
- All heavy imports (sklearn, lightgbm, xgboost, joblib) are done lazily inside
  functions so that importing this module never raises ImportError even when
  optional ML deps are absent. Callers must check ml_available() first.
- File permissions follow the same security pattern as client.py: output
  directory is chmod 700, individual files are chmod 600 (best-effort, no crash
  on non-POSIX systems).
- The module is testable without a real Telegram connection — it operates only
  on plain Python dicts (feature vectors) and string labels.

Security note on joblib:
  joblib.dump/load uses Python pickle for sklearn models. These files are
  trusted artifacts produced and consumed locally by the same installation.
  Never load a joblib model file from an untrusted source.
"""

from __future__ import annotations

import json
import os
import stat
import datetime
from typing import List, Dict, Optional, Any, Tuple

from .utils import channel_slug as _safe_channel_slug


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def ml_available() -> bool:
    """
    Return True if the core ML dependency (scikit-learn) is importable.

    Used by callers and CLI commands as a guard before calling train_model()
    or predict(). Returns False gracefully when sklearn is absent — no
    exception is raised.
    """
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------

def _get_available_models() -> List[Tuple[str, Any]]:
    """
    Probe LightGBM, XGBoost, and sklearn RandomForest in priority order.

    Returns a list of (name, class) pairs for every backend that can be
    imported without error. The order determines which backend is tried
    first during training: LightGBM is preferred because it is the fastest
    and generally best-performing on tabular data; sklearn RF is the
    universal fallback.

    Each import is wrapped in a try/except so a missing native library
    (e.g. libomp.dylib on macOS) does not abort the entire probe.
    """
    available = []

    # --- LightGBM ---
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        available.append(("lightgbm", LGBMClassifier))
    except Exception:
        # ImportError or OSError (missing shared library such as libomp.dylib).
        pass

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier  # type: ignore
        available.append(("xgboost", XGBClassifier))
    except Exception:
        pass

    # --- sklearn RandomForest (universal fallback) ---
    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        available.append(("sklearn_rf", RandomForestClassifier))
    except Exception:
        pass

    return available


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _features_to_array(features: List[Dict[str, float]], keys: List[str]):
    """
    Convert a list of feature dicts to a 2-D numpy float64 array.

    Args:
        features: List of dicts, each mapping feature name to float value.
        keys: Ordered list of feature names to extract. Keys missing from a
            dict default to 0.0 so partial vectors are handled gracefully.

    Returns:
        numpy.ndarray of shape (n_samples, n_features).
    """
    import numpy as np  # lazy import — guaranteed available when sklearn is

    n = len(features)
    k = len(keys)
    X = np.zeros((n, k), dtype=np.float64)
    for i, fvec in enumerate(features):
        for j, key in enumerate(keys):
            X[i, j] = fvec.get(key, 0.0)
    return X


def _set_secure_permissions(path: str) -> None:
    """
    Apply restrictive permissions to a file (chmod 600) best-effort.

    Mirrors the pattern used in client.py for session files. Failures
    (e.g. on Windows, or when the path is on a read-only FS) are silently
    swallowed — the function must never crash the caller.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass


def _make_output_dir(output_dir: str) -> None:
    """
    Create output_dir (and parents) with restrictive permissions (700).

    Safe to call when the directory already exists — os.makedirs with
    exist_ok=True is used. chmod is best-effort.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        os.chmod(output_dir, stat.S_IRWXU)  # 0o700
    except OSError:
        pass



def _build_model_filename(output_dir: str, algorithm: str, channel: Optional[str]) -> str:
    """
    Construct the model file path from the output directory, algorithm, and channel.

    File extension conventions:
      lightgbm  -> .model  (LightGBM Booster text format)
      xgboost   -> .model  (XGBoost model format)
      sklearn_rf -> .joblib (joblib-serialised RandomForest)

    The channel slug (if any) is prepended so files from different channels
    do not collide in a shared models/ directory.
    """
    ext_map = {
        "lightgbm":  "model",
        "xgboost":   "model",
        "sklearn_rf": "joblib",
    }
    ext = ext_map.get(algorithm, "model")
    slug = _safe_channel_slug(channel)
    stem = f"{slug}_{algorithm}" if slug else algorithm
    return os.path.join(output_dir, f"{stem}.{ext}")


def _build_metadata_filename(model_path: str) -> str:
    """
    Derive the metadata JSON path from the model file path.

    Strips the existing extension and appends '.json' so the two files
    always share the same base name and sit in the same directory.
    """
    base, _ = os.path.splitext(model_path)
    return f"{base}.json"


def _train_single_backend(
    name: str,
    cls: Any,
    X_train,
    y_train,
    scale_pos_weight: float,
) -> Any:
    """
    Instantiate and fit a single classifier backend.

    Hyper-parameters are fixed for reproducibility:
    - n_estimators=100: small enough to be fast, large enough to generalise.
    - random_state=42: reproducible splits.
    - scale_pos_weight / class_weight: compensate for class imbalance.

    Args:
        name: Backend identifier string ('lightgbm', 'xgboost', 'sklearn_rf').
        cls: The classifier class to instantiate.
        X_train: numpy array of training features.
        y_train: numpy array of binary labels (0=human, 1=bot).
        scale_pos_weight: Ratio of negatives to positives for imbalance handling.

    Returns:
        Fitted classifier instance.
    """
    if name == "lightgbm":
        model = cls(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,  # suppress all LightGBM output
        )
    elif name == "xgboost":
        model = cls(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",  # avoid deprecated default warning
            verbosity=0,            # suppress XGBoost output
        )
    else:  # sklearn_rf
        model = cls(
            n_estimators=100,
            class_weight="balanced",  # sklearn RF uses class_weight instead of scale_pos_weight
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model


def _evaluate_model(model, name: str, X_test, y_test) -> Dict[str, float]:
    """
    Compute F1, precision, recall, and AUC-ROC on the test split.

    Uses binary F1/precision/recall (positive class = bot = 1), which is
    appropriate for our binary classification task.

    Args:
        model: Fitted classifier.
        name: Backend name — used to select the probability API.
        X_test: numpy array of test features.
        y_test: numpy array of binary test labels.

    Returns:
        Dict with keys: f1, precision, recall, auc_roc (all floats in [0, 1]).
    """
    from sklearn.metrics import (  # type: ignore
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Predict class labels and probabilities.
    y_pred = model.predict(X_test)
    # All three backends expose predict_proba(); column 1 is P(class=1=bot).
    y_prob = model.predict_proba(X_test)[:, 1]

    # Cast predictions to int for sklearn metric functions.
    y_pred_int = [int(v) for v in y_pred]

    # Guard against degenerate test splits where only one class is present.
    # roc_auc_score raises ValueError in that case.
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.0

    return {
        "f1":        float(f1_score(y_test, y_pred_int, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_int, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred_int, zero_division=0)),
        "auc_roc":   auc,
    }


def _save_model(model, name: str, model_path: str) -> None:
    """
    Persist the fitted model to disk using the backend-native serialisation format.

    - LightGBM: booster_.save_model() writes the text-format booster file.
    - XGBoost: model.save_model() writes the model format.
    - sklearn: joblib.dump() uses pickle-based compression (trusted local use only —
      see module-level security note).

    Each format is chosen to maximise forward compatibility within its own
    library ecosystem.
    """
    if name == "lightgbm":
        # LGBMClassifier wraps a Booster; save via the underlying booster.
        model.booster_.save_model(model_path)
    elif name == "xgboost":
        model.save_model(model_path)
    else:  # sklearn_rf — uses joblib (trusted local artifacts only)
        import joblib  # type: ignore
        joblib.dump(model, model_path)


def _save_metadata(
    metadata: Dict[str, Any],
    metadata_path: str,
) -> None:
    """
    Write the metadata dict to a JSON file with 2-space indentation.

    Performs an atomic write-then-rename to avoid leaving a partial file on
    crash. Permissions are set to 600 afterwards.
    """
    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    os.replace(tmp_path, metadata_path)
    _set_secure_permissions(metadata_path)


# ---------------------------------------------------------------------------
# Public API — Training
# ---------------------------------------------------------------------------

def train_model(
    features: List[Dict[str, float]],
    labels: List[str],
    output_dir: str = "models",
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a bot-detection classifier from labelled feature vectors.

    Training process:
      1. Validate inputs (>= 10 samples, both classes present).
      2. Convert features to numpy array using sorted keys from features[0].
      3. Convert "bot"/"human" labels to 1/0 integers.
      4. Stratified 80/20 train/test split (random_state=42).
      5. Compute scale_pos_weight = n_negatives / n_positives.
      6. Try all available backends (LightGBM then XGBoost then sklearn RF).
      7. Pick the backend with the highest test-set F1 score.
      8. Save model and JSON metadata to output_dir with chmod 600.

    Args:
        features: List of feature dicts (output of extract_features()).
            All dicts must share the same keys; missing keys default to 0.0.
        labels: Parallel list of "bot" or "human" strings.
        output_dir: Directory path where model and metadata files are written.
            Created (including parents) if it does not exist.
        channel: Optional channel name (with or without leading @). Used to
            prefix output filenames so multiple channels can share one dir.

    Returns:
        On success::
            {
                "success": True,
                "model_file": str,       # absolute path to saved model
                "metadata_file": str,    # absolute path to JSON metadata
                "metrics": dict,         # f1, precision, recall, auc_roc
                "algorithm": str,        # winning backend name
            }
        On failure::
            {"success": False, "error": str}
    """
    # --- Guard: sklearn must be available ---
    if not ml_available():
        return {"success": False, "error": "ML dependencies not installed (sklearn missing)"}

    # --- Guard: minimum sample count ---
    n_total = len(features)
    if n_total < 10:
        return {
            "success": False,
            "error": f"Insufficient samples: need at least 10, got {n_total}",
        }

    # --- Guard: both classes must be present ---
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return {
            "success": False,
            "error": (
                f"Only one class present in labels: {unique_labels}. "
                "Both 'bot' and 'human' labels are required."
            ),
        }

    # --- Convert features to numpy ---
    # Sort keys so the array column order is deterministic across calls.
    import numpy as np

    feature_names: List[str] = sorted(features[0].keys())
    X = _features_to_array(features, feature_names)

    # Map "bot" -> 1, "human" -> 0 (any other string -> 0 with a silent cast).
    y = np.array([1 if lbl == "bot" else 0 for lbl in labels], dtype=np.int32)

    # --- Stratified train/test split ---
    from sklearn.model_selection import train_test_split  # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --- Class imbalance weight ---
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    # Guard: avoid ZeroDivisionError if one class disappears after split.
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    # --- Try all available backends, keep the one with the best F1 ---
    backends = _get_available_models()
    if not backends:
        return {
            "success": False,
            "error": "No ML backend available (sklearn, lightgbm, xgboost all missing)",
        }

    best_model = None
    best_name = ""
    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0

    for name, cls in backends:
        try:
            model = _train_single_backend(name, cls, X_train, y_train, scale_pos_weight)
            metrics = _evaluate_model(model, name, X_test, y_test)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_name = name
                best_metrics = metrics
        except Exception:
            # A backend may partially fail (numpy incompatibility, etc.).
            # Skip it silently and try the next available backend.
            continue

    if best_model is None:
        return {"success": False, "error": "All available ML backends failed during training"}

    # --- Persist model and metadata ---
    _make_output_dir(output_dir)
    model_path = _build_model_filename(output_dir, best_name, channel)
    metadata_path = _build_metadata_filename(model_path)

    _save_model(best_model, best_name, model_path)
    _set_secure_permissions(model_path)

    # Per-class sample counts for the metadata (spec requires {"bot": N, "human": M}).
    import numpy as np  # already imported above, but keep this block self-contained
    n_bot = int(np.sum(y == 1))
    n_human = int(np.sum(y == 0))

    metadata: Dict[str, Any] = {
        "version":       1,
        "trained_on":    [channel] if channel else [],
        "algorithm":     best_name,
        "n_samples":     {"bot": n_bot, "human": n_human},
        "metrics":       best_metrics,
        "feature_names": feature_names,
        "threshold":     0.5,
        "created":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _save_metadata(metadata, metadata_path)

    return {
        "success":       True,
        "model_file":    os.path.abspath(model_path),
        "metadata_file": os.path.abspath(metadata_path),
        "metrics":       best_metrics,
        "algorithm":     best_name,
    }


# ---------------------------------------------------------------------------
# Public API — Metadata
# ---------------------------------------------------------------------------

def load_model_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Read and deserialise the JSON metadata file saved by train_model().

    Args:
        metadata_path: Path to the .json metadata file.

    Returns:
        dict with the same structure as the metadata dict written by train_model().

    Raises:
        FileNotFoundError: if metadata_path does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Public API — Inference
# ---------------------------------------------------------------------------

def predict(
    features: List[Dict[str, float]],
    model_path: str,
    metadata_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a saved model and run inference on a list of feature vectors.

    The decision threshold from the metadata (default 0.5) is used to
    convert the raw bot probability into a "bot" or "human" label.

    Loading strategy by algorithm (stored in metadata):
    - lightgbm:  Booster.load_model() then predict() which returns probabilities
                 directly for binary classification.
    - xgboost:   XGBClassifier.load_model() then predict_proba()[:, 1].
    - sklearn_rf: joblib.load() then predict_proba()[:, 1] (trusted local files only).

    Args:
        features: List of feature dicts. Keys must match the feature_names
            stored in the metadata; missing keys default to 0.0.
        model_path: Path to the saved model file.
        metadata_path: Path to the JSON metadata file. When None, derived from
            model_path by stripping the extension and appending '.json'.

    Returns:
        List of dicts, one per input sample:
            [{"probability": float, "label": "bot" | "human"}, ...]
    """
    # --- Derive metadata path if not given ---
    if metadata_path is None:
        metadata_path = _build_metadata_filename(model_path)

    meta = load_model_metadata(metadata_path)
    algorithm: str = meta["algorithm"]
    feature_names: List[str] = meta["feature_names"]
    threshold: float = float(meta.get("threshold", 0.5))

    # --- Build feature matrix ---
    X = _features_to_array(features, feature_names)

    # --- Load model and produce probabilities ---
    probabilities = _load_and_predict(algorithm, model_path, X)

    # --- Apply threshold to assign labels ---
    # Ensemble mode: when the model's probability is below the main threshold
    # but above a lower "boost" threshold, and the heuristic score confirms
    # suspicion (>= heuristic_boost_threshold), label as bot. This catches
    # stealthy bots that evade the ML model but trigger heuristic signals.
    #
    # Default: boost_threshold=0.15, heuristic_boost=2 (validated on 1,132
    # human reviews: F1 0.821→0.872, catches 87 more bots with 41 more FP).
    results: List[Dict[str, Any]] = []
    for i, prob in enumerate(probabilities):
        p = float(prob)
        heuristic_score = features[i].get("heuristic_score", 0.0) if i < len(features) else 0.0
        h = float(heuristic_score)

        # Primary: ML model confident → bot.
        # Boost: ML has some signal AND heuristic agrees → bot.
        if p >= threshold:
            label = "bot"
        elif p >= 0.15 and h >= 2.0:
            label = "bot"
        else:
            label = "human"

        results.append({
            "probability": p,
            "label": label,
            "heuristic_score": heuristic_score,
        })
    return results


def _load_and_predict(algorithm: str, model_path: str, X) -> list:
    """
    Dispatch model loading and probability extraction by algorithm name.

    Args:
        algorithm: One of 'lightgbm', 'xgboost', 'sklearn_rf'.
        model_path: Filesystem path to the saved model artifact.
        X: numpy array of shape (n_samples, n_features).

    Returns:
        Flat list of float probabilities (P(bot)) of length n_samples.

    Raises:
        ValueError: if algorithm is not recognised.
    """
    if algorithm == "lightgbm":
        # Booster.predict() returns raw probabilities for binary classification.
        # No need to index a second column.
        from lightgbm import Booster  # type: ignore
        booster = Booster(model_file=model_path)
        probs = booster.predict(X)
        return list(probs)

    elif algorithm == "xgboost":
        from xgboost import XGBClassifier  # type: ignore
        model = XGBClassifier()
        model.load_model(model_path)
        # predict_proba returns (n_samples, 2); column 1 is P(bot=1).
        probs = model.predict_proba(X)[:, 1]
        return list(probs)

    elif algorithm == "sklearn_rf":
        # Security note: joblib loads a pickle-based file. Only load files
        # produced by this application from trusted local storage.
        import joblib  # type: ignore
        model = joblib.load(model_path)
        probs = model.predict_proba(X)[:, 1]
        return list(probs)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
