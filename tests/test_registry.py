"""Tests for registry JSON parsing, validation, and management."""

import json
import os
import pytest
import tempfile
from pathlib import Path

from tg_purge.commands.registry import _load_registry, _save_registry, _existing_ids


class TestLoadRegistry:
    """Test registry file loading and validation."""

    def test_new_registry_when_file_missing(self):
        registry = _load_registry("/nonexistent/file.json")
        assert registry["version"] == 1
        assert registry["entries"] == []
        assert registry["last_updated"] is None

    def test_load_valid_registry(self):
        data = {
            "version": 1,
            "description": "Test registry",
            "last_updated": "2026-02-28T00:00:00Z",
            "channel": "@test",
            "entries": [
                {"user_id": 0, "score": 5, "date_flagged": "2026-02-28",
                 "detection_method": "heuristic_scoring", "notes": "test"}
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = _load_registry(path)
            assert len(registry["entries"]) == 1
            assert registry["entries"][0]["user_id"] == 0
            assert registry["channel"] == "@test"
        finally:
            os.unlink(path)

    def test_malformed_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{invalid json!!!")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Malformed"):
                _load_registry(path)
        finally:
            os.unlink(path)

    def test_not_a_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([1, 2, 3], f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="JSON object"):
                _load_registry(path)
        finally:
            os.unlink(path)

    def test_missing_entries_key(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 1}, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="entries"):
                _load_registry(path)
        finally:
            os.unlink(path)

    def test_entries_not_a_list(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 1, "entries": "not a list"}, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="entries.*list"):
                _load_registry(path)
        finally:
            os.unlink(path)

    def test_empty_registry(self):
        data = {"version": 1, "entries": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = _load_registry(path)
            assert registry["entries"] == []
        finally:
            os.unlink(path)


class TestSaveRegistry:
    """Test registry file saving."""

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "registry.json")
            data = {
                "version": 1,
                "description": "Test",
                "entries": [{"user_id": 0, "score": 5}],
            }
            _save_registry(data, path)

            assert os.path.exists(path)
            with open(path) as f:
                saved = json.load(f)
            assert len(saved["entries"]) == 1
            assert saved["last_updated"] is not None

    def test_save_updates_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            data = {"version": 1, "entries": [], "last_updated": None}
            _save_registry(data, path)

            with open(path) as f:
                saved = json.load(f)
            assert saved["last_updated"] is not None
            assert "2026" in saved["last_updated"] or "20" in saved["last_updated"]


class TestExistingIds:
    """Test existing ID extraction."""

    def test_extracts_ids(self):
        data = {
            "entries": [
                {"user_id": 100},
                {"user_id": 200},
                {"user_id": 300},
            ]
        }
        ids = _existing_ids(data)
        assert ids == {100, 200, 300}

    def test_empty_entries(self):
        data = {"entries": []}
        ids = _existing_ids(data)
        assert ids == set()

    def test_skips_entries_without_user_id(self):
        data = {
            "entries": [
                {"user_id": 100},
                {"notes": "no id here"},
                {"user_id": 300},
            ]
        }
        ids = _existing_ids(data)
        assert ids == {100, 300}

    def test_duplicate_ids(self):
        data = {
            "entries": [
                {"user_id": 100},
                {"user_id": 100},
            ]
        }
        ids = _existing_ids(data)
        assert ids == {100}
