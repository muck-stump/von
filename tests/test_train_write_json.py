"""Regression test for training/train_option_marker.py's _write_json.

Laya issue-tracker research (NandhaKishorM/laya PR #267) flagged a matching
bug class: writing a config file directly via open(path, 'w') truncates the
old file before the new content is flushed, so a process killed mid-write
(exactly what launch_universal_training.py's watchdog does to training
instances -- see 8fd35a3db9c5) leaves a corrupt or empty file behind.
marker_calibration.json is read at serve time, so a truncated file there
breaks every request against that checkpoint, not just the training run.
"""

import importlib.util
import json
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "train_option_marker", os.path.join(os.path.dirname(__file__), "..", "training", "train_option_marker.py")
)
train_option_marker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_option_marker)


def test_write_json_writes_valid_content(tmp_path):
    path = tmp_path / "marker_calibration.json"
    train_option_marker._write_json(str(path), {"independent_options": True, "temperature": 1.23})
    assert json.loads(path.read_text()) == {"independent_options": True, "temperature": 1.23}


def test_write_json_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "marker_calibration.json"
    train_option_marker._write_json(str(path), {"a": 1})
    assert os.listdir(tmp_path) == ["marker_calibration.json"]


def test_write_json_does_not_truncate_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "marker_calibration.json"
    train_option_marker._write_json(str(path), {"good": "config"})

    # Simulate a mid-write crash (e.g. the training watchdog killing the
    # process) by making json.dump raise after the temp file is opened but
    # before the atomic rename happens.
    def _boom(*a, **kw):
        raise RuntimeError("simulated kill mid-write")

    monkeypatch.setattr(train_option_marker.json, "dump", _boom)
    with pytest.raises(RuntimeError):
        train_option_marker._write_json(str(path), {"bad": "config"})

    # The original file must be untouched -- a plain open(path, "w") would
    # have truncated it before json.dump ever ran.
    assert json.loads(path.read_text()) == {"good": "config"}
    assert os.listdir(tmp_path) == ["marker_calibration.json"]
