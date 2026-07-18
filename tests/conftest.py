"""Shared pytest fixtures: isolate the SQLite database to a temp path per test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point core.db at a fresh temp database for every test."""
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "test.db")
    db._initialized_paths.clear()
