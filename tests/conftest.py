"""Shared test fixtures for alibi test suite."""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.extraction.yaml_cache import set_yaml_store_root, reset_yaml_store


@pytest.fixture(autouse=True)
def _isolate_yaml_store(tmp_path: Path) -> Generator[None, None, None]:
    """Isolate yaml_store_root to tmp_path for every test.

    Prevents tests from writing to the real data/yaml_store/ directory.
    Tests that need a specific store root can override via set_yaml_store_root().
    """
    store = tmp_path / "yaml_store"
    set_yaml_store_root(store)
    yield
    set_yaml_store_root(None)
    reset_yaml_store()


@pytest.fixture
def db_manager(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create an initialized database manager with temp database."""
    reset_config()
    config = Config(db_path=tmp_path / "test.db")
    manager = DatabaseManager(config)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def db(db_manager: DatabaseManager) -> DatabaseManager:
    """Alias for db_manager — used by many test files."""
    return db_manager


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock DatabaseManager for unit tests."""
    mock = MagicMock(spec=DatabaseManager)
    mock.is_initialized.return_value = True
    mock.fetchall.return_value = []
    return mock
