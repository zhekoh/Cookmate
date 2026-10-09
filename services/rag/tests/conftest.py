from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.config import Settings, get_database_settings, get_settings

# Fake values only. Tests must never need a real key: CI has none, and a test
# that silently used a real one could spend money or leak it in a log.
FAKE_ENV = {
    "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "VOYAGE_API_KEY": "pa-test-not-a-real-key",
    "RAG_SERVICE_TOKEN": "test-token",
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Give every test a clean environment.

    Clears the four variables, runs from an empty directory so a developer's
    real .env is never picked up, and resets the settings cache so one test's
    settings cannot leak into the next.
    """
    for name in FAKE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    get_database_settings.cache_clear()
    yield
    get_settings.cache_clear()
    get_database_settings.cache_clear()


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def settings(fake_env: None) -> Settings:
    return get_settings()
