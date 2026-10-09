import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import pytest
from psycopg.rows import TupleRow

from rag.config import Settings, get_database_settings, get_settings
from rag.db import connect

FIXTURES = Path(__file__).parent / "fixtures"

Conn = psycopg.Connection[TupleRow]

# The `db` fixture drops the whole `rag` schema. These are the only hosts it
# will do that to, so a TEST_DATABASE_URL pasted from Supabase by mistake fails
# the test run instead of deleting real data.
THROWAWAY_HOSTS = {"localhost", "127.0.0.1", "postgres", "rag-test-db"}

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


@pytest.fixture
def db() -> Iterator[Conn]:
    """An autocommit connection to a throwaway Postgres, with `rag` wiped.

    Needs TEST_DATABASE_URL (see services/rag/README.md). Without it the test
    is skipped locally, but fails in CI, where a skip would let the job go
    green without testing anything.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is not set in CI, so the database tests would not run")
        pytest.skip("TEST_DATABASE_URL not set; see services/rag/README.md")

    host = urlparse(url).hostname
    if host not in THROWAWAY_HOSTS:
        pytest.fail(f"refusing to drop the rag schema on {host!r}: not a throwaway database")

    with connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS rag CASCADE")
        yield conn
