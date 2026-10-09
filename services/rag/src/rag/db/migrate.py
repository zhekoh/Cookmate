"""Apply the SQL files in ``rag/migrations/`` in order, each exactly once.

    uv run python -m rag.db.migrate          # needs only DATABASE_URL

How it stays safe:

* **Recorded.** Each applied file is written to ``rag.schema_migrations`` with
  a checksum, and later runs skip it. Running twice is a no-op.
* **Atomic per file.** Each file runs in its own transaction, together with
  its ledger row. Postgres DDL is transactional, so a file that fails halfway
  leaves no half-built tables and no ledger row; fix it and run again.
* **Edits are refused.** If a file changes after it was applied, the database
  no longer matches the file people read to understand it. The runner stops
  instead of guessing. Write a new numbered file for the change.
* **One runner at a time.** A Postgres advisory lock means two deploys
  starting together cannot both apply the same file.
"""

import hashlib
import sys
from dataclasses import dataclass
from importlib.resources import files

import psycopg
from psycopg.rows import TupleRow

from rag.config import ConfigError, get_database_settings
from rag.db import connect

# Any fixed number works; every migration run takes the same one.
ADVISORY_LOCK_ID = 7_301_726

LEDGER_SQL = """
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.schema_migrations (
    version     text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


class MigrationError(RuntimeError):
    """The migrations on disk and the ledger in the database disagree."""


@dataclass(frozen=True)
class Migration:
    version: str  # the file name, e.g. "001_chunks.sql"; sorting it gives the order
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover() -> list[Migration]:
    """Every ``.sql`` file shipped inside the package, in name order."""
    folder = files("rag").joinpath("migrations")
    found = [
        Migration(version=entry.name, sql=entry.read_text(encoding="utf-8"))
        for entry in folder.iterdir()
        if entry.name.endswith(".sql")
    ]
    return sorted(found, key=lambda migration: migration.version)


def apply(
    conn: psycopg.Connection[TupleRow], migrations: list[Migration] | None = None
) -> list[str]:
    """Apply what has not been applied yet. Returns the versions applied now.

    ``conn`` must be in autocommit mode, so each ``conn.transaction()`` below
    is a real transaction rather than a savepoint inside one long one.
    """
    if not conn.autocommit:
        raise ValueError("apply() needs an autocommit connection")
    migrations = discover() if migrations is None else migrations

    conn.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
    try:
        with conn.transaction():
            conn.execute(LEDGER_SQL)

        rows = conn.execute("SELECT version, checksum FROM rag.schema_migrations").fetchall()
        recorded: dict[str, str] = {row[0]: row[1] for row in rows}

        applied_now: list[str] = []
        for migration in migrations:
            if migration.version in recorded:
                if recorded[migration.version] != migration.checksum:
                    raise MigrationError(
                        f"{migration.version} was edited after it was applied. "
                        "Put the change in a new migration file instead."
                    )
                continue

            with conn.transaction():
                # The SQL is a file shipped inside this package, never user
                # input, so it is safe to run as-is.
                conn.execute(migration.sql)
                conn.execute(
                    "INSERT INTO rag.schema_migrations (version, checksum) VALUES (%s, %s)",
                    (migration.version, migration.checksum),
                )
            applied_now.append(migration.version)
        return applied_now
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_ID,))


def main() -> int:
    try:
        settings = get_database_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    with connect(settings.database_url.get_secret_value(), autocommit=True) as conn:
        try:
            applied = apply(conn)
        except MigrationError as exc:
            print(f"migration refused: {exc}", file=sys.stderr)
            return 1

    if applied:
        for version in applied:
            print(f"applied {version}")
    else:
        print("database is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
