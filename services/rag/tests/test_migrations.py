"""Migration runner and schema tests, against a real Postgres with pgvector.

These need a throwaway database. Locally:

    docker compose --profile test up -d rag-test-db
    export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres

Without TEST_DATABASE_URL they are skipped, except in CI, where a skip would
let the job go green without testing anything, so it fails instead.
"""

import os
from collections.abc import Iterator
from urllib.parse import urlparse

import psycopg
import pytest
from psycopg.rows import TupleRow

from rag.db import connect
from rag.db.migrate import Migration, MigrationError, apply, discover

# The fixture below drops the whole `rag` schema. These are the only hosts it
# will do that to, so a TEST_DATABASE_URL pasted from Supabase by mistake
# fails the test run instead of deleting real data.
THROWAWAY_HOSTS = {"localhost", "127.0.0.1", "postgres", "rag-test-db"}

Conn = psycopg.Connection[TupleRow]


@pytest.fixture
def db() -> Iterator[Conn]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is not set in CI, so the database tests would not run")
        pytest.skip("TEST_DATABASE_URL not set; see the docstring at the top of this file")

    host = urlparse(url).hostname
    if host not in THROWAWAY_HOSTS:
        pytest.fail(f"refusing to drop the rag schema on {host!r}: not a throwaway database")

    with connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS rag CASCADE")
        yield conn


def table_exists(conn: Conn, name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (name,)).fetchone()
    return bool(row and row[0])


def ledger(conn: Conn) -> list[str]:
    rows = conn.execute("SELECT version FROM rag.schema_migrations ORDER BY version").fetchall()
    return [row[0] for row in rows]


# ---- the runner -------------------------------------------------------------


def test_a_fresh_database_gets_every_migration(db: Conn) -> None:
    applied = apply(db)

    assert applied == [m.version for m in discover()]
    assert ledger(db) == applied
    for table in ("rag.sources", "rag.index_versions", "rag.chunks"):
        assert table_exists(db, table)


def test_running_twice_applies_nothing_the_second_time(db: Conn) -> None:
    apply(db)

    assert apply(db) == []
    assert ledger(db) == [m.version for m in discover()]


def test_a_migration_edited_after_it_ran_is_refused(db: Conn) -> None:
    apply(db, [Migration("001_example.sql", "SELECT 1")])

    with pytest.raises(MigrationError, match=r"001_example\.sql was edited"):
        apply(db, [Migration("001_example.sql", "SELECT 2")])


def test_a_failing_migration_leaves_nothing_behind(db: Conn) -> None:
    broken = Migration(
        "001_broken.sql",
        "CREATE TABLE rag.half_done (x int); SELECT this_function_does_not_exist();",
    )

    with pytest.raises(psycopg.errors.UndefinedFunction):
        apply(db, [broken])

    # The CREATE TABLE ran, then the transaction rolled it back.
    assert not table_exists(db, "rag.half_done")
    assert ledger(db) == []


def test_apply_refuses_a_connection_that_is_not_autocommit(db: Conn) -> None:
    db.autocommit = False
    try:
        with pytest.raises(ValueError, match="autocommit"):
            apply(db)
    finally:
        db.rollback()
        db.autocommit = True


# ---- the schema -------------------------------------------------------------


def add_source_and_version(conn: Conn, *, status: str = "building") -> int:
    conn.execute(
        """
        INSERT INTO sources (id, kind, title, url, licence, authority, trusted,
                             jurisdiction, retrieved_on)
        VALUES ('bccampus-food-safety', 'bccampus',
                'Food Safety, Sanitation, and Personal Hygiene',
                'https://opentextbc.ca/foodsafety/', 'CC BY 4.0', 'high', true,
                'CA', '2026-10-09')
        ON CONFLICT (id) DO NOTHING
        """
    )
    row = conn.execute(
        """
        INSERT INTO index_versions (chunker, chunker_version, embedding_model,
                                    embedding_dim, status)
        VALUES ('heading_aware', '1', 'test-model', 3, %s)
        RETURNING id
        """,
        (status,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def add_chunk(conn: Conn, version_id: int, ordinal: int, content: str, embedding: str) -> None:
    conn.execute(
        """
        INSERT INTO chunks (index_version_id, source_id, ordinal, chapter, section,
                            heading_path, url, attribution, content, token_count, embedding)
        VALUES (%s, 'bccampus-food-safety', %s, 'Chapter 1', 'Section', 'Book > Chapter 1',
                'https://opentextbc.ca/foodsafety/', 'attribution line', %s, 10, %s)
        """,
        (version_id, ordinal, content, embedding),
    )


def test_a_chunk_round_trips_with_its_vector_and_keywords(db: Conn) -> None:
    apply(db)
    version = add_source_and_version(db)
    add_chunk(db, version, 0, "Wash hands before handling food.", "[1, 0, 0]")
    add_chunk(db, version, 1, "Keep cold food below 4 degrees.", "[0, 1, 0]")

    # Nearest neighbour by cosine distance. The question vector points almost
    # exactly at the first chunk.
    row = db.execute(
        """
        SELECT c.content, c.embedding::text, s.licence, v.embedding_model
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        JOIN index_versions v ON v.id = c.index_version_id
        WHERE c.index_version_id = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT 1
        """,
        (version, "[0.9, 0.1, 0]"),
    ).fetchone()

    assert row == ("Wash hands before handling food.", "[1,0,0]", "CC BY 4.0", "test-model")


def test_the_keyword_column_is_filled_in_automatically(db: Conn) -> None:
    apply(db)
    version = add_source_and_version(db)
    add_chunk(db, version, 0, "Wash hands before handling food.", "[1, 0, 0]")

    # 'handling' is stemmed, so a search for 'handle' still matches.
    row = db.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('english', 'handle')"
    ).fetchone()

    assert row == (1,)


def test_every_table_in_the_rag_schema_has_row_level_security(db: Conn) -> None:
    # A table added later without RLS would be one config change away from
    # public, so this covers every table, not a fixed list.
    apply(db)

    rows = db.execute(
        """
        SELECT c.relname
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'rag' AND c.relkind = 'r' AND NOT c.relrowsecurity
        """
    ).fetchall()

    assert rows == []


def test_only_one_index_version_can_be_current(db: Conn) -> None:
    apply(db)
    add_source_and_version(db, status="current")

    with pytest.raises(psycopg.errors.UniqueViolation):
        add_source_and_version(db, status="current")


def test_a_chunk_cannot_be_stored_twice_in_one_version(db: Conn) -> None:
    apply(db)
    version = add_source_and_version(db)
    add_chunk(db, version, 0, "first", "[1, 0, 0]")

    with pytest.raises(psycopg.errors.UniqueViolation):
        add_chunk(db, version, 0, "duplicate", "[0, 1, 0]")
