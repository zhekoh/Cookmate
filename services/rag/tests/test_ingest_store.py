"""Writing sections to Postgres: idempotent, atomic, and it cleans up after itself."""

from dataclasses import replace
from datetime import date

import psycopg
import pytest

from rag.db.migrate import apply
from rag.ingest.pressbooks import Section
from rag.ingest.store import SourceRow, write_source
from tests.conftest import Conn

SOURCE = SourceRow(
    id="bccampus-sample",
    kind="bccampus",
    title="Sample Book",
    url="https://opentextbc.ca/sample/",
    licence="CC BY 4.0",
    authority="high",
    trusted=True,
    jurisdiction="CA",
    retrieved_on=date(2026, 10, 9),
    snapshot="test",
)


def make_sections(count: int) -> list[Section]:
    return [
        Section(
            ordinal=i,
            level=2,
            chapter="Chapter",
            section=f"Section {i}",
            heading_path=f"Sample Book > Chapter > Section {i}",
            url=f"https://opentextbc.ca/sample/chapter/c/#s{i}",
            content=f"Text of section {i}.",
        )
        for i in range(count)
    ]


def stored(conn: Conn) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT ordinal, content FROM sections WHERE source_id = %s ORDER BY ordinal",
        (SOURCE.id,),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def test_writing_twice_gives_the_same_rows_not_twice_as_many(db: Conn) -> None:
    apply(db)
    sections = make_sections(3)

    first = write_source(db, SOURCE, sections)
    second = write_source(db, SOURCE, sections)

    assert (first.removed, first.inserted) == (0, 3)
    assert (second.removed, second.inserted) == (3, 3)
    assert stored(db) == [
        (0, "Text of section 0."),
        (1, "Text of section 1."),
        (2, "Text of section 2."),
    ]
    count = db.execute("SELECT count(*) FROM sources WHERE id = %s", (SOURCE.id,)).fetchone()
    assert count == (1,)


def test_a_shorter_re_ingest_leaves_no_stale_sections(db: Conn) -> None:
    apply(db)
    write_source(db, SOURCE, make_sections(5))

    write_source(db, SOURCE, make_sections(2))

    assert [ordinal for ordinal, _ in stored(db)] == [0, 1]


def test_re_ingest_updates_the_source_row(db: Conn) -> None:
    apply(db)
    write_source(db, SOURCE, make_sections(1))

    write_source(db, replace(SOURCE, snapshot="newer"), make_sections(1))

    row = db.execute("SELECT snapshot FROM sources WHERE id = %s", (SOURCE.id,)).fetchone()
    assert row == ("newer",)


def test_a_failed_write_keeps_the_previous_sections(db: Conn) -> None:
    apply(db)
    write_source(db, SOURCE, make_sections(3))
    broken = [*make_sections(2), replace(make_sections(1)[0], ordinal=5, content="")]

    # Empty content breaks a CHECK constraint on the last row, after the
    # delete and the first inserts have already run inside the transaction.
    with pytest.raises(psycopg.errors.CheckViolation):
        write_source(db, SOURCE, broken)

    assert len(stored(db)) == 3
