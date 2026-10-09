"""Write a parsed source to Postgres: one source row plus its sections.

Re-running is safe. The source row is upserted, and the source's sections are
deleted and re-inserted inside one transaction, so a reader sees either the
old sections or the new ones, never a mix, and a crash halfway leaves the old
ones in place. Delete-then-insert is simpler than diffing, and correct here
because nothing references section ids (see 002_sections.sql).
"""

from dataclasses import dataclass
from datetime import date

import psycopg
from psycopg.rows import TupleRow

from rag.ingest.pressbooks import Section


@dataclass(frozen=True)
class SourceRow:
    id: str
    kind: str  # 'bccampus' | 'wikibooks' | 'usda'
    title: str
    url: str
    licence: str
    authority: str  # 'high' | 'low'
    trusted: bool
    jurisdiction: str | None
    retrieved_on: date
    snapshot: str | None


@dataclass(frozen=True)
class WriteResult:
    removed: int
    inserted: int


def write_source(
    conn: psycopg.Connection[TupleRow], source: SourceRow, sections: list[Section]
) -> WriteResult:
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO sources (id, kind, title, url, licence, authority, trusted,
                                 jurisdiction, retrieved_on, snapshot)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                kind = EXCLUDED.kind, title = EXCLUDED.title, url = EXCLUDED.url,
                licence = EXCLUDED.licence, authority = EXCLUDED.authority,
                trusted = EXCLUDED.trusted, jurisdiction = EXCLUDED.jurisdiction,
                retrieved_on = EXCLUDED.retrieved_on, snapshot = EXCLUDED.snapshot
            """,
            (
                source.id,
                source.kind,
                source.title,
                source.url,
                source.licence,
                source.authority,
                source.trusted,
                source.jurisdiction,
                source.retrieved_on,
                source.snapshot,
            ),
        )
        removed = conn.execute("DELETE FROM sections WHERE source_id = %s", (source.id,)).rowcount

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO sections (source_id, ordinal, level, chapter, section,
                                      heading_path, url, content)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        source.id,
                        s.ordinal,
                        s.level,
                        s.chapter,
                        s.section,
                        s.heading_path,
                        s.url,
                        s.content,
                    )
                    for s in sections
                ],
            )
    return WriteResult(removed=removed, inserted=len(sections))
