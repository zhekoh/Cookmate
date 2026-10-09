"""Postgres connections.

Every connection gets the same ``search_path``, so SQL can say ``chunks`` and
``vector`` instead of ``rag.chunks`` and ``extensions.vector``. It is set with
a plain ``SET`` after connecting rather than as a connection option, because
Supabase's connection pooler does not pass every startup option through.
"""

import psycopg
from psycopg.rows import TupleRow


def connect(url: str, *, autocommit: bool = False) -> psycopg.Connection[TupleRow]:
    conn = psycopg.connect(url, autocommit=autocommit)
    try:
        conn.execute("SET search_path TO rag, extensions, public")
        if not autocommit:
            # SET ran inside the implicit transaction psycopg opened; commit it
            # so the setting survives the caller's first rollback.
            conn.commit()
    except BaseException:
        conn.close()
        raise
    return conn
