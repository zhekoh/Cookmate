"""Find the chunks closest in meaning to a question.

    retrieval = retrieve(conn, embedder, "how long can cooked rice sit out?", k=5)
    retrieval.hits    # best first, each with its similarity, text and citation metadata
    retrieval.trace   # what happened: version, timings, which chunks, which scores

This is dense retrieval only: the question becomes a vector and is compared
with every chunk's vector. Keyword search (COO-15) and re-ranking (COO-16)
arrive later as separate, measured additions on top of this.

SCORE DIRECTION
pgvector's ``<=>`` operator returns cosine DISTANCE: 0 means pointing the same
way, 2 means opposite, and lower is better. Everything outside the SQL uses
SIMILARITY = 1 - distance, where higher is better, because that is what people
expect a score to mean. Sorting by one while reading the other the wrong way
round silently returns the worst chunks first, so the conversion happens in
exactly one place, below.

EXACT SEARCH
There is no approximate index (see 001_chunks.sql), so the query compares the
question with every chunk in the version. The results are the true nearest
neighbours under this embedding model: if a relevant chunk is missing, the
cause is the embedding or the chunking, never the index. That keeps the eval
(COO-11) measuring one thing.

ONE MODEL PER COMPARISON
A question must be embedded by the same model that embedded the chunks.
Vectors from two models live in unrelated spaces, so comparing them gives
numbers that look like scores and mean nothing. It would not raise an error
either, when the dimensions happen to match, so ``retrieve`` checks the model
name against the index version before paying for the embedding.
"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import psycopg
from psycopg.rows import TupleRow

from rag.db import vector_literal
from rag.embed import Embedder

Conn = psycopg.Connection[TupleRow]

# One JSON line per retrieval goes to this logger. Later stages (re-rank,
# generate, support check) add their own fields to the same trace, and COO-34
# stores it in a table; until then it is a log line.
trace_log = logging.getLogger("rag.trace")

MAX_K = 50  # beyond this, the caller wants a re-ranker's candidate pool, not retrieval


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexVersion:
    id: int
    status: str
    embedding_model: str
    embedding_dim: int
    chunk_count: int | None


@dataclass(frozen=True)
class Hit:
    rank: int  # 1 = best
    chunk_id: int
    similarity: float  # 1 - cosine distance: 1 identical, 0 unrelated, higher is better
    source_id: str
    chapter: str | None
    section: str | None
    heading_path: str
    part: int
    parts: int
    url: str  # where the citation links to
    attribution: str  # the licence line shown beside it
    content: str  # exactly what was embedded: heading path, blank line, body
    token_count: int

    @property
    def body(self) -> str:
        """The passage without the heading path the chunker put in front of it."""
        prefix = f"{self.heading_path}\n\n"
        return self.content[len(prefix) :] if self.content.startswith(prefix) else self.content


@dataclass(frozen=True)
class RetrievalTrace:
    """Everything needed to explain one retrieval after the fact."""

    question: str
    k: int
    index_version_id: int
    embedding_model: str
    query_tokens: int  # as billed for embedding the question
    embed_ms: float  # the network call to the embedding provider
    search_ms: float  # the SQL query
    total_ms: float
    chunk_ids: tuple[int, ...]  # in rank order
    similarities: tuple[float, ...]  # same order

    def to_json(self) -> str:
        record = asdict(self)
        record["event"] = "retrieval"
        return json.dumps(record, ensure_ascii=False)


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    trace: RetrievalTrace


# ---- which index version to search ---------------------------------------------


def current_version(conn: Conn) -> IndexVersion:
    row = conn.execute(
        """
        SELECT id, status, embedding_model, embedding_dim, chunk_count
        FROM index_versions WHERE status = 'current'
        """
    ).fetchone()
    if row is None:
        raise RetrievalError(
            "no current index version; build one with `python -m rag.indexer build --make-current`"
        )
    return IndexVersion(*row)


def get_version(conn: Conn, version_id: int) -> IndexVersion:
    """Any finished version, current or not. Evals use this to compare arms."""
    row = conn.execute(
        """
        SELECT id, status, embedding_model, embedding_dim, chunk_count
        FROM index_versions WHERE id = %s
        """,
        (version_id,),
    ).fetchone()
    if row is None:
        raise RetrievalError(f"index version {version_id} does not exist")
    version = IndexVersion(*row)
    if version.status == "building":
        raise RetrievalError(f"index version {version_id} is still building")
    return version


# ---- the search itself -----------------------------------------------------------


def search(conn: Conn, vector: Sequence[float], version: IndexVersion, k: int) -> list[Hit]:
    """The k chunks of ``version`` closest to ``vector``, best first.

    Pure database work: no embedding, no network. Tests call this directly
    with hand-made vectors to check the ordering.
    """
    if len(vector) != version.embedding_dim:
        raise RetrievalError(
            f"query vector has {len(vector)} dimensions; "
            f"index version {version.id} has {version.embedding_dim}"
        )
    rows = conn.execute(
        """
        SELECT id, embedding <=> %(query)s::vector AS distance,
               source_id, chapter, section, heading_path, part, parts,
               url, attribution, content, token_count
        FROM chunks
        WHERE index_version_id = %(version)s
        -- Ties are broken by id, so the same question always returns the
        -- same list. Without it, two chunks with equal scores could swap
        -- places between runs and an eval would see noise that is not there.
        ORDER BY distance, id
        LIMIT %(k)s
        """,
        {"query": vector_literal(list(vector)), "version": version.id, "k": k},
    ).fetchall()
    return [
        Hit(
            rank=rank,
            chunk_id=int(chunk_id),
            similarity=1.0 - float(distance),  # the one place distance becomes similarity
            source_id=source_id,
            chapter=chapter,
            section=section,
            heading_path=heading_path,
            part=int(part),
            parts=int(parts),
            url=url,
            attribution=attribution,
            content=content,
            token_count=int(token_count),
        )
        for rank, (chunk_id, distance, source_id, chapter, section, heading_path, part, parts,
                   url, attribution, content, token_count) in enumerate(rows, start=1)
    ]  # fmt: skip


def retrieve(
    conn: Conn,
    embedder: Embedder,
    question: str,
    k: int = 5,
    *,
    version_id: int | None = None,
) -> Retrieval:
    """Embed ``question`` and return the ``k`` closest chunks, with a trace.

    Searches the current index version unless ``version_id`` names another.
    """
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")

    started = time.perf_counter()
    version = current_version(conn) if version_id is None else get_version(conn, version_id)
    if embedder.model != version.embedding_model:
        raise RetrievalError(
            f"index version {version.id} was embedded with {version.embedding_model}; "
            f"a question embedded with {embedder.model} would be compared in a different space"
        )

    embed_started = time.perf_counter()
    # "query", not "document": the model embeds the two sides differently,
    # trained so a question lands near the passage that answers it.
    embedded = embedder.embed([question], "query")
    search_started = time.perf_counter()
    hits = search(conn, embedded.vectors[0], version, k)
    finished = time.perf_counter()

    trace = RetrievalTrace(
        question=question,
        k=k,
        index_version_id=version.id,
        embedding_model=version.embedding_model,
        query_tokens=embedded.tokens,
        embed_ms=round((search_started - embed_started) * 1000, 1),
        search_ms=round((finished - search_started) * 1000, 1),
        total_ms=round((finished - started) * 1000, 1),
        chunk_ids=tuple(h.chunk_id for h in hits),
        similarities=tuple(round(h.similarity, 4) for h in hits),
    )
    trace_log.info(trace.to_json())
    return Retrieval(hits=hits, trace=trace)
