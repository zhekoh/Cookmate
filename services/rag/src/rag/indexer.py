"""Build an index version: sections -> chunks -> vectors -> Postgres.

    uv run python -m rag.indexer build --dry-run         # chunk only: no API call, no writes
    uv run python -m rag.indexer build --make-current    # chunk, embed, store, serve it
    uv run python -m rag.indexer list                    # every version and its numbers

HOW A BUILD STAYS SAFE
1. Sections are read and chunked. Nothing is written yet.
2. A new index_versions row is created with status 'building'. Retrieval only
   ever reads the 'current' version, so this one is invisible to it.
3. Every chunk is embedded. This is the slow, paid step, and it runs with no
   database transaction open: holding one through minutes of network calls
   would pin a connection and its locks for nothing.
4. All chunks are written and the version is marked 'ready', in one
   transaction. A crash before this leaves a 'building' row with no chunks,
   which the next build deletes.
5. With --make-current, the old current version is retired and this one
   becomes current, in one transaction. A partial unique index guarantees
   there is never more than one current version.

Old versions are kept, not deleted. They are the baseline every experiment
compares against, and at this corpus size they cost almost nothing to keep.
"""

import argparse
import statistics
import sys
from collections import Counter
from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from rag.chunk import Chunk, HeadingAware, SectionText, Strategy, chunks_hash
from rag.config import ConfigError, get_database_settings, get_settings
from rag.db import connect
from rag.embed import NO_LIMIT, VOYAGE_FREE_TIER, Embedder, VoyageEmbedder
from rag.ingest.sources import attribution_for

Conn = psycopg.Connection[TupleRow]


@dataclass(frozen=True)
class BuildReport:
    version_id: int
    chunks: list[Chunk]
    embed_tokens: int
    embed_seconds: float  # on the API, excluding rate-limit pacing
    embed_waited_seconds: float
    embed_cost_usd: float
    embed_calls: int
    content_hash: str
    made_current: bool


def load_sections(conn: Conn) -> list[SectionText]:
    rows = conn.execute(
        """
        SELECT source_id, ordinal, chapter, section, heading_path, url, content
        FROM sections
        ORDER BY source_id, ordinal
        """
    ).fetchall()
    return [SectionText(*row) for row in rows]


def vector_literal(vector: list[float]) -> str:
    """pgvector's text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(x) for x in vector) + "]"


def build(conn: Conn, strategy: Strategy, embedder: Embedder, *, make_current: bool) -> BuildReport:
    """``conn`` must be autocommit, so each step below commits on its own."""
    if not conn.autocommit:
        raise ValueError("build() needs an autocommit connection")

    sections = load_sections(conn)
    if not sections:
        raise RuntimeError("no sections to index; run `python -m rag.ingest` first")
    chunks = strategy.chunk(sections)
    attributions = [
        attribution_for(c.source_id, c.section or c.chapter, c.url) for c in chunks
    ]  # fail now, before anything is paid for, if a source has no licence line
    params = strategy.params()
    content_hash = chunks_hash(chunks, params)

    # A crashed earlier build leaves a 'building' row; it was never served.
    conn.execute("DELETE FROM index_versions WHERE status = 'building'")
    row = conn.execute(
        """
        INSERT INTO index_versions (chunker, chunker_version, chunk_params,
                                    embedding_model, embedding_dim, status)
        VALUES (%s, %s, %s, %s, %s, 'building')
        RETURNING id
        """,
        (strategy.name, strategy.version, Jsonb(params), embedder.model, embedder.dimension),
    ).fetchone()
    if row is None:  # INSERT ... RETURNING always returns a row; guard anyway
        raise RuntimeError("could not create an index version")
    version_id = int(row[0])

    result = embedder.embed([c.text for c in chunks], "document")
    cost = result.cost_usd(embedder.model)
    api_seconds = result.seconds - result.waited_seconds

    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (index_version_id, source_id, ordinal, section_ordinal,
                                    part, parts, chapter, section, heading_path, url,
                                    attribution, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                [
                    (
                        version_id, c.source_id, ordinal, c.section_ordinal, c.part, c.parts,
                        c.chapter, c.section, c.heading_path, c.url, attribution, c.text,
                        c.token_estimate, vector_literal(vector),
                    )
                    for ordinal, (c, attribution, vector) in enumerate(
                        zip(chunks, attributions, result.vectors, strict=True)
                    )
                ],
            )  # fmt: skip
        conn.execute(
            """
            UPDATE index_versions
            SET status = 'ready', chunk_count = %s, content_hash = %s,
                embed_tokens = %s, embed_seconds = %s, embed_cost_usd = %s,
                completed_at = now()
            WHERE id = %s
            """,
            # Time spent on the API itself. Our own rate-limit pacing is left
            # out: it measures the account's tier, not the model or the build.
            (len(chunks), content_hash, result.tokens, round(api_seconds, 3), cost, version_id),
        )

    if make_current:
        promote(conn, version_id)
    return BuildReport(
        version_id=version_id,
        chunks=chunks,
        embed_tokens=result.tokens,
        embed_seconds=api_seconds,
        embed_waited_seconds=result.waited_seconds,
        embed_cost_usd=cost,
        embed_calls=result.calls,
        content_hash=content_hash,
        made_current=make_current,
    )


def promote(conn: Conn, version_id: int) -> None:
    """Make a ready version the one retrieval serves, retiring the old one."""
    with conn.transaction():
        status = conn.execute(
            "SELECT status FROM index_versions WHERE id = %s FOR UPDATE", (version_id,)
        ).fetchone()
        if status is None or status[0] not in ("ready", "current", "retired"):
            raise ValueError(f"version {version_id} is not ready to serve")
        conn.execute(
            "UPDATE index_versions SET status = 'retired' WHERE status = 'current' AND id <> %s",
            (version_id,),
        )
        conn.execute("UPDATE index_versions SET status = 'current' WHERE id = %s", (version_id,))


# ---- reporting -----------------------------------------------------------------


def describe_chunks(chunks: list[Chunk], max_tokens: int) -> list[str]:
    sizes = sorted(c.token_estimate for c in chunks)
    split_sections = {(c.source_id, c.section_ordinal) for c in chunks if c.parts > 1}
    parts = Counter(c.parts for c in chunks if c.part == 1)
    small = sum(1 for s in sizes if s < 60)
    return [
        f"chunks: {len(chunks)} from {len({(c.source_id, c.section_ordinal) for c in chunks})}"
        f" sections",
        f"estimated tokens per chunk: min {sizes[0]}, median {statistics.median(sizes):.0f}, "
        f"max {sizes[-1]} (limit {max_tokens}), total {sum(sizes)}",
        f"sections split: {len(split_sections)}  "
        + ", ".join(f"{n} into {p}" for p, n in sorted(parts.items()) if p > 1),
        f"small chunks (< 60 tokens): {small}",
    ]


def print_versions(conn: Conn) -> None:
    rows = conn.execute(
        """
        SELECT id, status, chunker, chunker_version, chunk_params, embedding_model,
               chunk_count, embed_tokens, embed_cost_usd, embed_seconds,
               left(content_hash, 12), completed_at
        FROM index_versions ORDER BY id
        """
    ).fetchall()
    if not rows:
        print("no index versions yet")
    for (vid, status, chunker, cversion, params, model, count, tokens, cost, seconds,
         digest, built) in rows:  # fmt: skip
        if built is None:
            print(f"v{vid:<3} {status:<8} (incomplete)")
            continue
        print(
            f"v{vid:<3} {status:<8} {chunker} v{cversion} {params}  {model}  chunks={count}  "
            f"tokens={tokens}  ${cost}  {seconds}s  hash={digest}  built={built:%Y-%m-%d %H:%M}"
        )


# ---- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.indexer")
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="chunk, embed and store a new index version")
    build_cmd.add_argument("--max-tokens", type=int, default=HeadingAware.max_tokens)
    build_cmd.add_argument("--make-current", action="store_true", help="serve it afterwards")
    build_cmd.add_argument("--dry-run", action="store_true", help="chunk only; no API, no writes")
    build_cmd.add_argument(
        "--free-tier",
        action="store_true",
        help="pace calls for a Voyage account with no payment method (3 req, 10K tokens/min)",
    )
    commands.add_parser("list", help="show every index version")
    args = parser.parse_args(argv)

    try:
        database_url = get_database_settings().database_url.get_secret_value()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    with connect(database_url, autocommit=True) as conn:
        if args.command == "list":
            print_versions(conn)
            return 0

        strategy = HeadingAware(max_tokens=args.max_tokens)
        if args.dry_run:
            chunks = strategy.chunk(load_sections(conn))
            for line in describe_chunks(chunks, strategy.max_tokens):
                print(line)
            print("dry run: nothing embedded, nothing written")
            return 0

        try:
            api_key = get_settings().voyage_api_key.get_secret_value()
        except ConfigError as exc:
            print(exc, file=sys.stderr)
            return 1
        limit = VOYAGE_FREE_TIER if args.free_tier else NO_LIMIT
        embedder = VoyageEmbedder(api_key, rate_limit=limit)
        report = build(conn, strategy, embedder, make_current=args.make_current)

    words = sum(len(c.text.split()) for c in report.chunks)
    estimated = sum(c.token_estimate for c in report.chunks)
    for line in describe_chunks(report.chunks, strategy.max_tokens):
        print(line)
    print(
        f"embedded: {report.embed_tokens} tokens billed in {report.embed_calls} call(s), "
        f"{report.embed_seconds:.1f}s on the API (+{report.embed_waited_seconds:.0f}s waiting "
        f"on rate limits), ${report.embed_cost_usd:.6f}"
    )
    print(
        f"token estimate check: estimated {estimated}, billed {report.embed_tokens} "
        f"({report.embed_tokens / words:.2f} tokens per word; the chunker assumes "
        f"{strategy.tokens_per_word})"
    )
    print(
        f"index version {report.version_id}: "
        f"{'current' if report.made_current else 'ready (not served)'}, "
        f"hash {report.content_hash[:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
