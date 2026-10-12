"""Ask the index a question and see what comes back, before any metric exists.

    uv run python -m rag.search "how long can cooked rice sit out?"
    uv run python -m rag.search "why did my sauce split?" -k 10
    uv run python -m rag.search "what is the danger zone?" --full      # whole passages
    uv run python -m rag.search "what is the danger zone?" --version 2 # an older build

Results go to stdout. The trace line (version, timings, chunk ids, scores)
goes to stderr, the same line every retrieval logs, so `2>/dev/null` hides it
and `2>trace.jsonl` keeps it.

Retrieval always returns k chunks, however unrelated the question. Read the
similarity column before trusting a result: deciding when the best match is
too weak to answer from is the relevance gate's job (COO-12), and this tool is
how its threshold gets a first look.
"""

import argparse
import logging
import sys
import textwrap

from rag.config import ConfigError, get_database_settings, get_settings
from rag.db import connect
from rag.embed import VoyageEmbedder
from rag.retrieve import MAX_K, Retrieval, RetrievalError, current_version, get_version, retrieve


def render(retrieval: Retrieval, *, full: bool) -> list[str]:
    trace = retrieval.trace
    lines = [
        f"index version {trace.index_version_id} ({trace.embedding_model}), "
        f"top {trace.k} by cosine similarity, higher is closer",
        "",
    ]
    for hit in retrieval.hits:
        where = " > ".join(part for part in (hit.chapter, hit.section) if part)
        split = f"  [part {hit.part} of {hit.parts}]" if hit.parts > 1 else ""
        lines.append(f"{hit.rank:>2}  {hit.similarity:.3f}  {where}{split}")
        if full:
            lines.extend(textwrap.indent(hit.body, " " * 12).splitlines())
        else:
            first_line = hit.body.split("\n", 1)[0]
            lines.append(" " * 12 + textwrap.shorten(first_line, width=100, placeholder=" ..."))
        lines.append(f"{'':12}chunk {hit.chunk_id}  {hit.url}")
        lines.append("")
    lines.append(
        f"{trace.total_ms:.0f} ms total: {trace.embed_ms:.0f} ms embedding the question "
        f"({trace.query_tokens} tokens), {trace.search_ms:.0f} ms searching"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.search")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help=f"how many chunks, 1 to {MAX_K}")
    parser.add_argument("--version", type=int, help="search this index version, not the current")
    parser.add_argument("--full", action="store_true", help="print whole passages")
    args = parser.parse_args(argv)

    # The trace line, on stderr, apart from the results.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    # The HTTP client logs every request at INFO; keep stderr to trace lines only, so
    # `2>trace.jsonl` is valid JSON Lines.
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    try:
        database_url = get_database_settings().database_url.get_secret_value()
        api_key = get_settings().voyage_api_key.get_secret_value()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        with connect(database_url, autocommit=True) as conn:
            version = (
                current_version(conn) if args.version is None else get_version(conn, args.version)
            )
            # Embed with whatever model built the version being searched.
            embedder = VoyageEmbedder(api_key, model=version.embedding_model)
            retrieval = retrieve(conn, embedder, args.question, args.k, version_id=version.id)
    except (RetrievalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\n".join(render(retrieval, full=args.full)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
