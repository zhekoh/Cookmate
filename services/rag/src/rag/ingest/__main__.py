"""Ingest a source into the sections table.

    uv run python -m rag.ingest bccampus --title foodsafety
    uv run python -m rag.ingest bccampus --title foodsafety --dry-run

Reads the book from data/raw/<title>.xhtml (or --file). The download is a
manual step: opentextbc.ca sits behind a bot check that blocks scripted
downloads, and getting past it would mean pretending to be a browser. One
click in a real browser on the export link is the honest route, and the
command below prints that link if the file is missing.

--dry-run parses and prints the summary without touching the database, which
is how to check a parser change before it reaches the sections table.
"""

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rag.config import ConfigError, get_database_settings
from rag.db import connect
from rag.ingest.pressbooks import ParseResult, parse_book
from rag.ingest.sources import BCCAMPUS_BOOKS, BCCAMPUS_LICENCE
from rag.ingest.store import SourceRow, write_source

RAW_DIR = Path("data/raw")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.ingest")
    kinds = parser.add_subparsers(dest="kind", required=True)
    bccampus = kinds.add_parser("bccampus", help="a BCcampus open textbook")
    bccampus.add_argument("--title", required=True, choices=sorted(BCCAMPUS_BOOKS))
    bccampus.add_argument("--file", type=Path, help="default: data/raw/<title>.xhtml")
    bccampus.add_argument("--dry-run", action="store_true", help="parse only, write nothing")
    args = parser.parse_args(argv)

    book = BCCAMPUS_BOOKS[args.title]
    path: Path = args.file or RAW_DIR / f"{book.slug}.xhtml"
    if not path.is_file():
        print(
            f"{path} not found.\n"
            f"Download it in a browser from:\n  {book.download_url}\n"
            f"and save it as {path} (run this command from services/rag).",
            file=sys.stderr,
        )
        return 1

    result = parse_book(
        path.read_text(encoding="utf-8"), book_title=book.title, base_url=book.base_url
    )
    print_summary(book.title, path, result)
    if not result.sections:
        print("nothing to write: no sections parsed", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    try:
        settings = get_database_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    # The file's own date stands in for the download date: the download is
    # manual, so this is the closest record of when the copy was taken.
    retrieved_on = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    source = SourceRow(
        id=book.source_id,
        kind="bccampus",
        title=book.title,
        url=book.base_url,
        licence=BCCAMPUS_LICENCE,
        authority="high",
        trusted=True,
        jurisdiction="CA",
        retrieved_on=retrieved_on,
        snapshot=f"xhtml export, content sha256 {result.content_hash[:16]}",
    )
    with connect(settings.database_url.get_secret_value(), autocommit=True) as conn:
        written = write_source(conn, source, result.sections)
    print(f"\nwrote {written.inserted} sections for {book.source_id} (replaced {written.removed})")
    return 0


def print_summary(title: str, path: Path, result: ParseResult) -> None:
    sizes = sorted(len(s.content.split()) for s in result.sections)
    levels = Counter(s.level for s in result.sections)
    print(f"{title}\n  file: {path}")
    print(
        f"  pages kept: {result.units}   sections: {len(result.sections)}   tables: {result.tables}"
    )
    print(
        "  sections by depth: "
        + ", ".join(f"level {level}: {count}" for level, count in sorted(levels.items()))
    )
    if sizes:
        print(
            f"  words per section: min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
            f"max {sizes[-1]}, total {sum(sizes)}"
        )
    print(
        f"  headings with no text: {result.empty_headings}   "
        f"text before any heading: {result.text_before_first_heading}"
    )
    print(f"  content hash: {result.content_hash[:16]}")
    if result.skipped_pages:
        print(f"  skipped {len(result.skipped_pages)} pages (about the book, not the subject):")
        for page in result.skipped_pages:
            print(f"    - {page}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


if __name__ == "__main__":
    sys.exit(main())
