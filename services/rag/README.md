# Cookmate RAG service

Cited cooking answers from open-licensed sources. See `SOURCES.md` for what is
ingested and under which licence, and `RAG-DESIGN.md` at the repo root for the
design.

## Run it

```bash
cd services/rag
uv sync                                   # install, including dev tools
uv run pytest                             # tests (no network, no real keys)
uv run ruff check                         # lint
uv run mypy                               # strict type check
uv run uvicorn --factory rag.api:create_app --port 8000 --reload
curl localhost:8000/health
```

The service refuses to start unless `DATABASE_URL`, `ANTHROPIC_API_KEY`,
`VOYAGE_API_KEY` and `RAG_SERVICE_TOKEN` are set. It reads them from the
environment, or from a `.env` file in the directory you run it from.

With Docker, from the repo root:

```bash
docker compose up --build                 # API on :8787, RAG on :8000
```

## Database

Chunks, vectors and source metadata live in a `rag` schema in Postgres with
pgvector. The schema is plain SQL in `src/rag/migrations/`, applied in order
by a small runner that records each file it has run:

```bash
uv run python -m rag.db.migrate           # needs only DATABASE_URL
```

Running it again is a no-op. Never edit a migration that has been applied
anywhere; add a new numbered file instead. The runner refuses edited files.

The database tests need a throwaway Postgres. They drop the `rag` schema, so
they refuse to run against anything but a local host:

```bash
docker compose --profile test up -d rag-test-db      # from the repo root
export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres
uv run pytest
```

Without `TEST_DATABASE_URL` those tests are skipped locally. In CI they fail
instead, so the job cannot go green without running them.

## Ingest

Each source is downloaded by hand (see `SOURCES.md` for why), parsed into
sections, and written to `rag.sections`. Re-running replaces that source's
sections in one transaction, so it is always safe.

```bash
# save the export as data/raw/foodsafety.xhtml first; the command prints the link
uv run python -m rag.ingest bccampus --title foodsafety --dry-run   # parse and summarise only
uv run python -m rag.ingest bccampus --title foodsafety             # write to the database
```
