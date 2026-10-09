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
