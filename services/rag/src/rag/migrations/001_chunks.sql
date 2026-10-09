-- 001: where retrieval data lives.
--
--   rag.sources         one row per document we ingest (a textbook, a dataset)
--   rag.index_versions  one row per build of the index (a chunker + an embedding model)
--   rag.chunks          the passages retrieval returns, each tied to a source and a build
--
-- WHY A SEPARATE `rag` SCHEMA, NOT `public`
-- Supabase publishes every table in `public` through its auto-generated REST
-- API, reachable with the anon key that ships inside the web app. A table
-- created there without row-level security is readable and writable by anyone
-- holding that key. `rag` is not an exposed schema, so these tables are
-- reachable only through a direct Postgres connection with the database
-- password. RLS is also switched on below, as a second lock on the same door.
--
-- The runner sets `search_path` to rag, extensions, public, so the `vector`
-- type and its operators resolve without being schema-qualified.

-- Supabase keeps extensions in their own schema. IF NOT EXISTS makes this a
-- no-op where pgvector is already enabled, wherever it was installed.
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;


-- ---------------------------------------------------------------------------
-- sources: licence and provenance, stored once per document and joined onto
-- every chunk, rather than copied into thousands of chunk rows.
-- ---------------------------------------------------------------------------
CREATE TABLE rag.sources (
    id            text PRIMARY KEY,  -- stable slug, e.g. 'bccampus-food-safety'
    kind          text NOT NULL CHECK (kind IN ('bccampus', 'wikibooks', 'usda')),
    title         text NOT NULL,
    url           text NOT NULL,
    licence       text NOT NULL,     -- e.g. 'CC BY 4.0'
    -- Ranking inputs for safety questions (COO-20): textbooks are 'high'.
    authority     text NOT NULL CHECK (authority IN ('high', 'low')),
    -- false = publicly editable (Wikibooks). Its text is data, never instructions.
    trusted       boolean NOT NULL,
    jurisdiction  text,              -- 'CA' for the BCcampus books; NULL if not applicable
    retrieved_on  date NOT NULL,
    snapshot      text,              -- dataset snapshot or release, when there is one
    created_at    timestamptz NOT NULL DEFAULT now()
);


-- ---------------------------------------------------------------------------
-- index_versions: one row per build. Changing the chunker or the embedding
-- model means re-embedding everything, so each build gets its own version and
-- is built beside the one being served. Retrieval reads only the 'current'
-- version; evals can read any 'ready' one, which is how two chunkers or two
-- embedding models are compared on the same questions.
-- ---------------------------------------------------------------------------
CREATE TABLE rag.index_versions (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunker          text NOT NULL,      -- 'heading_aware' | 'fixed_size'
    chunker_version  text NOT NULL,
    embedding_model  text NOT NULL,      -- 'voyage-4-lite' | 'BAAI/bge-small-en-v1.5'
    embedding_dim    integer NOT NULL CHECK (embedding_dim > 0),
    status           text NOT NULL DEFAULT 'building'
                     CHECK (status IN ('building', 'ready', 'current', 'retired')),
    chunk_count      integer,
    content_hash     text,               -- hash of every chunk's text, to spot accidental changes
    embed_cost_usd   numeric(12, 6),
    created_at       timestamptz NOT NULL DEFAULT now(),
    completed_at     timestamptz
);

-- At most one version is served at a time, enforced by the database rather
-- than by remembering to flip the old one first.
CREATE UNIQUE INDEX index_versions_one_current
    ON rag.index_versions ((true))
    WHERE status = 'current';


-- ---------------------------------------------------------------------------
-- chunks: what retrieval returns.
-- ---------------------------------------------------------------------------
CREATE TABLE rag.chunks (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id  bigint NOT NULL REFERENCES rag.index_versions (id) ON DELETE CASCADE,
    source_id         text   NOT NULL REFERENCES rag.sources (id),
    ordinal           integer NOT NULL,  -- position within the source, for stable ordering
    chapter           text,
    section           text,
    heading_path      text NOT NULL,     -- 'Book > Chapter > Section', also embedded in the text
    url               text NOT NULL,     -- deep link to the section, for the citation
    attribution       text NOT NULL,     -- the licence line shown beside the citation
    content           text NOT NULL,
    token_count       integer NOT NULL CHECK (token_count > 0),

    -- No fixed dimension, on purpose. The embedding experiment (COO-14)
    -- compares a 1024-dimension model with a 384-dimension one, so the column
    -- has to hold both. Vectors are only ever compared within one
    -- index_version, where the dimension is fixed (index_versions.embedding_dim).
    embedding         vector NOT NULL,

    -- Keyword search (COO-15). Postgres maintains it from `content`, so it can
    -- never drift out of sync. Its GIN index is added in COO-15, with the
    -- experiment that decides whether keyword search earns its place.
    tsv               tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    created_at        timestamptz NOT NULL DEFAULT now(),

    UNIQUE (index_version_id, source_id, ordinal)
);

-- The unique constraint above already indexes lookups by index_version_id.
-- This one serves "delete or re-ingest everything from one source".
CREATE INDEX chunks_source_id ON rag.chunks (source_id);

-- WHY THERE IS NO VECTOR INDEX
-- The corpus is a few thousand chunks. Exact search compares the question with
-- every chunk in the current version, which takes milliseconds at this size,
-- and it finds the true nearest neighbours every time. An approximate index
-- (HNSW, IVFFlat) trades some recall for speed this corpus does not need, and
-- it would add a tuning knob that can quietly lower the numbers the
-- experiments measure. If the corpus ever grows enough to need one, it would
-- be a per-version expression index, for example:
--   CREATE INDEX ON rag.chunks
--     USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
--     WHERE index_version_id = 42;


-- ---------------------------------------------------------------------------
-- Row-level security, with no policies: the anon and authenticated roles get
-- nothing. The service connects as the table owner, which RLS does not
-- restrict. This only matters if someone later exposes the `rag` schema
-- through Supabase's API by mistake.
-- ---------------------------------------------------------------------------
ALTER TABLE rag.sources        ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag.index_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag.chunks         ENABLE ROW LEVEL SECURITY;
