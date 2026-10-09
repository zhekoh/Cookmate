-- 002: parsed sections, the step between a downloaded book and its chunks.
--
-- Ingest (COO-8) turns a source file into sections: one row per heading, with
-- the text under it, in reading order. Chunking (COO-9) then reads sections,
-- not the raw file. Keeping this layer means a chunking experiment re-chunks
-- from the database instead of re-downloading and re-parsing the book, and a
-- parser fix shows up as a diff in this table before it reaches retrieval.
--
-- Sections do not depend on any chunker or embedding model, so they have no
-- index_version. Chunks do not reference section ids either: re-ingesting a
-- book replaces its sections, and that must not delete chunks belonging to an
-- index version that is still being served. Chunks carry their own
-- heading_path and url instead.

CREATE TABLE rag.sections (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id     text NOT NULL REFERENCES rag.sources (id),
    ordinal       integer NOT NULL,   -- reading order within the source
    level         smallint NOT NULL CHECK (level BETWEEN 1 AND 6),  -- 1 = chapter title
    chapter       text NOT NULL,
    section       text,               -- NULL for text directly under the chapter title
    heading_path  text NOT NULL,      -- 'Book > Chapter > Section > Subsection'
    url           text NOT NULL,      -- deep link to the heading, for citations
    content       text NOT NULL CHECK (content <> ''),
    created_at    timestamptz NOT NULL DEFAULT now(),

    UNIQUE (source_id, ordinal)
);

ALTER TABLE rag.sections ENABLE ROW LEVEL SECURITY;
