-- 003: record exactly how each index version was built, and where each chunk
-- came from.
--
-- An index version is an experiment arm (COO-13, COO-14). To compare two arms
-- honestly, each must say precisely what produced it: the chunker settings,
-- what the embedding call actually billed, and how long it took. Without
-- these, "version 3 beat version 2" cannot be traced back to a cause.

ALTER TABLE rag.index_versions
    -- The chunker's settings, e.g. {"max_tokens": 400, "tokens_per_word": 1.3}.
    ADD COLUMN chunk_params    jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- Tokens Voyage reported billing for this build. chunks.token_count is
    -- the chunker's ESTIMATE (it sizes chunks before any API call); this is
    -- the measured total, so the two can be compared.
    ADD COLUMN embed_tokens    bigint,
    ADD COLUMN embed_seconds   numeric(10, 3);

ALTER TABLE rag.chunks
    -- Which section the chunk came from (sections.ordinal for the same
    -- source), and which piece of it: part 2 of 3 means the section was too
    -- long for one chunk and was split. Not a foreign key, for the reason in
    -- 002_sections.sql: re-ingesting replaces sections, and that must not
    -- break an index version still being served.
    ADD COLUMN section_ordinal integer,
    ADD COLUMN part            smallint NOT NULL DEFAULT 1 CHECK (part >= 1),
    ADD COLUMN parts           smallint NOT NULL DEFAULT 1 CHECK (parts >= 1);

-- chunks.token_count is an estimate; say so where anyone reading the schema
-- will see it.
COMMENT ON COLUMN rag.chunks.token_count IS
    'Estimated tokens (words x tokens_per_word from index_versions.chunk_params), '
    'used to size chunks before embedding. The billed total is index_versions.embed_tokens.';
