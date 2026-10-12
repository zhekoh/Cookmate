-- Rules that must hold for the current index version. Every problem count should be 0.
WITH cur AS (
    SELECT id, embedding_dim, chunk_count, (chunk_params->>'max_tokens')::int AS max_tokens
    FROM rag.index_versions WHERE status = 'current'
),
cc AS (
    SELECT c.* FROM rag.chunks c JOIN cur ON c.index_version_id = cur.id
)
SELECT 'exactly one current version' AS rule,
       abs((SELECT count(*) FROM rag.index_versions WHERE status = 'current') - 1) AS problems
UNION ALL
SELECT 'every vector has the version''s dimension',
       count(*) FROM cc, cur WHERE vector_dims(cc.embedding) <> cur.embedding_dim
UNION ALL
SELECT 'every vector has length 1',
       count(*) FROM cc WHERE abs(vector_norm(cc.embedding) - 1) > 0.001
UNION ALL
SELECT 'stored chunk count matches the version row',
       abs((SELECT count(*) FROM cc) - (SELECT chunk_count FROM cur))
UNION ALL
SELECT 'no chunk over the token limit',
       count(*) FROM cc, cur WHERE cc.token_count > cur.max_tokens
UNION ALL
SELECT 'every section has at least one chunk',
       count(*) FROM rag.sections s
       WHERE NOT EXISTS (SELECT 1 FROM cc WHERE cc.source_id = s.source_id
                                            AND cc.section_ordinal = s.ordinal)
UNION ALL
SELECT 'every split section has all its parts',
       count(*) FROM (SELECT source_id, section_ordinal, max(parts) AS p, count(*) AS n
                      FROM cc GROUP BY 1, 2) x WHERE p <> n
UNION ALL
SELECT 'every chunk has a url and a licence line',
       count(*) FROM cc WHERE cc.url = '' OR cc.attribution NOT LIKE '%Licensed under%'
UNION ALL
SELECT 'every chunk starts with its heading path',
       count(*) FROM cc WHERE left(cc.content, length(cc.heading_path)) <> cc.heading_path;
