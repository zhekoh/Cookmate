"""Building index versions end to end, against Postgres, with a fake embedder."""

from collections.abc import Sequence
from datetime import date

import pytest

from rag.chunk import HeadingAware
from rag.db.migrate import apply
from rag.embed import EmbeddingError, EmbedResult, InputType
from rag.indexer import build, promote
from rag.ingest.pressbooks import Section
from rag.ingest.store import SourceRow, write_source
from tests.conftest import Conn

SOURCE = SourceRow(
    id="bccampus-foodsafety",
    kind="bccampus",
    title="Food Safety, Sanitation, and Personal Hygiene",
    url="https://opentextbc.ca/foodsafety/",
    licence="CC BY 4.0",
    authority="high",
    trusted=True,
    jurisdiction="CA",
    retrieved_on=date(2026, 10, 9),
    snapshot="test",
)


class FakeEmbedder:
    """Deterministic 3-dimension vectors; counts calls so tests can see them."""

    model = "fake-model"
    dimension = 3

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embed(self, texts: Sequence[str], input_type: InputType) -> EmbedResult:
        self.calls += 1
        if self.fail:
            raise EmbeddingError("simulated outage")
        vectors = [[1.0, float(len(t) % 7), 0.5] for t in texts]
        return EmbedResult(vectors=vectors, tokens=10 * len(texts), seconds=0.01, calls=1)


@pytest.fixture
def corpus(db: Conn) -> Conn:
    apply(db)
    write_source(
        db,
        SOURCE,
        [
            Section(0, 1, "Preventing Foodborne Illness", None,
                    "Food Safety > Preventing Foodborne Illness",
                    "https://opentextbc.ca/foodsafety/chapter/p/", "Intro text."),
            Section(1, 2, "Preventing Foodborne Illness", "The Danger Zone",
                    "Food Safety > Preventing Foodborne Illness > The Danger Zone",
                    "https://opentextbc.ca/foodsafety/chapter/p/#dz",
                    "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(5))),
        ],
    )  # fmt: skip
    return db


def versions(conn: Conn) -> list[tuple[int, str]]:
    rows = conn.execute("SELECT id, status FROM index_versions ORDER BY id").fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def test_a_build_stores_every_chunk_with_its_vector_and_licence_line(corpus: Conn) -> None:
    report = build(corpus, HeadingAware(max_tokens=300), FakeEmbedder(), make_current=False)

    row = corpus.execute(
        """
        SELECT status, chunk_count, embed_tokens, chunk_params->>'max_tokens', content_hash
        FROM index_versions WHERE id = %s
        """,
        (report.version_id,),
    ).fetchone()
    assert row == ("ready", len(report.chunks), 10 * len(report.chunks), "300",
                   report.content_hash)  # fmt: skip

    chunk = corpus.execute(
        """
        SELECT attribution, vector_dims(embedding), parts, content
        FROM chunks WHERE index_version_id = %s AND section = 'The Danger Zone' AND part = 1
        """,
        (report.version_id,),
    ).fetchone()
    assert chunk is not None
    attribution, dims, parts, content = chunk
    assert attribution.startswith('"The Danger Zone", in Food Safety, Sanitation')
    assert "CC BY 4.0" in attribution
    assert dims == 3
    assert parts > 1  # the long section was split
    assert content.startswith("Food Safety > Preventing Foodborne Illness > The Danger Zone\n\n")


def test_a_new_build_is_not_served_until_promoted(corpus: Conn) -> None:
    first = build(corpus, HeadingAware(), FakeEmbedder(), make_current=True)
    second = build(corpus, HeadingAware(max_tokens=200), FakeEmbedder(), make_current=False)

    assert versions(corpus) == [(first.version_id, "current"), (second.version_id, "ready")]

    promote(corpus, second.version_id)

    assert versions(corpus) == [(first.version_id, "retired"), (second.version_id, "current")]


def test_a_failed_embedding_call_leaves_the_served_version_untouched(corpus: Conn) -> None:
    served = build(corpus, HeadingAware(), FakeEmbedder(), make_current=True)

    with pytest.raises(EmbeddingError):
        build(corpus, HeadingAware(), FakeEmbedder(fail=True), make_current=True)

    statuses = dict(versions(corpus))
    assert statuses[served.version_id] == "current"
    assert "building" in statuses.values()  # the failed build, never served

    # The next build clears the abandoned one.
    build(corpus, HeadingAware(), FakeEmbedder(), make_current=False)
    assert "building" not in dict(versions(corpus)).values()


def test_the_same_corpus_and_settings_give_the_same_hash(corpus: Conn) -> None:
    a = build(corpus, HeadingAware(), FakeEmbedder(), make_current=False)
    b = build(corpus, HeadingAware(), FakeEmbedder(), make_current=False)
    c = build(corpus, HeadingAware(max_tokens=200), FakeEmbedder(), make_current=False)

    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_nearest_neighbour_search_works_on_the_stored_vectors(corpus: Conn) -> None:
    report = build(corpus, HeadingAware(), FakeEmbedder(), make_current=True)

    row = corpus.execute(
        """
        SELECT count(*) FROM (
            SELECT 1 FROM chunks c JOIN index_versions v ON v.id = c.index_version_id
            WHERE v.status = 'current'
            ORDER BY c.embedding <=> '[1, 0, 0.5]'::vector
            LIMIT 3
        ) top
        """
    ).fetchone()
    assert row == (min(3, len(report.chunks)),)


def test_promoting_a_version_that_is_still_building_is_refused(corpus: Conn) -> None:
    corpus.execute(
        """
        INSERT INTO index_versions (chunker, chunker_version, embedding_model, embedding_dim)
        VALUES ('heading_aware', '1', 'fake-model', 3)
        """
    )
    building_id = versions(corpus)[-1][0]

    with pytest.raises(ValueError, match="not ready"):
        promote(corpus, building_id)
