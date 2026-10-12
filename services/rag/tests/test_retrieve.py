"""Dense retrieval against Postgres, with hand-made vectors whose order is known."""

import json
import logging
import math
from collections.abc import Sequence

import pytest

from rag.db import vector_literal
from rag.db.migrate import apply
from rag.embed import EmbedResult, InputType
from rag.retrieve import (
    Hit,
    RetrievalError,
    current_version,
    get_version,
    retrieve,
    search,
)
from rag.search import render
from tests.conftest import Conn

SOURCE_ID = "bccampus-foodsafety"
MODEL = "fake-model"

# 3-dimension unit vectors. Against the question [1, 0, 0] their cosine
# similarities are exactly 1, 0.8, 0 and -1, so the right order is known.
SAME = [1.0, 0.0, 0.0]
CLOSE = [0.8, 0.6, 0.0]
UNRELATED = [0.0, 1.0, 0.0]
OPPOSITE = [-1.0, 0.0, 0.0]
QUESTION = [1.0, 0.0, 0.0]


class FakeEmbedder:
    """Returns one fixed vector, and records how it was asked."""

    dimension = 3

    def __init__(self, vector: list[float] = QUESTION, model: str = MODEL) -> None:
        self.vector = vector
        self.model = model
        self.calls: list[tuple[list[str], InputType]] = []

    def embed(self, texts: Sequence[str], input_type: InputType) -> EmbedResult:
        self.calls.append((list(texts), input_type))
        return EmbedResult(vectors=[self.vector for _ in texts], tokens=7, seconds=0.0, calls=1)


def add_version(conn: Conn, status: str, model: str = MODEL) -> int:
    row = conn.execute(
        """
        INSERT INTO index_versions (chunker, chunker_version, embedding_model,
                                    embedding_dim, status, chunk_count)
        VALUES ('heading_aware', '1', %s, 3, %s, 0) RETURNING id
        """,
        (model, status),
    ).fetchone()
    assert row is not None
    return int(row[0])


def add_chunk(conn: Conn, version_id: int, ordinal: int, name: str, vector: list[float]) -> int:
    heading = f"Food Safety > {name}"
    row = conn.execute(
        """
        INSERT INTO chunks (index_version_id, source_id, ordinal, section_ordinal, chapter,
                            section, heading_path, url, attribution, content, token_count,
                            embedding)
        VALUES (%s, %s, %s, %s, 'Food Safety', %s, %s, %s, 'Licensed under CC BY 4.0',
                %s, 10, %s::vector)
        RETURNING id
        """,
        (version_id, SOURCE_ID, ordinal, ordinal, name, heading,
         f"https://example.test/{name}", f"{heading}\n\nAbout {name}.", vector_literal(vector)),
    ).fetchone()  # fmt: skip
    assert row is not None
    return int(row[0])


@pytest.fixture
def index(db: Conn) -> tuple[Conn, int]:
    """A current version holding four chunks, inserted out of order on purpose."""
    apply(db)
    db.execute(
        """
        INSERT INTO sources (id, kind, title, url, licence, authority, trusted, retrieved_on)
        VALUES (%s, 'bccampus', 'Food Safety', 'https://example.test/', 'CC BY 4.0',
                'high', true, '2026-10-09')
        """,
        (SOURCE_ID,),
    )
    version = add_version(db, "current")
    for ordinal, (name, vector) in enumerate(
        [("unrelated", UNRELATED), ("opposite", OPPOSITE), ("same", SAME), ("close", CLOSE)]
    ):
        add_chunk(db, version, ordinal, name, vector)
    return db, version


def names(hits: list[Hit]) -> list[str | None]:
    return [h.section for h in hits]


def test_search_returns_the_closest_chunks_first(index: tuple[Conn, int]) -> None:
    conn, _ = index
    hits = search(conn, QUESTION, current_version(conn), k=4)

    assert names(hits) == ["same", "close", "unrelated", "opposite"]
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_scores_are_similarity_where_higher_is_better(index: tuple[Conn, int]) -> None:
    """pgvector returns distance (lower is better); the hits must say similarity."""
    conn, _ = index
    hits = search(conn, QUESTION, current_version(conn), k=4)

    expected = [1.0, 0.8, 0.0, -1.0]
    for hit, want in zip(hits, expected, strict=True):
        assert math.isclose(hit.similarity, want, abs_tol=1e-6)


def test_k_limits_the_results(index: tuple[Conn, int]) -> None:
    conn, _ = index
    assert names(search(conn, QUESTION, current_version(conn), k=2)) == ["same", "close"]


def test_only_the_requested_version_is_searched(index: tuple[Conn, int]) -> None:
    conn, _ = index
    retired = add_version(conn, "retired")
    add_chunk(conn, retired, 0, "old-perfect-match", SAME)

    current_hits = retrieve(conn, FakeEmbedder(), "q", k=5).hits
    old_hits = retrieve(conn, FakeEmbedder(), "q", k=5, version_id=retired).hits

    assert "old-perfect-match" not in names(current_hits)
    assert len(current_hits) == 4  # every chunk of the current version, none of the old one
    assert names(old_hits) == ["old-perfect-match"]


def test_equal_scores_always_come_back_in_the_same_order(index: tuple[Conn, int]) -> None:
    conn, version = index
    first = add_chunk(conn, version, 10, "twin-a", SAME)
    second = add_chunk(conn, version, 11, "twin-b", SAME)

    hits = search(conn, QUESTION, current_version(conn), k=3)

    # Three chunks score exactly 1.0; the lowest id wins each tie.
    assert [h.chunk_id for h in hits][1:] == [first, second]


def test_the_question_is_embedded_as_a_query(index: tuple[Conn, int]) -> None:
    conn, _ = index
    embedder = FakeEmbedder()

    retrieve(conn, embedder, "  how long can rice sit out?  ")

    assert embedder.calls == [(["how long can rice sit out?"], "query")]


def test_a_question_embedded_by_another_model_is_refused_before_paying(
    index: tuple[Conn, int],
) -> None:
    conn, _ = index
    embedder = FakeEmbedder(model="some-other-model")

    with pytest.raises(RetrievalError, match="different space"):
        retrieve(conn, embedder, "q")
    assert embedder.calls == []


def test_a_vector_of_the_wrong_size_is_refused(index: tuple[Conn, int]) -> None:
    conn, _ = index
    with pytest.raises(RetrievalError, match="2 dimensions"):
        search(conn, [1.0, 0.0], current_version(conn), k=3)


def test_no_current_version_says_how_to_make_one(db: Conn) -> None:
    apply(db)
    with pytest.raises(RetrievalError, match="--make-current"):
        retrieve(db, FakeEmbedder(), "q")


def test_a_version_still_building_cannot_be_searched(index: tuple[Conn, int]) -> None:
    conn, _ = index
    building = add_version(conn, "building")
    with pytest.raises(RetrievalError, match="still building"):
        get_version(conn, building)


@pytest.mark.parametrize(("question", "k"), [("", 5), ("   ", 5), ("q", 0), ("q", 51)])
def test_bad_arguments_fail_before_any_work(index: tuple[Conn, int], question: str, k: int) -> None:
    conn, _ = index
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        retrieve(conn, embedder, question, k)
    assert embedder.calls == []


def test_every_retrieval_logs_one_trace_line(
    index: tuple[Conn, int], caplog: pytest.LogCaptureFixture
) -> None:
    conn, version = index
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        result = retrieve(conn, FakeEmbedder(), "what is the danger zone?", k=2)

    assert len(caplog.records) == 1
    logged = json.loads(caplog.records[0].getMessage())
    assert logged["event"] == "retrieval"
    assert logged["question"] == "what is the danger zone?"
    assert logged["k"] == 2
    assert logged["index_version_id"] == version
    assert logged["embedding_model"] == MODEL
    assert logged["query_tokens"] == 7
    assert logged["chunk_ids"] == [h.chunk_id for h in result.hits]
    assert logged["similarities"] == [1.0, 0.8]
    assert logged["total_ms"] >= logged["embed_ms"] + logged["search_ms"] - 0.2


def test_the_body_drops_the_heading_path_and_the_cli_shows_it(index: tuple[Conn, int]) -> None:
    conn, _ = index
    result = retrieve(conn, FakeEmbedder(), "q", k=1)
    hit = result.hits[0]

    assert hit.content == "Food Safety > same\n\nAbout same."
    assert hit.body == "About same."
    printed = "\n".join(render(result, full=False))
    assert "1.000  Food Safety > same" in printed
    assert "About same." in printed
    assert "https://example.test/same" in printed
