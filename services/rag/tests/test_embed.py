"""The Voyage client, against a fake network: no key, no cost, no flakiness."""

import json
from typing import Any

import httpx2 as httpx
import pytest

from rag.embed import (
    NO_LIMIT,
    EmbeddingError,
    EmbedResult,
    RateLimit,
    VoyageEmbedder,
    build_request,
    parse_response,
)

DIM = 1024


def voyage_reply(request: httpx.Request, *, reverse: bool = False) -> httpx.Response:
    """Answer like Voyage: one vector per input, tokens = total words."""
    body = json.loads(request.content)
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i)] * DIM}
        for i in range(len(body["input"]))
    ]
    if reverse:
        data.reverse()
    tokens = sum(len(text.split()) for text in body["input"])
    return httpx.Response(
        200, json={"object": "list", "data": data, "model": body["model"],
                   "usage": {"total_tokens": tokens}},
    )  # fmt: skip


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def embedder(handler: Any, **kwargs: Any) -> VoyageEmbedder:
    return VoyageEmbedder("pa-test", client=client(handler), sleep=lambda _: None, **kwargs)


def test_the_request_asks_for_no_truncation_and_the_right_input_type() -> None:
    body = build_request(["a", "b"], "voyage-4-lite", "query")

    assert body == {
        "input": ["a", "b"],
        "model": "voyage-4-lite",
        "input_type": "query",
        "truncation": False,
    }


def test_texts_are_sent_in_batches_and_come_back_in_order() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["input"]))
        return voyage_reply(request)

    result = embedder(handler, batch_size=4).embed([f"text {i}" for i in range(10)], "document")

    assert sizes == [4, 4, 2]
    assert result.calls == 3
    assert len(result.vectors) == 10
    assert result.tokens == 20  # 2 words each


def test_vectors_are_matched_to_inputs_by_index_not_by_position() -> None:
    result = embedder(lambda r: voyage_reply(r, reverse=True)).embed(
        ["a", "b", "c", "d"], "document"
    )

    assert [v[0] for v in result.vectors] == [0.0, 1.0, 2.0, 3.0]


def test_the_api_key_is_sent_as_a_bearer_token() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return voyage_reply(request)

    embedder(handler).embed(["a"], "query")

    assert seen == ["Bearer pa-test"]


def test_rate_limits_are_retried_on_a_per_minute_timescale() -> None:
    # A 1-second backoff against a per-minute limit is just refused again.
    attempts: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, json={"detail": "slow down"})
        return voyage_reply(request)

    voyage = VoyageEmbedder("pa-test", client=client(handler), sleep=waits.append)
    result = voyage.embed(["a"], "document")

    assert len(attempts) == 3
    assert waits == [15.0, 30.0]
    assert len(result.vectors) == 1


def test_outages_are_retried_with_short_backoff() -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="unavailable")
        return voyage_reply(request)

    voyage = VoyageEmbedder("pa-test", client=client(handler), sleep=waits.append)
    voyage.embed(["a"], "document")

    assert waits == [1, 2]


def test_retries_count_against_the_requests_per_minute_limit() -> None:
    clock = FakeClock()
    sent_at: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_at.append(clock.now)
        if len(sent_at) == 1:
            return httpx.Response(503, text="unavailable")
        return voyage_reply(request)

    # One request a minute: the retry must wait for the window, not just 1s.
    paced_embedder(handler, RateLimit(requests_per_minute=1), clock).embed(["a"], "document")

    assert sent_at[0] == 0.0
    assert sent_at[1] >= 60.0


def test_a_bad_request_is_not_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"detail": "input too long"})

    with pytest.raises(EmbeddingError, match=r"400.*input too long"):
        embedder(handler).embed(["a"], "document")
    assert len(attempts) == 1


def test_it_gives_up_after_the_last_attempt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(EmbeddingError, match="503"):
        embedder(handler, max_attempts=3).embed(["a"], "document")


def test_a_network_failure_is_retried_then_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(EmbeddingError, match="unreachable"):
        embedder(handler, max_attempts=2).embed(["a"], "document")


def test_a_response_with_the_wrong_number_of_vectors_is_refused() -> None:
    payload = {"data": [{"index": 0, "embedding": [0.0] * DIM}], "usage": {"total_tokens": 1}}

    with pytest.raises(EmbeddingError, match="expected 2"):
        parse_response(payload, expected=2, dimension=DIM)


def test_a_vector_of_the_wrong_size_is_refused() -> None:
    payload = {"data": [{"index": 0, "embedding": [0.0] * 512}], "usage": {"total_tokens": 1}}

    with pytest.raises(EmbeddingError, match="1024-dimension"):
        parse_response(payload, expected=1, dimension=DIM)


class FakeClock:
    """Time that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def paced_embedder(handler: Any, limit: RateLimit, clock: FakeClock, **kw: Any) -> VoyageEmbedder:
    return VoyageEmbedder(
        "pa-test", client=client(handler), rate_limit=limit, sleep=clock.sleep, clock=clock, **kw
    )


def test_batches_are_sized_by_tokens_as_well_as_count() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["input"]))
        return voyage_reply(request)

    # 10 texts of 100 words (~150 estimated tokens each), at most ~450 per batch.
    texts = [" ".join(["word"] * 100)] * 10
    VoyageEmbedder("pa-test", client=client(handler), max_batch_tokens=450).embed(texts, "document")

    assert sizes == [3, 3, 3, 1]


def test_requests_are_paced_to_the_requests_per_minute_limit() -> None:
    clock = FakeClock()
    sent_at: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_at.append(clock.now)
        return voyage_reply(request)

    voyage = paced_embedder(handler, RateLimit(requests_per_minute=2), clock, batch_size=1)
    result = voyage.embed(["a", "b", "c"], "document")

    # Two go at once; the third waits for the first to leave the minute window.
    assert sent_at[:2] == [0.0, 0.0]
    assert sent_at[2] >= 60.0
    assert result.waited_seconds >= 60.0


def test_requests_are_paced_to_the_tokens_per_minute_limit() -> None:
    clock = FakeClock()
    sent_at: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_at.append(clock.now)
        return voyage_reply(request)

    # ~150 estimated tokens per text; 200 per minute fits one batch a minute.
    texts = [" ".join(["word"] * 100)] * 3
    voyage = paced_embedder(handler, RateLimit(tokens_per_minute=200), clock, batch_size=1)
    voyage.embed(texts, "document")

    assert sent_at[0] == 0.0
    assert sent_at[1] >= 60.0
    assert sent_at[2] >= 120.0


def test_without_limits_nothing_waits() -> None:
    clock = FakeClock()
    voyage = paced_embedder(voyage_reply, NO_LIMIT, clock, batch_size=1)

    result = voyage.embed(["a", "b", "c", "d"], "document")

    assert clock.sleeps == []
    assert result.waited_seconds == 0.0


def test_retry_after_from_the_server_is_honoured() -> None:
    clock = FakeClock()
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")
        return voyage_reply(request)

    paced_embedder(handler, NO_LIMIT, clock).embed(["a"], "document")

    assert clock.sleeps == [7.0]


def test_cost_is_tokens_times_the_listed_price() -> None:
    result = EmbedResult(vectors=[], tokens=1_000_000, seconds=0.0, calls=1)

    assert result.cost_usd("voyage-4-lite") == pytest.approx(0.02)
