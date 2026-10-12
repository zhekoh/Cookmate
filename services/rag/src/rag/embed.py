"""Turn text into vectors, through one seam that any provider can sit behind.

Everything else in the service calls ``Embedder.embed(texts, input_type)`` and
never talks to a vendor directly, so swapping Voyage for the local model in
COO-14 is a new class here and a config value, not a rewrite.

DOCUMENT VS QUERY
Voyage embeds the two sides differently. ``input_type="document"`` is used for
chunks and ``"query"`` for questions; the model prepends a different
instruction to each ("Represent the document for retrieval" / "Represent the
query for retrieving supporting documents"). Questions and passages are
different kinds of text, a short question versus a paragraph of answer, and
this asymmetry is trained to pull a question towards the passage that answers
it rather than towards other questions. Using the wrong one for either side
measurably lowers retrieval quality.

FAILING LOUDLY
``truncation`` is off. With it on, Voyage silently cuts a text that is too
long and embeds only the start, so a chunk could be stored, searched and
cited while its second half was never represented at all. Off, an oversized
text is an error at index time, which is where it should surface.

Vectors come back with length 1.0 (checked against the live API on
2026-10-09), so cosine similarity and dot product rank identically.
"""

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx2 as httpx

InputType = Literal["document", "query"]

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# USD per million tokens, from https://docs.voyageai.com/docs/pricing, checked
# 2026-10-09. Each account also gets its first 200M tokens free, which this
# figure ignores: it is the cost at the margin, which is what decisions need.
PRICE_PER_MILLION_TOKENS = {"voyage-4-lite": 0.02, "voyage-4": 0.06, "voyage-4-large": 0.12}

# Default output size for each model (1024 for the voyage-4 family).
DIMENSIONS = {"voyage-4-lite": 1024, "voyage-4": 1024, "voyage-4-large": 1024}

# Retry these: rate limiting and the server's own failures are usually brief.
# Never retry a 400: the request itself is wrong, and will be wrong again.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]  # same order as the input texts
    tokens: int  # as billed by the provider
    seconds: float  # wall clock, including any time spent waiting on rate limits
    calls: int
    waited_seconds: float = 0.0  # of `seconds`, how much was rate-limit pacing

    def cost_usd(self, model: str) -> float:
        return self.tokens * PRICE_PER_MILLION_TOKENS.get(model, 0.0) / 1_000_000


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str], input_type: InputType) -> EmbedResult: ...


# ---- request and response, as pure functions --------------------------------


def build_request(texts: Sequence[str], model: str, input_type: InputType) -> dict[str, Any]:
    return {
        "input": list(texts),
        "model": model,
        "input_type": input_type,
        "truncation": False,
    }


def parse_response(
    payload: dict[str, Any], expected: int, dimension: int
) -> tuple[list[list[float]], int]:
    """Vectors in input order, plus billed tokens. Refuses anything malformed."""
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        got = len(data) if isinstance(data, list) else "no"
        raise EmbeddingError(f"expected {expected} embeddings, got {got}")
    # Each item says which input it belongs to; sort rather than trust order.
    ordered = sorted(data, key=lambda item: item["index"])
    if [item["index"] for item in ordered] != list(range(expected)):
        raise EmbeddingError("embedding indexes do not match the inputs")
    vectors = [item["embedding"] for item in ordered]
    if any(len(v) != dimension for v in vectors):
        raise EmbeddingError(f"expected {dimension}-dimension vectors")
    tokens = int(payload.get("usage", {}).get("total_tokens", 0))
    return vectors, tokens


# ---- rate limits ---------------------------------------------------------------


@dataclass(frozen=True)
class RateLimit:
    """The provider's per-minute limits, respected on our side.

    Waiting before a request is cheaper than sending it, being refused with a
    429, and backing off: the refusal costs a round trip and says nothing
    about how long to wait unless the server bothers to send Retry-After.
    """

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None


# Voyage accounts with no payment method on file (seen in a 429 on 2026-10-09).
# Adding a card lifts these; the 200M free tokens still apply either way.
VOYAGE_FREE_TIER = RateLimit(requests_per_minute=3, tokens_per_minute=10_000)
NO_LIMIT = RateLimit()


def estimate_tokens(text: str) -> int:
    """Deliberately high (1.5 per word), so pacing errs on the early side."""
    return math.ceil(len(text.split()) * 1.5)


class _Pacer:
    """A sliding one-minute window of what has been sent."""

    WINDOW = 60.0

    def __init__(
        self, limit: RateLimit, clock: Callable[[], float], sleep: Callable[[float], None]
    ) -> None:
        self.limit = limit
        self.clock = clock
        self.sleep = sleep
        self.sent: deque[tuple[float, int]] = deque()
        self.waited = 0.0

    def wait_for(self, tokens: int) -> None:
        rpm, tpm = self.limit.requests_per_minute, self.limit.tokens_per_minute
        if rpm is None and tpm is None:
            return
        while True:
            now = self.clock()
            while self.sent and now - self.sent[0][0] >= self.WINDOW:
                self.sent.popleft()
            fits_requests = rpm is None or len(self.sent) < rpm
            fits_tokens = tpm is None or sum(t for _, t in self.sent) + tokens <= tpm
            if (fits_requests and fits_tokens) or not self.sent:
                self.sent.append((now, tokens))
                return
            # Wait until the oldest request leaves the window, then re-check.
            pause = self.sent[0][0] + self.WINDOW - now + 0.1
            self.waited += pause
            self.sleep(pause)


# ---- the Voyage client -------------------------------------------------------


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4-lite",
        *,
        rate_limit: RateLimit = NO_LIMIT,
        batch_size: int = 128,
        max_batch_tokens: int = 100_000,
        max_attempts: int = 4,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if model not in DIMENSIONS:
            raise ValueError(f"unknown Voyage model {model!r}")
        self._model = model
        self._batch_size = batch_size  # well under Voyage's 1,000-text limit
        # Well under the 1M-token request limit. With a per-minute token limit,
        # at most 40% of it per batch: Voyage's limiter counts a request as
        # bigger than its bill (on 2026-10-09 a batch billed at about 7,300
        # tokens was refused under a 10,000 limit), and we cannot see its
        # formula, so leave a wide margin rather than guess.
        self._max_batch_tokens = (
            min(max_batch_tokens, int(rate_limit.tokens_per_minute * 0.4))
            if rate_limit.tokens_per_minute
            else max_batch_tokens
        )
        self._max_attempts = max_attempts
        self._client = client or httpx.Client(timeout=60.0)
        self._sleep = sleep  # injectable, so tests never actually wait
        self._pacer = _Pacer(rate_limit, clock, sleep)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _batches(self, texts: Sequence[str]) -> list[tuple[int, int]]:
        """[start, end) ranges, each within the count and token limits."""
        ranges: list[tuple[int, int]] = []
        start, size = 0, 0
        for i, text in enumerate(texts):
            cost = estimate_tokens(text)
            if i > start and (
                i - start >= self._batch_size or size + cost > self._max_batch_tokens
            ):
                ranges.append((start, i))
                start, size = i, 0
            size += cost
        if start < len(texts):
            ranges.append((start, len(texts)))
        return ranges

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return DIMENSIONS[self._model]

    def embed(self, texts: Sequence[str], input_type: InputType) -> EmbedResult:
        started = time.perf_counter()
        waited_before = self._pacer.waited
        vectors: list[list[float]] = []
        tokens = calls = 0
        for start, end in self._batches(texts):
            batch = texts[start:end]
            payload = self._post(
                build_request(batch, self._model, input_type),
                estimated_tokens=sum(estimate_tokens(t) for t in batch),
            )
            calls += 1
            batch_vectors, batch_tokens = parse_response(payload, len(batch), self.dimension)
            vectors.extend(batch_vectors)
            tokens += batch_tokens
        return EmbedResult(
            vectors=vectors,
            tokens=tokens,
            seconds=time.perf_counter() - started,
            calls=calls,
            waited_seconds=self._pacer.waited - waited_before,
        )

    def _post(self, body: dict[str, Any], *, estimated_tokens: int) -> dict[str, Any]:
        for attempt in range(1, self._max_attempts + 1):
            # Every attempt is a request the provider counts, retries included.
            self._pacer.wait_for(estimated_tokens)
            try:
                response = self._client.post(VOYAGE_URL, json=body, headers=self._headers)
            except httpx.TransportError as exc:  # connection failures and timeouts
                if attempt == self._max_attempts:
                    raise EmbeddingError(f"Voyage unreachable: {exc}") from exc
            else:
                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    return result
                if response.status_code not in RETRYABLE_STATUS or attempt == self._max_attempts:
                    # The response body says what was wrong (e.g. a text over
                    # the token limit). It never contains the API key.
                    raise EmbeddingError(
                        f"Voyage returned {response.status_code}: {response.text[:300]}"
                    )
                retry_after = _retry_after(response)
                if retry_after is not None:
                    self._sleep(retry_after)  # the server said how long; believe it
                    continue
                if response.status_code == 429:
                    # Limits are per minute, so a 1-second backoff would just
                    # be refused again. 15s, 30s, 60s.
                    self._sleep(min(15.0 * 2 ** (attempt - 1), 60.0))
                    continue
            self._sleep(2 ** (attempt - 1))  # 1s, 2s, 4s for outages and timeouts
        raise EmbeddingError("Voyage request failed after every attempt")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return min(float(value), 120.0) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds; fall back to backoff
        return None
