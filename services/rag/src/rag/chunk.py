"""Turn sections into chunks: the passages retrieval returns.

A chunk is what gets embedded, searched, and finally shown to the model as
evidence. Its size is the main trade-off in the whole pipeline:

* Too big, and one vector has to stand for several ideas, so it matches none
  of them sharply. The model also gets more irrelevant text per citation.
* Too small, and a chunk loses the context that made it make sense ("Reheat
  to 74°C" without saying it is about leftovers), and answers need more chunks.

HEADING-AWARE CHUNKING (this module's first strategy)
Sections already follow the authors' own structure, so the default is one
chunk per section. Only a section longer than ``max_tokens`` is split, at
paragraph boundaries first, then list items or table rows, then sentences,
and only as a last resort mid-sentence. The pieces are balanced: a 900-token
section with a 400 limit becomes three pieces of about 300, not 400+400+100,
because a 100-token tail is a weak chunk.

Every chunk's text starts with its heading path, for example
"Food Safety ... > Preventing Foodborne Illness > The Danger Zone". The
embedding then knows what the passage is about even when the passage itself
never says so, and keyword search can match the heading's words.

Sizes are ESTIMATED from word counts (``tokens_per_word``), because chunks
must be sized before any API call. The indexer records the tokens Voyage
actually billed, so the estimate can be checked against reality.

The fixed-size strategy with overlap arrives in COO-13, as the comparison arm.
"""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SectionText:
    """One row of rag.sections, as the chunker needs it."""

    source_id: str
    ordinal: int
    chapter: str
    section: str | None
    heading_path: str
    url: str
    content: str


@dataclass(frozen=True)
class Chunk:
    source_id: str
    section_ordinal: int
    part: int  # 1-based
    parts: int  # how many pieces the section was split into
    chapter: str
    section: str | None
    heading_path: str
    url: str
    text: str  # exactly what is embedded and stored: heading path, blank line, body
    token_estimate: int


class Strategy(Protocol):
    """Anything that turns sections into chunks. COO-13 adds a second one."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def params(self) -> dict[str, Any]: ...

    def chunk(self, sections: Sequence[SectionText]) -> list[Chunk]: ...


@dataclass(frozen=True)
class HeadingAware:
    max_tokens: int = 400
    # English prose runs at roughly 1.3 tokens per word for modern tokenizers.
    # Checked against Voyage's billed totals at every build.
    tokens_per_word: float = 1.3

    @property
    def name(self) -> str:
        return "heading_aware"

    @property
    def version(self) -> str:
        # Bump when the splitting rules change, so old and new builds are
        # never mistaken for the same chunker.
        return "1"

    def params(self) -> dict[str, Any]:
        return asdict(self)

    def estimate(self, text: str) -> int:
        return math.ceil(len(text.split()) * self.tokens_per_word)

    def chunk(self, sections: Sequence[SectionText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections:
            header = section.heading_path
            # The heading path is repeated in every piece, so the body gets
            # whatever is left of the budget. Never less than a quarter of it,
            # even under an absurdly long heading.
            budget = max(self.max_tokens - self.estimate(header), self.max_tokens // 4)
            bodies = self._split(section.content, budget)
            for part, body in enumerate(bodies, start=1):
                text = f"{header}\n\n{body}"
                chunks.append(
                    Chunk(
                        source_id=section.source_id,
                        section_ordinal=section.ordinal,
                        part=part,
                        parts=len(bodies),
                        chapter=section.chapter,
                        section=section.section,
                        heading_path=section.heading_path,
                        url=section.url,
                        text=text,
                        token_estimate=self.estimate(text),
                    )
                )
        return chunks

    # ---- splitting ------------------------------------------------------------

    def _split(self, content: str, budget: int) -> list[str]:
        if self.estimate(content) <= budget:
            return [content]
        # Break oversized blocks into pieces that each fit, then pack the
        # pieces back together towards an even target size.
        pieces: list[str] = []
        for block in content.split("\n\n"):
            pieces.extend(self._fit(block, budget))
        return self._pack(pieces, budget)

    def _fit(self, block: str, budget: int) -> list[str]:
        """Split one paragraph, list or table until every piece fits."""
        if self.estimate(block) <= budget:
            return [block]
        lines = block.split("\n")
        if _is_table(lines):
            return self._split_table(lines, budget)
        if len(lines) > 1:  # a list: split between items
            return self._pack(
                [piece for line in lines for piece in self._fit(line, budget)], budget, sep="\n"
            )
        sentences = re.split(r"(?<=[.!?])\s+", block)
        if len(sentences) > 1:
            return self._pack(
                [piece for s in sentences for piece in self._fit(s, budget)], budget, sep=" "
            )
        # One enormous sentence: cut by words. Rare, and ugly, but bounded.
        words = block.split()
        per_piece = max(1, int(budget / self.tokens_per_word))
        return [" ".join(words[i : i + per_piece]) for i in range(0, len(words), per_piece)]

    def _split_table(self, lines: list[str], budget: int) -> list[str]:
        """Split a Markdown table by rows, repeating caption and header in each piece.

        A row on its own is meaningless without its column names, so every
        piece carries them.
        """
        prefix = [line for line in lines[:1] if line.startswith("Table:")]
        rest = lines[len(prefix) :]
        head, rows = rest[:2], rest[2:]  # header row and the |---| line
        repeated = "\n".join([*prefix, *head])
        room = max(budget - self.estimate(repeated), 1)
        groups = self._pack(rows, room, sep="\n")
        return [f"{repeated}\n{group}" for group in groups]

    def _pack(self, pieces: list[str], budget: int, sep: str = "\n\n") -> list[str]:
        """Join pieces in order into groups of roughly equal size, none over budget."""
        total = sum(self.estimate(p) for p in pieces)
        groups_needed = max(1, math.ceil(total / budget))
        target = total / groups_needed
        groups: list[list[str]] = [[]]
        size = 0
        for piece in pieces:
            cost = self.estimate(piece)
            over_target = size + cost > target and size > 0
            over_budget = size + cost > budget
            if groups[-1] and (over_budget or (over_target and len(groups) < groups_needed)):
                groups.append([])
                size = 0
            groups[-1].append(piece)
            size += cost
        return [sep.join(group) for group in groups if group]


def _is_table(lines: list[str]) -> bool:
    body = lines[1:] if lines and lines[0].startswith("Table:") else lines
    return len(body) >= 2 and body[0].startswith("|") and body[1].startswith("|")


def chunks_hash(chunks: Sequence[Chunk], params: dict[str, Any]) -> str:
    """Identifies a chunking result: same settings and same texts, same hash."""
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
    for c in chunks:
        digest.update(f"{c.source_id}\x1f{c.section_ordinal}\x1f{c.part}\x1f{c.text}\x1e".encode())
    return digest.hexdigest()
