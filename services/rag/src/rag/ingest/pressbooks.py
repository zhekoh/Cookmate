"""Parse a Pressbooks XHTML export into sections.

BCcampus books are published with Pressbooks, and its single-file XHTML export
keeps the structure we need: each front-matter page, part, chapter and
back-matter page is a ``div`` whose id names it (``chapter-<slug>``), and the
headings inside it are real ``h1``-``h6`` elements.

A *section* is one heading plus the text under it, up to the next heading.
Its ``heading_path`` records where it sits ("Book > Chapter > Section"), and
that path is what later lets a chunk be cited by book, chapter and section.

What is kept, and how:
* paragraphs, block quotes, captions: plain text, whitespace collapsed
* lists: one "- item" (or "1. item") line per item, nested items indented
* tables: Markdown, with the caption first as "Table: ...". Tables in these
  books are short reference data (temperatures, times), and Markdown keeps
  rows and columns readable to both the embedding model and the generator.

What is dropped: images, scripts, navigation, image credit blocks and
footnote lists. They are noise for retrieval. The licence of the book as a
whole is recorded on the source row instead.
"""

import copy
import hashlib
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PreformattedString, Tag

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TEXT_BLOCKS = {"p", "blockquote", "pre", "figcaption", "dt", "dd"}
LIST_TAGS = {"ul", "ol"}
DROP_TAGS = {"script", "style", "nav", "img", "svg", "picture", "video", "audio", "iframe"}
# Pressbooks puts image credits and footnotes in blocks with these classes.
DROP_CLASSES = {"media-attributions", "footnotes", "before-footnotes"}

UNIT_ID = re.compile(r"^(front-matter|part|chapter|back-matter)-(.+)$")


@dataclass(frozen=True)
class Section:
    ordinal: int  # reading order within the book, from 0
    level: int  # 1 = the chapter's own title, 2 = a heading under it, ...
    chapter: str
    section: str | None  # None for text directly under the chapter title
    heading_path: str
    url: str
    content: str


@dataclass
class ParseResult:
    sections: list[Section]
    units: int = 0  # front-matter, part, chapter and back-matter pages found
    tables: int = 0
    empty_headings: int = 0  # headings with no text before the next heading
    text_before_first_heading: int = 0  # blocks dropped because no heading owned them
    warnings: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """Changes whenever any section's place or text changes."""
        digest = hashlib.sha256()
        for s in self.sections:
            digest.update(f"{s.ordinal}\x1f{s.heading_path}\x1f{s.content}\x1e".encode())
        return digest.hexdigest()


def parse_book(html: str, *, book_title: str, base_url: str) -> ParseResult:
    soup = BeautifulSoup(html, "html.parser")
    # A <br> separates words; without this, "line one<br>line two" would
    # become "line oneline two".
    for br in soup.find_all("br"):
        br.replace_with(" ")

    result = ParseResult(sections=[])
    units = [
        tag
        for tag in soup.find_all(id=UNIT_ID)
        if isinstance(tag, Tag) and not _inside_another_unit(tag)
    ]
    if not units:
        result.warnings.append(
            "no Pressbooks page divs found; parsed the whole document as one page"
        )
        body = soup.body or soup
        units = [body]

    for unit in units:
        result.units += 1
        _Walker(unit, book_title, _unit_url(unit, base_url), result).run()
    return result


# ---- walking one page ---------------------------------------------------------


class _Walker:
    """Walks one page in reading order, cutting it into sections at headings."""

    def __init__(self, unit: Tag, book_title: str, page_url: str, result: ParseResult) -> None:
        self.unit = unit
        self.book_title = book_title
        self.page_url = page_url
        self.result = result
        # (tag level, heading text) for every open heading, outermost first.
        self.stack: list[tuple[int, str]] = []
        self.anchor: str | None = None
        self.blocks: list[str] = []

    def run(self) -> None:
        self._walk(self.unit)
        self._close_section()

    def _walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                # Comments, CDATA and the doctype are strings too; skip them.
                if not isinstance(child, PreformattedString):
                    self._add_block(clean(str(child)))
                continue
            if not isinstance(child, Tag) or _dropped(child):
                continue
            if child.name in HEADING_TAGS:
                self._open_heading(child)
            elif child.name in TEXT_BLOCKS:
                self._add_block(clean(child.get_text(" ")))
            elif child.name in LIST_TAGS:
                self._add_block("\n".join(render_list(child)))
            elif child.name == "table":
                self.result.tables += 1
                self._add_block(table_to_markdown(child))
            else:
                # A container (div, section, figure, span...). Look inside.
                self._walk(child)

    def _open_heading(self, heading: Tag) -> None:
        text = clean(heading.get_text(" "))
        if not text:
            return
        self._close_section()
        level = int(heading.name[1])
        # A heading closes every open heading at its own level or deeper.
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        heading_id = heading.get("id")
        self.anchor = heading_id if isinstance(heading_id, str) and len(self.stack) > 1 else None

    def _add_block(self, text: str) -> None:
        if not text:
            return
        if not self.stack:
            self.result.text_before_first_heading += 1
            return
        self.blocks.append(text)

    def _close_section(self) -> None:
        if not self.stack:
            return
        if not self.blocks:
            self.result.empty_headings += 1
            return
        headings = [text for _, text in self.stack]
        self.result.sections.append(
            Section(
                ordinal=len(self.result.sections),
                level=len(headings),
                chapter=headings[0],
                section=headings[-1] if len(headings) > 1 else None,
                heading_path=" > ".join([self.book_title, *headings]),
                url=self.page_url + (f"#{self.anchor}" if self.anchor else ""),
                content="\n\n".join(self.blocks),
            )
        )
        self.blocks = []


# ---- rendering ----------------------------------------------------------------


def clean(text: str) -> str:
    """Collapse whitespace, and undo the spaces get_text(" ") puts before punctuation."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


def render_list(tag: Tag, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for number, item in enumerate(tag.find_all("li", recursive=False), start=1):
        nested = item.find_all(LIST_TAGS)
        own = copy.copy(item)
        for sub in own.find_all(LIST_TAGS):
            sub.decompose()
        text = clean(own.get_text(" "))
        marker = f"{number}." if tag.name == "ol" else "-"
        if text:
            lines.append(f"{'  ' * depth}{marker} {text}")
        # Only the lists directly inside this item; deeper ones are handled by
        # the recursive call.
        for sub in nested:
            if sub.find_parent("li") is item:
                lines.extend(render_list(sub, depth + 1))
    return lines


def table_to_markdown(table: Tag) -> str:
    caption_tag = table.find("caption")
    caption = clean(caption_tag.get_text(" ")) if caption_tag else ""
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [
            clean(cell.get_text(" ")).replace("|", r"\|") for cell in row.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return f"Table: {caption}" if caption else ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "|" + " --- |" * width,
        *("| " + " | ".join(row) + " |" for row in rows[1:]),
    ]
    if caption:
        lines.insert(0, f"Table: {caption}")
    return "\n".join(lines)


# ---- helpers ------------------------------------------------------------------


def _dropped(tag: Tag) -> bool:
    if tag.name in DROP_TAGS:
        return True
    # bs4 returns class as a list of names; anything else means no classes.
    classes = tag.get("class")
    if not isinstance(classes, list):
        return False
    return any(name in DROP_CLASSES for name in classes)


def _inside_another_unit(tag: Tag) -> bool:
    return any(
        isinstance(parent.get("id"), str) and UNIT_ID.match(str(parent.get("id")))
        for parent in tag.parents
        if isinstance(parent, Tag)
    )


def _unit_url(unit: Tag, base_url: str) -> str:
    """The page's web address: id "chapter-x" lives at {base}chapter/x/."""
    unit_id = unit.get("id")
    match = UNIT_ID.match(unit_id) if isinstance(unit_id, str) else None
    if not match:
        return base_url
    kind, slug = match.groups()
    return f"{base_url}{kind}/{slug}/"
