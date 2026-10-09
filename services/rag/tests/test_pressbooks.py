"""Parser tests against a saved page shaped like a Pressbooks XHTML export."""

import pytest

from rag.ingest.pressbooks import ParseResult, Section, parse_book, table_to_markdown
from tests.conftest import FIXTURES

BASE = "https://opentextbc.ca/sample/"


@pytest.fixture(scope="module")
def parsed() -> ParseResult:
    html = (FIXTURES / "pressbooks_sample.xhtml").read_text(encoding="utf-8")
    return parse_book(html, book_title="Sample Book", base_url=BASE)


def by_heading(result: ParseResult, heading: str) -> Section:
    matches = [s for s in result.sections if (s.section or s.chapter) == heading]
    assert len(matches) == 1, f"expected one section headed {heading!r}"
    return matches[0]


def test_sections_come_out_in_reading_order_with_their_heading_paths(
    parsed: ParseResult,
) -> None:
    assert [s.heading_path for s in parsed.sections] == [
        "Sample Book > Introduction",
        "Sample Book > Foodborne Illness",
        "Sample Book > Foodborne Illness > Causes of Foodborne Illness",
        "Sample Book > Foodborne Illness > Causes of Foodborne Illness > The Danger Zone",
        "Sample Book > Foodborne Illness > Prevention",
    ]
    assert [s.ordinal for s in parsed.sections] == list(range(len(parsed.sections)))


def test_depth_chapter_and_section_fields(parsed: ParseResult) -> None:
    chapter_text = by_heading(parsed, "Foodborne Illness")
    subsection = by_heading(parsed, "The Danger Zone")

    assert (chapter_text.level, chapter_text.chapter, chapter_text.section) == (
        1,
        "Foodborne Illness",
        None,
    )
    assert (subsection.level, subsection.chapter, subsection.section) == (
        3,
        "Foodborne Illness",
        "The Danger Zone",
    )


def test_urls_point_at_the_page_and_the_heading(parsed: ParseResult) -> None:
    assert by_heading(parsed, "Introduction").url == f"{BASE}front-matter/introduction/"
    assert by_heading(parsed, "Foodborne Illness").url == f"{BASE}chapter/foodborne-illness/"
    assert by_heading(parsed, "The Danger Zone").url == (
        f"{BASE}chapter/foodborne-illness/#danger-zone"
    )


def test_whitespace_and_line_breaks_are_normalised(parsed: ParseResult) -> None:
    assert by_heading(parsed, "Introduction").content == (
        "This book covers food safety for professional cooks."
    )
    danger = by_heading(parsed, "The Danger Zone").content
    assert danger.startswith("Keep food out of the danger zone. Cold food stays cold.")


def test_lists_keep_their_items_and_nesting(parsed: ParseResult) -> None:
    causes = by_heading(parsed, "Causes of Foodborne Illness").content
    assert "- Bacteria\n  - Salmonella\n  - Listeria\n- Viruses" in causes

    prevention = by_heading(parsed, "Prevention").content
    assert prevention == "1. Wash hands.\n2. Cook thoroughly."


def test_tables_become_markdown_with_the_caption_first(parsed: ParseResult) -> None:
    danger = by_heading(parsed, "The Danger Zone").content

    assert (
        "Table: Table 1.1 Safe temperatures\n"
        "| Food | Temperature |\n"
        "| --- | --- |\n"
        "| Poultry | 74°C \\| 165°F |\n"
        "| Ground beef | 71°C |"
    ) in danger
    assert parsed.tables == 1


def test_captions_are_kept_but_images_credits_scripts_and_footnotes_are_not(
    parsed: ParseResult,
) -> None:
    everything = "\n".join(s.content for s in parsed.sections)

    assert "Figure 1.1 Bacteria multiply quickly." in everything
    for dropped in (
        "bacteria under a microscope",
        "Image credit",
        "never ingested",
        "footnote",
        "color: red",
    ):
        assert dropped not in everything


def test_headings_with_no_text_are_counted_not_stored(parsed: ParseResult) -> None:
    headings = [s.section or s.chapter for s in parsed.sections]

    assert "A Heading With Nothing Under It" not in headings
    assert "Safety Basics" not in headings  # a part page has only its title
    assert parsed.empty_headings == 2


def test_spaces_before_punctuation_are_removed(parsed: ParseResult) -> None:
    causes = by_heading(parsed, "Causes of Foodborne Illness").content
    assert "(biological, chemical and physical)." in causes


def test_parsing_is_deterministic(parsed: ParseResult) -> None:
    html = (FIXTURES / "pressbooks_sample.xhtml").read_text(encoding="utf-8")
    again = parse_book(html, book_title="Sample Book", base_url=BASE)
    assert again.content_hash == parsed.content_hash


def test_a_document_without_pressbooks_pages_still_parses_with_a_warning() -> None:
    result = parse_book(
        "<html><body><h1>Only</h1><p>Text.</p></body></html>",
        book_title="Book",
        base_url=BASE,
    )

    assert [s.heading_path for s in result.sections] == ["Book > Only"]
    assert result.warnings


def test_a_table_with_uneven_rows_is_padded() -> None:
    from bs4 import BeautifulSoup

    table = BeautifulSoup(
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td></tr></table>", "html.parser"
    ).table
    assert table is not None

    assert table_to_markdown(table) == "| A | B |\n| --- | --- |\n| 1 |  |"
