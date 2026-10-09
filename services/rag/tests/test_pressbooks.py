"""Parser tests.

Two fixtures:
* foodsafety_excerpt.xhtml: a trimmed excerpt of the real BCcampus export.
  These tests pin the structure the parser has to get right on real books.
* pressbooks_sample.xhtml: hand-made in the same layout, for edge cases the
  real book does not happen to contain.
"""

import pytest
from bs4 import BeautifulSoup

from rag.ingest.pressbooks import (
    ParseResult,
    Section,
    parse_book,
    render_definitions,
    table_to_markdown,
)
from tests.conftest import FIXTURES

REAL_BASE = "https://opentextbc.ca/foodsafety/"
SAMPLE_BASE = "https://opentextbc.ca/sample/"


def parse(fixture: str, base: str) -> ParseResult:
    html = (FIXTURES / fixture).read_text(encoding="utf-8")
    return parse_book(html, book_title="Book", base_url=base)


@pytest.fixture(scope="module")
def real() -> ParseResult:
    return parse("foodsafety_excerpt.xhtml", REAL_BASE)


@pytest.fixture(scope="module")
def sample() -> ParseResult:
    return parse("pressbooks_sample.xhtml", SAMPLE_BASE)


def by_heading(result: ParseResult, heading: str) -> Section:
    matches = [s for s in result.sections if (s.section or s.chapter) == heading]
    assert len(matches) == 1, f"expected one section headed {heading!r}"
    return matches[0]


# ---- the real book's structure -------------------------------------------------


def test_chapter_titles_outrank_the_h1_headings_inside_them(real: ParseResult) -> None:
    # The bug this guards: chapter titles are <h2 class="chapter-title"> but
    # the sections inside are <h1>, so trusting tags made every section a
    # chapter of its own.
    assert [s.heading_path for s in real.sections] == [
        "Book > An Approach to Food Safety",
        "Book > An Approach to Food Safety > The HACCP approach",
        "Book > An Approach to Food Safety > The HACCP approach > Principle 1: Hazard analysis",
        "Book > An Approach to Food Safety > The HACCP approach > "
        "Principle 2: Identifying critical control points",
        "Book > Preventing Foodborne Illness > The Danger Zone",
        "Book > Key Terms",
    ]


def test_number_headings_never_appear(real: ParseResult) -> None:
    # Every page opens with <h3 class="chapter-number">2</h3>.
    for section in real.sections:
        assert not section.chapter.isdigit()
        assert " > 2 >" not in section.heading_path


def test_boilerplate_pages_are_skipped_and_reported(real: ParseResult) -> None:
    assert real.skipped_pages == ["front-matter-accessibility-statement"]
    everything = "\n".join(s.content for s in real.sections)
    assert "accessib" not in everything.lower()


def test_the_glossary_is_kept_as_term_colon_definition(real: ParseResult) -> None:
    assert by_heading(real, "Key Terms").content == (
        "aerobic bacteria: Bacteria that require oxygen in order to grow\n"
        "anaerobic bacteria: Bacteria that only grow in environments where oxygen "
        "is not present\n"
        "contaminants: Unwanted bacteria or substances"
    )


def test_the_danger_zone_table_survives_as_markdown(real: ParseResult) -> None:
    content = by_heading(real, "The Danger Zone").content

    assert "Table: Table 3. Important temperatures to remember" in content
    assert "| Celsius | Fahrenheit | What happens? |" in content
    assert "| 60° | 140° | Most pathogenic bacteria are destroyed." in content
    assert real.tables == 1


def test_citation_urls_land_on_the_heading(real: ParseResult) -> None:
    # No ids on these headings, so the link is a text fragment.
    assert by_heading(real, "The Danger Zone").url == (
        f"{REAL_BASE}chapter/preventing-foodborne-illness/#:~:text=The%20Danger%20Zone"
    )
    assert by_heading(real, "An Approach to Food Safety").url == (
        f"{REAL_BASE}chapter/an-approach-to-food-safety/"
    )
    assert by_heading(real, "Key Terms").url == f"{REAL_BASE}back-matter/key-terms/"


def test_levels_and_fields(real: ParseResult) -> None:
    principle = by_heading(real, "Principle 1: Hazard analysis")

    assert (principle.level, principle.chapter, principle.section) == (
        3,
        "An Approach to Food Safety",
        "Principle 1: Hazard analysis",
    )
    assert [s.ordinal for s in real.sections] == list(range(len(real.sections)))


def test_parsing_is_deterministic(real: ParseResult) -> None:
    assert parse("foodsafety_excerpt.xhtml", REAL_BASE).content_hash == real.content_hash


# ---- edge cases ----------------------------------------------------------------


def test_part_and_front_matter_pages_are_skipped(sample: ParseResult) -> None:
    assert sample.skipped_pages == ["front-matter-introduction", "part-safety-basics"]


def test_a_heading_with_an_id_links_to_the_id(sample: ParseResult) -> None:
    assert by_heading(sample, "Causes of Foodborne Illness").url == (
        f"{SAMPLE_BASE}chapter/foodborne-illness/#causes"
    )


def test_whitespace_and_line_breaks_are_normalised(sample: ParseResult) -> None:
    assert by_heading(sample, "Foodborne Illness").content == (
        "Foodborne illness is caused by eating contaminated food."
    )
    holding = by_heading(sample, "Holding Temperatures").content
    assert holding.startswith("Keep food out of the danger zone. Cold food stays cold.")


def test_spaces_before_punctuation_are_removed(sample: ParseResult) -> None:
    causes = by_heading(sample, "Causes of Foodborne Illness").content
    assert "(biological, chemical and physical)." in causes


def test_lists_keep_their_items_and_nesting(sample: ParseResult) -> None:
    causes = by_heading(sample, "Causes of Foodborne Illness").content
    assert "- Bacteria\n  - Salmonella\n  - Listeria\n- Viruses" in causes
    assert by_heading(sample, "Prevention").content == "1. Wash hands.\n2. Cook thoroughly."


def test_a_pipe_inside_a_table_cell_is_escaped(sample: ParseResult) -> None:
    assert "| Poultry | 74°C \\| 165°F |" in by_heading(sample, "Holding Temperatures").content


def test_captions_are_kept_but_images_credits_scripts_and_footnotes_are_not(
    sample: ParseResult,
) -> None:
    everything = "\n".join(s.content for s in sample.sections)

    assert "Figure 1.1 Bacteria multiply quickly." in everything
    for dropped in (
        "bacteria under a microscope",
        "Image credit",
        "never ingested",
        "footnote",
        "color: red",
    ):
        assert dropped not in everything


def test_headings_with_no_text_are_counted_not_stored(sample: ParseResult) -> None:
    headings = [s.section or s.chapter for s in sample.sections]

    assert "A Heading With Nothing Under It" not in headings
    assert sample.empty_headings == 1


def test_a_document_without_pressbooks_pages_still_parses_with_a_warning() -> None:
    result = parse_book(
        "<html><body><h1>Only</h1><p>Text.</p><h2>Sub</h2><p>More.</p></body></html>",
        book_title="Book",
        base_url=SAMPLE_BASE,
    )

    assert [s.heading_path for s in result.sections] == ["Book > Only", "Book > Only > Sub"]
    assert result.warnings


def test_a_table_with_uneven_rows_is_padded() -> None:
    table = BeautifulSoup(
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td></tr></table>", "html.parser"
    ).table
    assert table is not None

    assert table_to_markdown(table) == "| A | B |\n| --- | --- |\n| 1 |  |"


def test_a_glossary_term_without_a_definition_is_kept() -> None:
    dl = BeautifulSoup(
        "<dl><dt>alpha</dt><dd>first</dd><dt>orphan</dt><dt>beta</dt><dd>second</dd></dl>",
        "html.parser",
    ).dl
    assert dl is not None

    assert render_definitions(dl) == ["alpha: first", "orphan", "beta: second"]
