"""Chunking rules, as pure functions: no database, no API."""

from rag.chunk import HeadingAware, SectionText, chunks_hash


def section(content: str, ordinal: int = 0, heading: str = "Book > Chapter > Topic") -> SectionText:
    return SectionText(
        source_id="bccampus-test",
        ordinal=ordinal,
        chapter="Chapter",
        section="Topic",
        heading_path=heading,
        url="https://example.org/chapter/c/",
        content=content,
    )


def paragraph(words: int, word: str = "word") -> str:
    return " ".join([word] * (words - 1) + [f"{word}."])


def test_a_short_section_is_one_chunk_with_its_heading_path_first() -> None:
    chunks = HeadingAware(max_tokens=400).chunk([section("Wash hands before handling food.")])

    assert len(chunks) == 1
    assert chunks[0].text == "Book > Chapter > Topic\n\nWash hands before handling food."
    assert (chunks[0].part, chunks[0].parts) == (1, 1)


def test_a_long_section_splits_at_paragraph_boundaries_never_mid_paragraph() -> None:
    paragraphs = [paragraph(60, f"p{i}") for i in range(6)]  # ~78 tokens each
    chunks = HeadingAware(max_tokens=200).chunk([section("\n\n".join(paragraphs))])

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        for piece in body.split("\n\n"):
            assert piece in paragraphs  # every paragraph arrives whole


def test_every_chunk_fits_the_limit() -> None:
    content = "\n\n".join(paragraph(n) for n in (30, 250, 45, 120, 300, 10))
    strategy = HeadingAware(max_tokens=200)

    for chunk in strategy.chunk([section(content)]):
        assert chunk.token_estimate <= 200


def test_pieces_are_balanced_not_one_full_and_one_tiny() -> None:
    # 5 paragraphs of ~78 tokens: two pieces of 3+2, not 4+1.
    content = "\n\n".join(paragraph(60, f"p{i}") for i in range(5))
    chunks = HeadingAware(max_tokens=350).chunk([section(content)])

    sizes = [c.token_estimate for c in chunks]
    assert len(sizes) == 2
    assert min(sizes) > max(sizes) / 2


def test_every_piece_of_a_split_section_keeps_its_place() -> None:
    content = "\n\n".join(paragraph(60, f"p{i}") for i in range(6))
    chunks = HeadingAware(max_tokens=200).chunk([section(content, ordinal=7)])

    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert {c.parts for c in chunks} == {len(chunks)}
    assert {c.section_ordinal for c in chunks} == {7}
    assert all(c.text.startswith("Book > Chapter > Topic\n\n") for c in chunks)


def test_a_long_table_splits_by_rows_and_repeats_its_caption_and_header() -> None:
    header = "Table: Safe temperatures\n| Food | Temperature |\n| --- | --- |"
    rows = [f"| food number {i} with a longer description | {i}°C |" for i in range(40)]
    content = header + "\n" + "\n".join(rows)

    chunks = HeadingAware(max_tokens=150).chunk([section(content)])

    assert len(chunks) > 1
    seen: list[str] = []
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        assert body.startswith(header + "\n")  # every piece can be read on its own
        seen.extend(body.removeprefix(header + "\n").split("\n"))
    assert seen == rows  # no row lost, duplicated or reordered


def test_a_long_list_splits_between_items() -> None:
    items = [f"- {paragraph(20, f'item{i}')}" for i in range(20)]
    chunks = HeadingAware(max_tokens=150).chunk([section("\n".join(items))])

    seen: list[str] = []
    for chunk in chunks:
        seen.extend(chunk.text.split("\n\n", 1)[1].split("\n"))
    assert seen == items


def test_a_single_huge_paragraph_splits_by_sentence() -> None:
    sentences = [f"Sentence number {i} says something about food safety." for i in range(80)]
    chunks = HeadingAware(max_tokens=150).chunk([section(" ".join(sentences))])

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        assert body.endswith(".")  # every piece ends on a sentence boundary


def test_the_same_input_always_gives_the_same_chunks_and_hash() -> None:
    sections = [section(paragraph(500)), section("short.", ordinal=1)]
    strategy = HeadingAware(max_tokens=200)

    first, second = strategy.chunk(sections), strategy.chunk(sections)

    assert first == second
    assert chunks_hash(first, strategy.params()) == chunks_hash(second, strategy.params())


def test_different_settings_give_a_different_hash_even_for_identical_chunks() -> None:
    chunks = HeadingAware().chunk([section("short.")])

    assert chunks_hash(chunks, {"max_tokens": 400}) != chunks_hash(chunks, {"max_tokens": 300})
