"""The sources we are allowed to ingest, and what we know about each one.

Every fact here comes from SOURCES.md, which records where it was checked.
A title that is not listed cannot be ingested: adding a source means checking
its licence first, then adding it here.
"""

from dataclasses import dataclass

BCCAMPUS_LICENCE = "CC BY 4.0"
BCCAMPUS_AUTHOR = "The BC Cook Articulation Committee"
BCCAMPUS_YEAR = 2015


@dataclass(frozen=True)
class BCcampusBook:
    slug: str  # the path segment on opentextbc.ca, e.g. "foodsafety"
    title: str

    @property
    def source_id(self) -> str:
        return f"bccampus-{self.slug}"

    @property
    def base_url(self) -> str:
        return f"https://opentextbc.ca/{self.slug}/"

    @property
    def download_url(self) -> str:
        # The official single-file export. Headings survive in it, which is
        # what heading-aware chunking needs.
        return f"{self.base_url}open/download?type=xhtml"

    def attribution(self, section_heading: str, url: str) -> str:
        """The licence line shown with every citation (format from SOURCES.md)."""
        return (
            f'"{section_heading}", in {self.title} by {BCCAMPUS_AUTHOR} '
            f"(BCcampus, {BCCAMPUS_YEAR}). Licensed under {BCCAMPUS_LICENCE}. {url}"
        )


def attribution_for(source_id: str, heading: str, url: str) -> str:
    """The licence line for a chunk from ``source_id``, headed ``heading``.

    Raises for a source this registry does not know: a chunk whose licence
    line cannot be produced must not be indexed.
    """
    if source_id.startswith("bccampus-"):
        book = BCCAMPUS_BOOKS.get(source_id.removeprefix("bccampus-"))
        if book:
            return book.attribution(heading, url)
    raise KeyError(f"no attribution rule for source {source_id!r}")


BCCAMPUS_BOOKS: dict[str, BCcampusBook] = {
    book.slug: book
    for book in (
        BCcampusBook("foodsafety", "Food Safety, Sanitation, and Personal Hygiene"),
        BCcampusBook("ingredients", "Understanding Ingredients for the Canadian Baker"),
        BCcampusBook(
            "modernpastryandplateddesserts", "Modern Pastry and Plated Dessert Techniques"
        ),
        BCcampusBook("meatcutting", "Meat Cutting and Processing for Food Service"),
        BCcampusBook(
            "basickitchenandfoodservicemanagement",
            "Basic Kitchen and Food Service Management",
        ),
    )
}
