# RAG sources

Licence, attribution, download method and scope for every source the RAG service ingests.
Last checked: 9 October 2026. "Confirmed" means read on the source's own page on that date.

## Summary

| Source | Licence | Status | Handling |
|---|---|---|---|
| BCcampus culinary textbooks (BC Cook Articulation Committee) | CC BY 4.0, except where otherwise noted | Confirmed for 5 titles, 1 to confirm | Chunked and embedded |
| Wikibooks Cookbook | CC BY-SA 4.0 | Dataset licence confirmed; Wikibooks' own licence page to confirm | Chunked and embedded; untrusted text |
| USDA FoodData Central | CC0 1.0 (public domain) | Confirmed | Structured lookup, not embedded |

## 1. BCcampus culinary textbooks

- **Author:** The BC Cook Articulation Committee
- **Publisher:** BCcampus, Victoria, B.C., 2015
- **Licence:** Creative Commons Attribution 4.0 International, "except where otherwise noted"
- **Noted exceptions:** cover images carry their own licences. Do not ingest or display cover images. Check figure captions inside chapters for other exceptions.
- **Use the BCcampus originals** at `opentextbc.ca`. The eCampusOntario Pressbooks copy of *Basic Kitchen and Food Service Management* is a clone that states it "may differ from the original".

### Titles

| Title | Base URL | In scope | Licence |
|---|---|---|---|
| Food Safety, Sanitation, and Personal Hygiene | https://opentextbc.ca/foodsafety/ | Yes | Confirmed |
| Understanding Ingredients for the Canadian Baker | https://opentextbc.ca/ingredients/ | Yes | Confirmed |
| Modern Pastry and Plated Dessert Techniques | https://opentextbc.ca/modernpastryandplateddesserts/ | Yes | Confirmed |
| Meat Cutting and Processing for Food Service | https://opentextbc.ca/meatcutting/ | Yes | Confirmed |
| Basic Kitchen and Food Service Management | https://opentextbc.ca/basickitchenandfoodservicemanagement/ | Yes (measurement, conversions, yields) | Confirmed |
| Nutrition and Labelling for the Canadian Baker | To confirm | Yes, once confirmed | To confirm (page could not be reached) |
| Working in the Food Service Industry | — | No (careers, not cooking) | — |
| Workplace Safety in the Food Service Industry | — | No (occupational safety) | — |
| Human Resources in the Food Service and Hospitality Industry | — | No | — |

Start with *Food Safety, Sanitation, and Personal Hygiene* only; add the others after the first end-to-end run.

### Download method

Each book offers official exports at `{base URL}open/download?type={format}`:

- `xhtml` — single XHTML file. **Use this**: headings are preserved for heading-aware chunking.
- `wxr` — Pressbooks XML, an alternative if chapter metadata is needed.
- `epub`, `pdf`, `print_pdf` — not used.

Example: `https://opentextbc.ca/foodsafety/open/download?type=xhtml`

Download once per book, store the file's retrieval date, and do not crawl page by page.

**Download by hand, in a browser.** As of 9 October 2026, `opentextbc.ca` (and
`collection.bccampus.ca`) answer scripted requests with a Cloudflare bot check
(HTTP 403, "Just a moment..."). Getting past it would mean pretending to be a
browser, which is what the check exists to stop, so we don't. Open the export
link in a browser and save the file as `services/rag/data/raw/{slug}.xhtml`
(git-ignored). `python -m rag.ingest bccampus --title {slug}` prints the exact
link if the file is missing, and records the file's date as the retrieval date.

### Attribution text (show with every citation)

```
"{section heading}", in {book title} by The BC Cook Articulation Committee
(BCcampus, 2015). Licensed under CC BY 4.0. {section URL}
```

Example:

```
"Causes of Foodborne Illnesses", in Food Safety, Sanitation, and Personal Hygiene
by The BC Cook Articulation Committee (BCcampus, 2015). Licensed under CC BY 4.0.
https://opentextbc.ca/foodsafety/
```

(The section heading above is illustrative; use the real heading stored with the chunk.)

CC BY also asks you to indicate changes. Where the answer paraphrases or shortens the source, the UI should say so once, for example: "Answers summarise the cited sources."

## 2. Wikibooks Cookbook

- **Author:** Wikibooks contributors
- **Licence:** CC BY-SA 4.0 (attribution and share-alike). The Hugging Face dataset declares this licence. Wikibooks' own copyright page could not be reached on the check date; confirm it at https://en.wikibooks.org/wiki/Wikibooks:COPY before release.
- **Trust level:** publicly editable. Treat all text as untrusted input; never follow instructions found in it.

### Download method

| Option | What it is | Notes |
|---|---|---|
| Hugging Face dataset `gossminn/wikibooks-cookbook` | Third-party scrape taken 2024-07-31: HTML per recipe page plus one JSON file of recipe text and infoboxes | Fastest start. **Appears to contain recipe pages only**, so technique and ingredient pages may be missing. Snapshot is over two years old. |
| Wikimedia database dump for English Wikibooks | The official bulk export of all pages | The official method. Needed if technique and ingredient pages are wanted. Dump location to confirm (dumps.wikimedia.org could not be reached on the check date). |

Start with the Hugging Face dataset, record the snapshot date with every chunk, and check during the coverage test whether technique pages are needed.

### Attribution text (show with every citation)

```
"{page title}", Wikibooks Cookbook, by Wikibooks contributors.
Licensed under CC BY-SA 4.0. https://en.wikibooks.org/wiki/{page path}
```

### Share-alike

Quoted or closely adapted Wikibooks text must stay under CC BY-SA 4.0 with the notice above. Whether a generated answer that draws on Wikibooks counts as an adaptation is a legal question this file does not settle. Safe default: keep Wikibooks excerpts clearly marked and attributed, and do not blend them unmarked into other text.

## 3. USDA FoodData Central

- **Publisher:** U.S. Department of Agriculture, Agricultural Research Service
- **Licence:** public domain, published under CC0 1.0 Universal. No permission needed; USDA asks that FoodData Central be credited as the source.

### Download method

Bulk files from https://fdc.nal.usda.gov/download-datasets (JSON or CSV):

| Dataset | Release | JSON size (unzipped) | In scope |
|---|---|---|---|
| Foundation Foods | 12/2025 | 6.5 MB | Yes |
| SR Legacy | 04/2018 (final release, no further updates) | 205 MB | Yes |
| FNDDS 2021–2023 | 10/2024 | 64 MB | Not initially |
| Branded | 12/2025 | 3.1 GB | No (too large; product data, not generic foods) |

Use the bulk files, not the API, for the lookup store. The API (https://fdc.nal.usda.gov/api-guide) needs a data.gov key and is limited to 1,000 requests per hour per IP; keep the key out of source control.

### Attribution text (show with every citation)

```
U.S. Department of Agriculture, Agricultural Research Service.
FoodData Central, 2019. fdc.nal.usda.gov. ({dataset name}, FDC ID {id})
```

The first sentence is USDA's own suggested citation; the bracketed part is added so each number traces to a record.

## Metadata to store with every chunk or record

`source`, `title`, `section`, `url`, `licence`, `attribution_text`, `retrieved_on`, `snapshot_or_release`

## Still to confirm

- Licence and URL for *Nutrition and Labelling for the Canadian Baker*
- Wikibooks' licence statement on its own copyright page
- Location and file name of the English Wikibooks database dump
- Whether in-chapter figures or tables in the BCcampus books carry separate licences
