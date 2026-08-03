#!/usr/bin/env python3
"""Regenerate the committed test PDFs — stdlib only, byte-deterministic.

The fixtures are hand-built minimal PDFs rather than real papers so the test suite stays
blob-free (the Phase 1 discipline: no test may require the gitignored corpus on disk).

    python3 rag/tests/fixtures/make_fixtures.py

  two_page.pdf       2 pages, one text block each -> the schema round-trip fixture
  no_text_layer.pdf  1 page, a drawn rectangle and no text operators -> the failure fixture
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

PAGE_BOX = "[0 0 200 200]"


def build_pdf(page_streams: list[str]) -> bytes:
    """A flat, uncompressed PDF: catalog, page tree, one page + one stream per entry."""
    page_count = len(page_streams)
    font_id = 3
    first_page_id = font_id + 1
    objects: dict[int, str] = {
        1: f"<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{first_page_id + 2 * i} 0 R" for i in range(page_count)), page_count
        ),
        font_id: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, stream in enumerate(page_streams):
        page_id = first_page_id + 2 * index
        stream_id = page_id + 1
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox {PAGE_BOX} "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {stream_id} 0 R >>"
        )
        objects[stream_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n".encode("ascii")

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for obj_id in sorted(objects):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return bytes(out)


def text_stream(lines: list[str]) -> str:
    body = "\n".join(f"({line}) Tj 0 -14 Td" for line in lines)
    return f"BT /F1 12 Tf 20 170 Td\n{body}\nET"


def main() -> None:
    two_page = build_pdf(
        [
            text_stream(["Atlas fixture page one.", "Abstract: a tiny committed PDF."]),
            text_stream(["Atlas fixture page two.", "Results: extraction keeps page order."]),
        ]
    )
    # No BT/ET at all: a drawn box is the entire content, so pypdf finds no text layer.
    no_text = build_pdf(["0.5 w 20 20 160 160 re S"])

    (FIXTURES / "two_page.pdf").write_bytes(two_page)
    (FIXTURES / "no_text_layer.pdf").write_bytes(no_text)
    print(f"wrote two_page.pdf ({len(two_page)} B), no_text_layer.pdf ({len(no_text)} B)")


if __name__ == "__main__":
    main()
