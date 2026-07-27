"""
Mistral OCR extraction + hierarchical chunking for PDF/Word documents.

Mistral's OCR returns markdown per page, which preserves
headings (#, ##, ...), tables (| a | b |), and lists -- so we can chunk
along that structure instead of blindly slicing every N characters.

Chunking strategy:
  1. Walk the combined markdown line by line, tracking a "heading stack"
     (the current # > ## > ### path). A new section starts at every
     heading.
  2. Any contiguous block of table rows is consumed atomically -- a table
     is never split across two chunks.
  3. Within a section, paragraphs (blank-line-separated blocks) are packed
     into chunks up to MAX_CHUNK_CHARS. A chunk is only ever cut at a
     paragraph boundary, never mid-table, mid-list-item is possible only
     if a single list item alone exceeds the limit (rare).
  4. Every chunk is prefixed with its heading breadcrumb (e.g.
     "[Section: Report > Q3 Results > Revenue]") so it reads sensibly even
     retrieved on its own, out of context.
"""
import base64
import os
import re
from pathlib import Path

try:
    # SDK >= 2.x
    from mistralai.client import Mistral
except ImportError:
    # SDK 1.x
    from mistralai import Mistral

MAX_CHUNK_CHARS = 1800

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _get_client() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Get a free key at https://console.mistral.ai "
            "and add it to your .env file."
        )
    return Mistral(api_key=api_key)


def ocr_extract_pages(file_path: Path) -> list[dict]:
    """
    Runs Mistral OCR on a local PDF/Word file. Returns
    [{"index": int, "markdown": str}, ...] -- one entry per page, with
    headings/tables/lists preserved as markdown structure.
    """
    mime = _MIME_TYPES.get(file_path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported file type for OCR: {file_path.suffix}")

    client = _get_client()
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:{mime};base64,{b64}",
        },
    )
    return [{"index": p.index, "markdown": p.markdown or ""} for p in response.pages]


def _paged_lines(pages: list[dict]) -> list[tuple[int, str]]:
    """Flattens all pages into a single (page_index, line_text) stream, so
    each line carries its own accurate page number -- no marker-line
    bookkeeping, no leakage across section/page boundaries."""
    out = []
    for p in pages:
        for line in p["markdown"].split("\n"):
            out.append((p["index"], line))
    return out


def _finalize_chunk(breadcrumb: str, lines: list[tuple[int, str]]) -> dict:
    pages_seen = [pg for pg, _ in lines]
    content = "\n".join(text for _, text in lines).strip()
    if breadcrumb:
        content = f"[Section: {breadcrumb}]\n{content}"
    return {
        "content": content,
        "breadcrumb": breadcrumb,
        "page_start": min(pages_seen) if pages_seen else None,
        "page_end": max(pages_seen) if pages_seen else None,
    }


def chunk_markdown_pages(pages: list[dict]) -> list[dict]:
    """
    Returns [{"content": str, "breadcrumb": str, "page_start": int,
    "page_end": int}, ...] -- hierarchical chunks as described in the
    module docstring.
    """
    if not pages:
        return []

    lines = _paged_lines(pages)
    heading_stack: list[tuple[int, str]] = []
    sections: list[dict] = []
    current_lines: list[tuple[int, str]] = []

    def flush_section():
        nonlocal current_lines
        if any(text.strip() for _, text in current_lines):
            breadcrumb = " > ".join(h[1] for h in heading_stack)
            sections.append({"breadcrumb": breadcrumb, "lines": list(current_lines)})
        current_lines = []

    i = 0
    while i < len(lines):
        page_idx, line = lines[i]

        hm = _HEADING_RE.match(line)
        if hm:
            flush_section()
            level = len(hm.group(1))
            text = hm.group(2).strip()
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, text))
            current_lines.append((page_idx, line))
            i += 1
            continue

        if _TABLE_ROW_RE.match(line):
            # Consume the whole contiguous table block atomically so it's
            # never split across chunks.
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i][1]):
                current_lines.append(lines[i])
                i += 1
            continue

        current_lines.append((page_idx, line))
        i += 1

    flush_section()

    # Second pass: pack each section's paragraphs into <=MAX_CHUNK_CHARS
    # chunks, only ever cutting at a blank-line paragraph boundary (table
    # rows have no blank lines between them, so tables are never cut).
    chunks = []
    for sec in sections:
        paragraphs: list[list[tuple[int, str]]] = []
        buf: list[tuple[int, str]] = []
        for pg, text in sec["lines"]:
            if text.strip() == "":
                if buf:
                    paragraphs.append(buf)
                    buf = []
            else:
                buf.append((pg, text))
        if buf:
            paragraphs.append(buf)

        current_chunk_lines: list[tuple[int, str]] = []
        current_len = 0
        for para in paragraphs:
            para_text = "\n".join(t for _, t in para)
            if current_chunk_lines and current_len + len(para_text) > MAX_CHUNK_CHARS:
                chunks.append(_finalize_chunk(sec["breadcrumb"], current_chunk_lines))
                current_chunk_lines = []
                current_len = 0
            current_chunk_lines.extend(para)
            current_len += len(para_text)
        if current_chunk_lines:
            chunks.append(_finalize_chunk(sec["breadcrumb"], current_chunk_lines))

    return [c for c in chunks if c["content"].strip()]


def ocr_and_chunk(file_path: Path) -> list[dict]:
    """Convenience wrapper: OCR a file, then hierarchically chunk it."""
    pages = ocr_extract_pages(file_path)
    return chunk_markdown_pages(pages)