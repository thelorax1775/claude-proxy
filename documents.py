"""Files that aren't plain text: notebooks, PDFs, images.

read_file opens things as UTF-8 with errors replaced, which is right for source
and wrong for everything here. A notebook comes back as unreadable JSON with
base64 PNGs inline, a PDF as a screenful of mojibake, an image as nothing
usable at all -- and in each case the model reads the mess and concludes the
file is corrupt rather than that it asked the wrong way.

Each of the three needs something different:

  * A notebook is JSON, so it can be turned into something readable and, more
    importantly, edited a cell at a time. Rewriting one as a whole file is how
    you lose every output and every bit of metadata in it.

  * A PDF has text in it that simply needs extracting. That works with any
    model, which is what makes it worth doing.

  * An image needs the model to actually see it, and most models cannot. That
    one is gated on the model rather than attempted and fudged: a confident
    description of an image nobody looked at is the worst outcome available.
"""

import base64
import json
import logging
import mimetypes
import os
import struct

log = logging.getLogger("claude-proxy")

NOTEBOOK_EXT = ".ipynb"
PDF_EXT = ".pdf"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# Refusing to send a 40MB photo is kinder than a request that dies at the
# provider with a body-too-large the model can make no sense of.
MAX_IMAGE_BYTES = int(os.getenv("AGENT_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

# Which models can be sent an image. There is no capability endpoint to ask, so
# this is a substring list -- deliberately conservative, because guessing wrong
# means every request in the run fails at the provider with a 400 rather than
# the one tool call failing here. Override to add your own.
VISION_MODELS = [s.strip().lower() for s in os.getenv(
    "AGENT_VISION_MODELS",
    "gemma,llava,pixtral,vl,vision,gpt-4o,gpt-5,claude,gemini,minicpm,internvl,molmo",
).split(",") if s.strip()]


def kind_of(path: str) -> str:
    """"notebook", "pdf", "image" or "" for ordinary text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == NOTEBOOK_EXT:
        return "notebook"
    if ext == PDF_EXT:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    return ""


def model_sees_images(model: str) -> bool:
    name = (model or "").lower()
    return any(marker in name for marker in VISION_MODELS)


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

def load_notebook(path: str):
    """(notebook, "") or (None, error)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"[error] could not read the notebook: {e}"
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return None, "[error] that file is not a Jupyter notebook (no 'cells' array)"
    return nb, ""


def _source_text(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _output_text(cell) -> str:
    """A readable summary of what a cell produced.

    Summary rather than reproduction: a cell's outputs include base64 image
    payloads that would swamp the file itself, and what matters when reading a
    notebook is whether it ran and what it said, not the bytes of a plot.
    """
    parts = []
    for out in cell.get("outputs", []) or []:
        if not isinstance(out, dict):
            continue
        kind = out.get("output_type")
        if kind == "stream":
            text = out.get("text", "")
            parts.append("".join(text) if isinstance(text, list) else str(text))
        elif kind == "error":
            parts.append(f"{out.get('ename', 'Error')}: {out.get('evalue', '')}")
        elif kind in ("execute_result", "display_data"):
            data = out.get("data", {}) or {}
            if "text/plain" in data:
                text = data["text/plain"]
                parts.append("".join(text) if isinstance(text, list) else str(text))
            else:
                parts.append(f"[{', '.join(data.keys()) or 'no data'}]")
    return "".join(parts).strip()


def render_notebook(nb: dict, *, with_outputs: bool = True) -> str:
    """The notebook as something worth putting in a model's context."""
    lines = []
    for i, cell in enumerate(nb.get("cells", [])):
        kind = cell.get("cell_type", "code")
        count = cell.get("execution_count")
        marker = f" [{count}]" if kind == "code" and count is not None else ""
        lines.append(f"--- cell {i} ({kind}{marker}) ---")
        lines.append(_source_text(cell) or "(empty)")
        if with_outputs and kind == "code":
            produced = _output_text(cell)
            if produced:
                lines.append(f"  -> {produced}")
    return "\n".join(lines) if lines else "(the notebook has no cells)"


def save_notebook(path: str, nb: dict) -> str:
    """Write atomically. A half-written notebook is unopenable, and this is
    somebody's working file."""
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return f"[error] could not write the notebook: {e}"
    return ""


def new_cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        # Both required by nbformat for a code cell; a notebook missing them
        # opens with a schema complaint.
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def edit_notebook(nb: dict, *, mode: str, index, source: str, cell_type: str) -> str:
    """Apply one edit in place. Returns "" or an [error] line."""
    cells = nb["cells"]
    if mode in ("replace", "delete"):
        if not isinstance(index, int) or not 0 <= index < len(cells):
            return (f"[error] cell {index} does not exist; the notebook has "
                    f"{len(cells)} cells (0-{len(cells) - 1})" if cells else
                    "[error] the notebook has no cells")

    if mode == "delete":
        cells.pop(index)
        return ""

    if mode == "replace":
        cell = cells[index]
        cell["source"] = source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            # The outputs described the old source. Keeping them would show a
            # result that no longer belongs to the code above it.
            cell["outputs"] = []
            cell["execution_count"] = None
        return ""

    if mode == "insert":
        at = len(cells) if index is None else index
        if not isinstance(at, int) or not 0 <= at <= len(cells):
            return f"[error] cannot insert at {index}; valid positions are 0-{len(cells)}"
        cells.insert(at, new_cell(cell_type, source))
        return ""

    return f"[error] '{mode}' is not an edit mode; use insert, replace or delete"


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

def pdf_text(path: str, *, first_page: int = 1, last_page: int = 0) -> str:
    """Extract text, page by page. Returns text or an [error] line.

    Text only: this is for reading a document, not reproducing it. A scanned
    PDF has no text layer and will come back empty, and saying so is more use
    than silently returning nothing.
    """
    try:
        import pypdf
    except ImportError:
        return ("[error] reading PDFs needs the pypdf package, which isn't installed. "
                "Run: pip install pypdf")
    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        return f"[error] could not open the PDF: {e}"

    total = len(reader.pages)
    start = max(1, int(first_page or 1))
    end = min(total, int(last_page) if last_page else total)
    if start > total:
        return f"[error] page {start} does not exist; the PDF has {total} pages"

    chunks, empty = [], 0
    for number in range(start, end + 1):
        try:
            text = (reader.pages[number - 1].extract_text() or "").strip()
        except Exception as e:
            text = f"[could not extract this page: {e}]"
        if not text:
            empty += 1
        chunks.append(f"--- page {number} of {total} ---\n{text or '(no text on this page)'}")

    body = "\n\n".join(chunks)
    if empty == (end - start + 1):
        body += ("\n\n[note] no text at all was extracted. This is usually a scanned "
                 "PDF -- the pages are images, and reading it needs OCR rather than "
                 "text extraction.")
    return body


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def image_size(path: str) -> tuple[int, int] | None:
    """(width, height) from the file header, or None.

    Parsed by hand rather than with Pillow: the sizes live in the first few
    bytes of each format, and the alternative is a compiled dependency pulled
    in to read two integers.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return struct.unpack("<HH", head[6:10])
            if head[:2] == b"\xff\xd8":                       # JPEG
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                        f.read(3)
                        height, width = struct.unpack(">HH", f.read(4))
                        return width, height
                    length = struct.unpack(">H", f.read(2))[0]
                    f.seek(length - 2, os.SEEK_CUR)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                if head[12:16] == b"VP8 ":
                    return struct.unpack("<HH", head[26:30])
                if head[12:16] == b"VP8L":
                    bits = struct.unpack("<I", head[21:25])[0]
                    return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    except (OSError, struct.error):
        return None
    return None


def image_data_url(path: str) -> str | None:
    """The image as a data: URL, or None if it's too big to send."""
    try:
        if os.path.getsize(path) > MAX_IMAGE_BYTES:
            return None
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def describe_image(path: str) -> str:
    size = image_size(path)
    dimensions = f"{size[0]}x{size[1]}" if size else "unknown size"
    try:
        length = os.path.getsize(path)
    except OSError:
        length = 0
    mime = mimetypes.guess_type(path)[0] or "image"
    return f"{mime}, {dimensions}, {length} bytes"
