"""Notebooks, PDFs and images.

The thread running through all of it: opened as UTF-8 text these files read as
garbage, and a model reading garbage concludes the file is corrupt rather than
that it asked the wrong way. Each test is really asking "does the model get
something true, or something misleading?"
"""

import base64
import json
import os
import struct

import pytest

import agent_tools
import documents


@pytest.fixture
def ctx(app, tmp_path):
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    context = agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["files", "files_write"])
    context.model = "nemotron-3-ultra"          # no vision, the common case
    return context


def notebook(tmp_path, cells=None):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({
        "cells": cells if cells is not None else [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": 1,
             "source": ["print('hi')\n"],
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hi\n"]}]},
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }))
    return path


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

async def test_a_notebook_reads_as_cells_not_json(ctx, tmp_path):
    path = notebook(tmp_path)
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "cell 0 (markdown)" in out
    assert "# Title" in out
    assert "print('hi')" in out
    assert "-> hi" in out
    assert '"nbformat"' not in out          # not the raw file


async def test_notebook_outputs_are_summarised_not_reproduced(ctx, tmp_path):
    """A cell's outputs can include a base64 PNG that dwarfs the notebook. What
    matters on read is whether it ran and what it said."""
    path = notebook(tmp_path, cells=[
        {"cell_type": "code", "metadata": {}, "execution_count": 2, "source": ["plot()\n"],
         "outputs": [{"output_type": "display_data",
                      "data": {"image/png": "A" * 50000}}]},
    ])
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "image/png" in out
    assert "A" * 200 not in out


async def test_an_error_output_is_visible(ctx, tmp_path):
    """The most useful thing in a notebook is often the traceback."""
    path = notebook(tmp_path, cells=[
        {"cell_type": "code", "metadata": {}, "execution_count": 1, "source": ["boom()\n"],
         "outputs": [{"output_type": "error", "ename": "NameError",
                      "evalue": "name 'boom' is not defined", "traceback": []}]},
    ])
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "NameError" in out and "not defined" in out


async def test_editing_a_cell_leaves_the_others_alone(ctx, tmp_path):
    """The whole reason this isn't write_file."""
    path = notebook(tmp_path)
    await agent_tools.run_notebook_edit(ctx, str(path), mode="replace", cell=0,
                                        source="# Changed\n")
    nb = json.loads(path.read_text())
    assert "".join(nb["cells"][0]["source"]) == "# Changed\n"
    # Cell 1 keeps its output and execution count.
    assert nb["cells"][1]["outputs"][0]["text"] == ["hi\n"]
    assert nb["cells"][1]["execution_count"] == 1
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


async def test_replacing_code_clears_its_stale_output(ctx, tmp_path):
    """Keeping it would show a result that no longer belongs to the code above
    it -- worse than showing none."""
    path = notebook(tmp_path)
    await agent_tools.run_notebook_edit(ctx, str(path), mode="replace", cell=1,
                                        source="print('different')\n")
    nb = json.loads(path.read_text())
    assert nb["cells"][1]["outputs"] == []
    assert nb["cells"][1]["execution_count"] is None


async def test_insert_and_delete(ctx, tmp_path):
    path = notebook(tmp_path)
    await agent_tools.run_notebook_edit(ctx, str(path), mode="insert", cell=0,
                                        source="import os\n", cell_type="code")
    nb = json.loads(path.read_text())
    assert len(nb["cells"]) == 3
    assert "".join(nb["cells"][0]["source"]) == "import os\n"
    assert nb["cells"][0]["outputs"] == []          # nbformat needs these
    assert "execution_count" in nb["cells"][0]

    await agent_tools.run_notebook_edit(ctx, str(path), mode="delete", cell=0)
    assert len(json.loads(path.read_text())["cells"]) == 2


async def test_insert_without_an_index_appends(ctx, tmp_path):
    path = notebook(tmp_path)
    await agent_tools.run_notebook_edit(ctx, str(path), mode="insert",
                                        source="# last\n", cell_type="markdown")
    nb = json.loads(path.read_text())
    assert "".join(nb["cells"][-1]["source"]) == "# last\n"


async def test_a_bad_cell_index_says_what_exists(ctx, tmp_path):
    path = notebook(tmp_path)
    out = await agent_tools.run_notebook_edit(ctx, str(path), mode="replace", cell=99,
                                              source="x")
    assert out.startswith("[error]")
    assert "2 cells" in out


async def test_a_corrupt_notebook_is_reported_not_crashed_on(ctx, tmp_path):
    path = tmp_path / "bad.ipynb"
    path.write_text("{not json at all")
    assert (await agent_tools.run_read_file(ctx, str(path))).startswith("[error]")


async def test_a_json_file_that_is_not_a_notebook_is_refused(ctx, tmp_path):
    path = tmp_path / "notreally.ipynb"
    path.write_text('{"hello": "world"}')
    out = await agent_tools.run_read_file(ctx, str(path))
    assert out.startswith("[error]")
    assert "not a Jupyter notebook" in out


async def test_notebook_edit_refuses_a_non_notebook(ctx, tmp_path):
    path = tmp_path / "plain.py"
    path.write_text("x = 1\n")
    assert (await agent_tools.run_notebook_edit(ctx, str(path), source="y")).startswith("[error]")


async def test_notebook_edit_is_a_write_group_tool():
    """It changes somebody's file, so it belongs with the other things you
    can't take back."""
    entry = next(t for t in agent_tools.TOOLS if t["name"] == "notebook_edit")
    assert entry["group"] == "files_write"
    assert "files_write" not in agent_tools.DEFAULT_GROUPS


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

def make_pdf(path, pages_text):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in pages_text:
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


async def test_a_pdf_reads_as_text_with_page_markers(ctx, tmp_path):
    pytest.importorskip("pypdf")
    path = make_pdf(tmp_path / "doc.pdf", ["", ""])
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "page 1 of 2" in out
    assert "page 2 of 2" in out


async def test_a_pdf_with_no_text_layer_says_so(ctx, tmp_path):
    """Blank pages are indistinguishable from a scan, and silently returning
    nothing invites the model to invent the contents."""
    pytest.importorskip("pypdf")
    path = make_pdf(tmp_path / "scan.pdf", ["", ""])
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "scanned PDF" in out and "OCR" in out


def test_a_corrupt_pdf_is_reported(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    assert documents.pdf_text(str(path)).startswith("[error]")


def test_asking_past_the_end_says_how_many_pages(tmp_path):
    pytest.importorskip("pypdf")
    path = make_pdf(tmp_path / "doc.pdf", [""])
    out = documents.pdf_text(str(path), first_page=9)
    assert out.startswith("[error]") and "1 pages" in out


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


async def test_a_model_without_vision_is_told_it_cannot_see(ctx, tmp_path):
    """The worst available outcome is a confident description of an image
    nobody looked at."""
    path = tmp_path / "pic.png"
    path.write_bytes(RED_PNG)
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "can't be shown images" in out
    assert "rather than guessing" in out
    assert ctx.pending_images == []          # nothing queued to send


async def test_a_vision_model_gets_the_image_queued(ctx, tmp_path):
    ctx.model = "gemma4:31b"
    path = tmp_path / "pic.png"
    path.write_bytes(RED_PNG)
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "follows this round's tool results" in out
    assert len(ctx.pending_images) == 1
    assert ctx.pending_images[0]["data_url"].startswith("data:image/png;base64,")


async def test_an_oversized_image_is_refused_before_it_is_sent(ctx, tmp_path, monkeypatch):
    """Better than a request that dies at the provider with a body-too-large
    the model can make nothing of."""
    ctx.model = "gemma4:31b"
    monkeypatch.setattr(documents, "MAX_IMAGE_BYTES", 10)
    path = tmp_path / "big.png"
    path.write_bytes(RED_PNG)
    out = await agent_tools.run_read_file(ctx, str(path))
    assert "Too large to send" in out
    assert ctx.pending_images == []


def test_image_dimensions_come_out_of_the_header(tmp_path):
    path = tmp_path / "pic.png"
    path.write_bytes(RED_PNG)
    assert documents.image_size(str(path)) == (1, 1)
    assert "image/png" in documents.describe_image(str(path))


def test_a_truncated_image_does_not_crash_the_reader(tmp_path):
    path = tmp_path / "trunc.png"
    path.write_bytes(RED_PNG[:12])
    assert documents.image_size(str(path)) is None


@pytest.mark.parametrize("model, sees", [
    ("gemma4:31b", True),
    ("llava:13b", True),
    ("qwen2-vl", True),
    ("nemotron-3-ultra", False),
    ("gpt-oss:120b", False),
    ("", False),
])
def test_which_models_are_treated_as_vision_capable(model, sees):
    """Conservative on purpose: guessing wrong means every request in the run
    fails at the provider, not just this one tool call."""
    assert documents.model_sees_images(model) is sees


def test_kind_detection():
    assert documents.kind_of("/a/b.ipynb") == "notebook"
    assert documents.kind_of("/a/B.PDF") == "pdf"
    assert documents.kind_of("/a/pic.JPEG") == "image"
    assert documents.kind_of("/a/main.py") == ""
