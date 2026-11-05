"""Checks for the paper build, without requiring a TeX installation."""

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_paper_tracked.py"


def builder():
    assert SCRIPT.exists(), "The repeatable tracked-paper builder has not been added"
    spec = importlib.util.spec_from_file_location("paper_build", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bibliography_is_inlined_for_reference_entry_changes():
    source = "body\n\\bibliographystyle{splncs04}\n\\bibliography{references}\n"
    result = builder().inline_bibliography(source, "\\begin{thebibliography}{1}\nentry\n")
    assert "\\bibliography{" not in result
    assert "\\bibliographystyle{" not in result
    assert "entry" in result


def test_missing_bibliography_is_an_error():
    with pytest.raises(ValueError, match="bibliography"):
        builder().inline_bibliography("no bibliography", "entries")


def test_tracked_layout_fits_tables_without_removing_revision_markup():
    source = "\\begin{document}\n\\begin{tabular}{r}\n\\DIFdel{old} \\DIFadd{new}\\\\\n\\end{tabular}"
    result = builder().tracked_layout(source)
    assert "\\resizebox{\\textwidth}{!}{\\begin{tabular}" in result
    assert "\\end{tabular}}" in result
    assert "\\DIFdel{old} \\DIFadd{new}" in result
    assert "\\sloppy" in result


def test_tracked_graphic_frame_fits_within_text_width():
    source = r"\includegraphics[width=\textwidth]{plot.png}"
    result = builder().tracked_layout(source)
    assert r"width=\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax" in result
    assert "{plot.png}" in result


def paper_tree(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "manuscript.tex").write_text("clean manuscript stays unchanged")
    for suffix in ("tex", "pdf"):
        (paper / f"manuscript-tracked.{suffix}").write_text("stale")
    return paper


def test_missing_dependency_invalidates_stale_tracked_outputs(tmp_path, monkeypatch):
    module = builder()
    paper = paper_tree(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Missing"):
        module.build(tmp_path)
    assert not (paper / "manuscript-tracked.pdf").exists()
    assert not (paper / "manuscript-tracked.tex").exists()
    assert (paper / "manuscript.tex").read_text() == "clean manuscript stays unchanged"


def test_failed_compile_does_not_publish_partial_outputs(tmp_path, monkeypatch):
    module = builder()
    paper = paper_tree(tmp_path)
    original = paper / "original"
    original.mkdir()
    (original / "manuscript.tex").write_text("frozen baseline")
    for name in ("references.bib", "llncs.cls", "splncs04.bst"):
        (original / name).write_text("fixture")
    (original / "figures").mkdir()
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.build(tmp_path)
    assert not (paper / "manuscript-tracked.pdf").exists()
    assert not (paper / "manuscript-tracked.tex").exists()
    assert (original / "manuscript.tex").read_text() == "frozen baseline"


def test_missing_original_figure_cannot_reuse_a_stale_staged_image(tmp_path, monkeypatch):
    module = builder()
    paper = paper_tree(tmp_path)
    original = paper / "original"
    original.mkdir()
    (original / "figures").mkdir()
    (original / "manuscript.tex").write_text(
        r"\includegraphics{../results/figures/missing.png}"
    )
    for name in ("references.bib", "llncs.cls", "splncs04.bst"):
        (original / name).write_text("fixture")
    stale = paper / ".build/original/results/figures"
    stale.mkdir(parents=True)
    (stale / "missing.png").write_bytes(b"stale image")
    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/{name}")

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("Missing original input must be caught before compilation")

    monkeypatch.setattr(subprocess, "run", unexpected_compile)
    with pytest.raises(FileNotFoundError, match="missing.png"):
        module.build(tmp_path)
