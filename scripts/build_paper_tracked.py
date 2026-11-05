"""Rebuild the frozen original, clean revision, and bibliography-aware LaTeX diff."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def inline_bibliography(source: str, bbl: str) -> str:
    source = re.sub(r"\\bibliographystyle\{[^}]+\}", "", source)
    source, count = re.subn(r"\\bibliography\{[^}]+\}", lambda _: bbl, source)
    if count != 1:
        raise ValueError("Expected exactly one bibliography command")
    return source


def tracked_layout(source: str) -> str:
    # Inline old/new cells widen tables; size only the generated comparison.
    source = source.replace(r"\begin{tabular}", r"\resizebox{\textwidth}{!}{\begin{tabular}")
    source = source.replace(r"\end{tabular}", r"\end{tabular}}")
    source = source.replace(
        r"[width=\textwidth]",
        r"[width=\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax]",
    )
    return source.replace(r"\begin{document}", r"\begin{document}" + "\n\\sloppy\n", 1)


def build(root: Path) -> None:
    paper = root / "paper"
    # Invalidate only our generated deliverables, never the editable manuscript.
    for suffix in ("tex", "pdf"):
        (paper / f"manuscript-tracked.{suffix}").unlink(missing_ok=True)
    work = paper / ".build"
    work.mkdir(exist_ok=True)
    tools = {name: shutil.which(name) for name in ("latexmk", "pdflatex", "bibtex")}
    diff = os.environ.get("LATEXDIFF") or shutil.which("latexdiff")
    if not diff and (work / "tools/latexdiff-so").is_file():
        diff = str(work / "tools/latexdiff-so")
    missing = [name for name, path in tools.items() if not path]
    if not diff:
        missing.append("latexdiff (or set LATEXDIFF to latexdiff-so)")
    if missing:
        raise RuntimeError("Missing build dependencies: " + ", ".join(missing))
    if not os.access(diff, os.X_OK) and not shutil.which("perl"):
        raise RuntimeError("Missing perl for standalone latexdiff")

    def run(command: list[str], cwd: Path, log: Path) -> None:
        # Preserve unfiltered diagnostics; RTK tracks supported nested commands.
        if command[0] == "latexmk" and shutil.which("rtk"):
            command = ["rtk", "proxy", *command]
        with log.open("w") as output:
            subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)

    snapshots = {}
    for version, source in (("original", paper / "original"), ("revised", paper)):
        stage = work / version / "paper"
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("manuscript.tex", "references.bib", "llncs.cls", "splncs04.bst"):
            shutil.copyfile(source / name, stage / name)
        figures = source / "figures" if version == "original" else root / "results/figures"
        for name in re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
            (source / "manuscript.tex").read_text(),
        ):
            image = figures / Path(name).name
            if not image.is_file():
                raise FileNotFoundError(f"Missing {version} figure: {image}")
        shutil.copytree(figures, stage.parent / "results/figures", dirs_exist_ok=True)
        # -gg ensures the result cannot silently reuse an earlier successful build.
        run(["latexmk", "-gg", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             "manuscript.tex"], stage, work / f"{version}-build.log")
        snapshots[version] = inline_bibliography(
            (stage / "manuscript.tex").read_text(), (stage / "manuscript.bbl").read_text()
        )

    figure_notes = []
    prefix = "../results/figures/"
    for image in sorted((paper / "original/figures").iterdir()):
        revised = root / "results/figures" / image.name
        identical = revised.is_file() and image.read_bytes() == revised.read_bytes()
        if not identical:
            snapshots["original"] = snapshots["original"].replace(
                prefix + image.name, "original/figures/" + image.name
            )
        figure_notes.append(f"{image.name}: {'unchanged' if identical else 'changed or removed'}")
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", snapshots["revised"]):
        if not (paper / name).is_file():
            raise FileNotFoundError(f"Missing revised figure: {name}")
        if not (paper / "original/figures" / Path(name).name).exists():
            figure_notes.append(f"{Path(name).name}: added")

    stage = work / "tracked"
    stage.mkdir(exist_ok=True)
    for version, text in snapshots.items():
        (stage / f"{version}.tex").write_text(text)
    command = [diff] if os.access(diff, os.X_OK) else ["perl", diff]
    raw = stage / "raw.tex"
    with raw.open("w") as output, (work / "latexdiff.log").open("w") as errors:
        subprocess.run([*command, "--type=UNDERLINE", "--graphics-markup=2",
                        str(stage / "original.tex"), str(stage / "revised.tex")],
                       cwd=paper, stdout=output, stderr=errors, check=True)
    text = tracked_layout(raw.read_text())
    legend = (
        "\n\\begin{center}\\small Revision markup: "
        "\\DIFadd{blue underlined text is added}; "
        "\\DIFdel{red struck-through text is deleted}. "
        "Changed figures are shown with graphical markup.\\end{center}\n"
    )
    text = text.replace("\\maketitle", "\\maketitle" + legend, 1)
    text = "% Generated by scripts/build-paper-tracked.sh; do not edit.\n" + text
    tracked = stage / "manuscript-tracked.tex"
    tracked.write_text(text)
    run(["latexmk", "-gg", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"-outdir={stage}", str(tracked)], paper, work / "tracked-build.log")
    log = (stage / "manuscript-tracked.log").read_text()
    if re.search(r"undefined (references|citations)|(?:Reference|Citation) .+ undefined", log):
        raise RuntimeError("Unresolved references in tracked PDF; inspect paper/.build/")
    shutil.copyfile(work / "revised/paper/manuscript.pdf", paper / "manuscript.pdf")
    shutil.copyfile(tracked, paper / "manuscript-tracked.tex")
    shutil.copyfile(stage / "manuscript-tracked.pdf", paper / "manuscript-tracked.pdf")
    (work / "figure-changes.txt").write_text("\n".join(figure_notes) + "\n")
    for name in ("manuscript.tex", "manuscript.pdf", "manuscript-tracked.tex", "manuscript-tracked.pdf"):
        digest = hashlib.sha256((paper / name).read_bytes()).hexdigest()
        print(f"{digest}  paper/{name}")
    print("Figure review: paper/.build/figure-changes.txt; build logs: paper/.build/")


if __name__ == "__main__":
    try:
        build(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Paper build failed: {error}\nInspect paper/.build/*-build.log and latexdiff.log.",
              file=sys.stderr)
        sys.exit(1)
