# Original submitted manuscript

This directory is a frozen baseline. Do not edit it when revising the paper.

## Provenance

- Source commit: `8132d5e56299d1f813955a4d239820d35ff7f4b0`.
- Reference PDF: `manuscript-draft-yehezkielgunawan.pdf`, supplied by the author.
- Reference PDF SHA-256: `e3a37b2c46bf84da7cdf25bc2ec1c51d15feaaac0787261ccec725643b4f407e`.
- The PDF stored at `paper/manuscript.pdf` in that commit is byte-identical to the supplied PDF.
- `manuscript.tex`, `references.bib`, `llncs.cls`, and `splncs04.bst` were recovered without modification from that commit.

The original figure files were not tracked in Git. The precision–recall image was
recovered from page 5 of the supplied PDF. The current calibration PNG was checked
pixel-for-pixel against the image recovered from page 6; it is identical and is
preserved here with its original PNG encoding. Neither image was recreated from
new experimental results.

## Verification (5 November 2026)

The recovered source was rebuilt with the frozen bibliography, class, style, and
figures in an isolated staging tree. The result has 10 pages and identical extracted
text on every page compared with the submitted PDF (24,301 characters; no added or
removed terms). This is a text comparison, not a claim of byte-identical rebuilt
PDFs or pixel-identical page rendering across TeX versions.

The source retains its historical `../results/figures/` paths. Run the documented
build script from the repository; it stages these images at the expected paths.
Do not compile this directory directly or replace its paths by hand.

## Frozen file checksums (SHA-256)

```text
79bf2d49b754adc05129522988fed2bfb1be805a37a5d66943fef03db51d1b8b  manuscript.tex
e3bd8a4d84fab43cf3c44346f00c7bfe13b3f3c230db1acb7000c5db6ab1b1ec  references.bib
f1be281bc631f07c4405566d64b294ec8df018e75e6d6c6c643cbdd5806677a5  llncs.cls
a370ad563909f7264038e3388d59036306c3ecd53c6de26279ba921f3a9979df  splncs04.bst
403befc50c60ecbba60c8d29d6e67c47c497a208dac1dead446e7e363e1aa289  figures/precision_recall_curve.png
bf68284f49a193e311b9b64125f89936a81ac76c7af2561898ad1cea16958e7e  figures/calibration_curve.png
```
