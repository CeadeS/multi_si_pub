Multi-SI Coordination: Reviewer Reproduction Package
====================================================

This repository contains the manuscript sources, proofs (in-paper appendices),
and a fully scripted figure and validation pipeline. The goal is to make
reviewer verification fast and unambiguous.

Repository layout
-----------------
- `paper/` — manuscript sources (`main.tex`, `ref.bib`) and required figures.
- `paper/figures/` — PDF figures referenced by `paper/main.tex`.
- `reproduction/` — scripts and data for figure generation and validation.

Reviewer verification checklist
-------------------------------
1) Install requirements:
   ```
   python -m venv .venv
   . .venv/bin/activate
   pip install -r reproduction/requirements.txt
   ```
2) Generate all figures:
   ```
   python reproduction/generate_figures.py
   ```
3) Validate outputs:
   ```
   python reproduction/validate_outputs.py
   python reproduction/validate_figures.py
   python reproduction/validate_proofs.py
   ```
4) Build the paper PDF:
   ```
   cd paper
   tectonic -X compile main.tex
   ```

Where proofs live
-----------------
All formal proofs are contained in `paper/main.tex` (Supplementary Information).

Key validation artifacts
------------------------
- `reproduction/documentation/FIGURE_VALIDATION.md`
- `reproduction/documentation/PROOF_VALIDATION.md`
- `reproduction/documentation/revision_experiments_final/RESULTS.md`
- `reproduction/documentation/revision_experiments_final/RESULTS.json`

Notes
-----
- Under the conservative plausible ranges in the paper, worst-case C2* participation
  requires N >= 10 (see `python reproduction/validate_outputs.py` for the printed ranges).
