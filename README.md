# Game-Theoretic Stability Conditions for Multi-Agent AI Coordination Under Physical Constraints and Imperfect Monitoring

Manuscript sources and a fully scripted reproduction pipeline. Every quantitative
claim in the paper (stability volumes, range sensitivity, Sobol indices) regenerates
bit-identically from the scripts in `reproduction/` (fixed seeds).

**Author:** Martin Hofmann (martin.hofmann@tu-ilmenau.de), Technische Universität Ilmenau
**Co-authors:** Johannes Viehweg (FH Kufstein Tirol); Patrick Mäder (TU Ilmenau / FSU Jena / iDiv)

## Summary

We analyze when multiple independently developed AI systems can coordinate stably,
subject to physical constraints from information theory, thermodynamics, and relativity —
without assuming that actions are perfectly observed. Monitoring quality enters as a
detection-failure probability q derived from the paper's observability analysis and is
carried through every result. The paper proves that strategic hedging is unavoidable,
establishes a simulation-incompleteness result that rules out peer-to-peer internal
verification, derives three individually necessary and jointly sufficient conditions for
stable mediated coordination under imperfect public monitoring — with patience threshold
delta*(q) = g / (g + (1-q)·kappa_eff) — and shows that human institutions can mediate
through observable weakness rather than superior capability.

## Repository layout

```
paper/                       Manuscript sources (self-contained; all proofs in-paper)
  main.tex                   Article incl. appendices A-G and reproducibility checklist
  ref.bib                    Bibliography (biblatex)
  jair.cls, acmart.cls       Vendored journal class (ACM acmart-based)
  acmauthoryear.*, acmdatamodel.dbx  Vendored bibliography style files
  figures/                   PDF figures referenced by main.tex
  main.pdf                   Compiled PDF
reproduction/                Scripts and data to regenerate all figures and results
  run_revision_experiments.py  Monte Carlo / Sobol experiment driver (seed 0)
  range_sensitivity_jair.py    +/-20% range-sensitivity analysis (all 8 parameters)
  run_q0_baseline.py           Perfect-monitoring (q=0) baseline
  generate_figures.py          Regenerate every figure in paper/figures/
  validate_outputs.py          Reproduce and check the numerical results
  validate_proofs.py           Independent checks of the closed-form results
  validate_figures.py          Check generated figures against committed results
  repro/                       Reusable model, figure, and metric modules
  documentation/               Experiment results and validation reports
  requirements.txt             Python dependencies
```

## Build the paper

```
cd paper
pdflatex main && biber main && pdflatex main && pdflatex main
```

All class and bibliography style files are vendored in `paper/`, so no template
download is needed. The build requires a TeX distribution with `biber`
(e.g., TeX Live / TinyTeX).

## Reproduce figures and results

```
python -m venv .venv && . .venv/bin/activate
pip install -r reproduction/requirements.txt

python reproduction/run_revision_experiments.py --out-dir reproduction/documentation/revision_experiments_jair
python reproduction/range_sensitivity_jair.py
python reproduction/generate_figures.py --results reproduction/documentation/revision_experiments_jair/RESULTS.json --out-dir paper/figures
python reproduction/validate_outputs.py --fig-dir paper/figures
python reproduction/validate_proofs.py
python reproduction/validate_figures.py
```

Key result artifacts:
- `reproduction/documentation/revision_experiments_jair/RESULTS.{json,md}` — stability volumes and Sobol indices
- `reproduction/documentation/revision_experiments_jair/RANGE_SENSITIVITY.{json,md}` — range-sensitivity table
- `reproduction/documentation/{PROOF,FIGURE}_VALIDATION.md` — validation reports

All experiments are deterministic (seed 0); a from-scratch rerun reproduces the
committed results exactly.

## License and citation

The manuscript text and figures are © the authors. The vendored `acmart`-based class
and bibliography style files are redistributed under the LaTeX Project Public License
(LPPL). Until the article is published, please cite this repository and contact
martin.hofmann@tu-ilmenau.de.
