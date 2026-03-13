# PNAS Rewrite Plan — Tightened

## Target
- **Article type:** Research Report, Direct Submission
- **Template:** `pnasmathematics` (single-column, math-heavy) — up to 14 pages
- **Class:** `\documentclass[9pt,twocolumn,twoside]{pnas-new}` with `\templatetype{pnasmathematics}`
- **Classification:** Physical Sciences → Applied Mathematics (major); Computer Sciences or Social Sciences → Economic Sciences (minor)

## Hard Constraints (PNAS 2025)
| Item | Limit |
|---|---|
| Main text | ~4,000 words / 6 pp preferred; 12 pp max; **14 pp for math** |
| Abstract | ≤250 words, single paragraph, no references |
| Significance statement | 50–120 words, no refs, no acronyms, undergrad-accessible |
| Display items (main) | ~4 figures/tables for 6 pp |
| References | ~50, numbered `(1)`, order of appearance |
| Footnotes | ≤13 |
| SI | single PDF, items numbered S1, S2, … |
| Intro heading | **none** — implied by opening paragraphs |
| Discussion | may have subheadings (unlike NMI which forbids them) |
| Methods | after Discussion; may move to SI if main has enough detail |

## LaTeX Metadata Fields
```latex
\significancestatement{...}       % 50–120 words, required
\authorcontributions{...}         % required
\authordeclaration{...}           % competing interests
\datasharing{...}                 % data availability, required
\keywords{kw1 | kw2 | kw3}       % 3–5, pipe-separated
\leadauthor{Hofmann}              % running footer
```

## Section Structure (main.tex)
1. Title, authors, affiliations
2. Abstract (≤250 w, no refs)
3. Significance statement (50–120 w)
4. *(no heading)* — 3 paragraphs: problem → gap → contribution
5. `\section*{Results}` — 4 subsections:
   - Strategic hedging is necessary (Lemma 1 sketch)
   - Simulation incompleteness limits peer verification (Thm 1 sketch)
   - Three stability conditions (Thm 2: C1*, C1**, C2*)
   - Optimal group size depends on benefit structure
6. `\section*{Discussion}` — single block, no subsections:
   - Key findings recap (2 sentences)
   - Policy implications (patience-free regime, group-size selection, β_α measurement)
   - Weak Leviathan mechanism (Thm 3 condensed)
   - Falsifiable predictions (4, one sentence each)
   - Limitations (4, one sentence each)
   - Future work (2 sentences)
7. `\section*{Materials and Methods}` — brief; point to SI for full detail:
   - Stage game + payoff matrix (compact)
   - Parameter ranges table (hard vs. soft bounds)
   - Monte Carlo / Sobol methods (1 paragraph)
8. Acknowledgments, declarations, references

## Figure Budget (main text: 4 items)
| # | Content | Source NMI fig |
|---|---|---|
| 1 | Phase diagram + functional forms | fig:functional |
| 2 | V_dynamic vs. N (stability volume) | fig:v_dynamic |
| 3 | Benefit-structure scaling | fig:scaling |
| 4 | Summary panel (3D param space + Sobol + margins) | fig:summary |

All other figures → SI.

## SI Structure (si.tex → single PDF)
- **S1** Symbol glossary (Table S1)
- **S2** Physical constraints & parameter derivations (SM-A)
- **S3** Proof of Lemma 1 — hedging necessity (SM-B)
- **S4** Proof of Theorem 2 — stability conditions (SM-C)
  - S4.1 C1* (CE obedience)
  - S4.2 C1** (PPE / folk theorem)
  - S4.3 C2* (participation / Shapley / core)
  - S4.4 Coalition deviation cases 1–4
- **S5** Extended sensitivity & Sobol analysis (SM-D)
- **S6** Simulation incompleteness — full proof of Theorem 1 (SM-E)
  - S6.1 Formal setup & resource model
  - S6.2 Proof
  - S6.3 Three monitoring scenarios
  - S6.4 Weak Leviathan formalization (Theorem 3 full)
  - S6.5 Falsifiability conditions
- **S7** Range sensitivity tables (Table S2)
- **S8** Additional figures (Figs S1–Sn)

## Claim-per-Section Map (main text)
Each Results subsection delivers exactly one claim:

| Subsection | Claim | Key equation |
|---|---|---|
| Hedging | Hedging unavoidable under physical constraints | replicator dynamics → ESS at p*∈(0,1) |
| Incompleteness | Peer simulation infeasible for equal-resource systems | C₁ ≥ C₂ + overhead → contradiction |
| Stability | Three conditions necessary & sufficient | C1*: β_α+β_κ≥β_D; C1**: δ≥δ*; C2*: φ_i≥β_ℓ/N |
| Group size | N*=4–5 under linear benefits; varies with structure | V(N)=(N−1)f−cN²/2 → N*=f/c |

Discussion adds: Weak Leviathan (Thm 3 condensed), sensitivity (β_α dominates), predictions, limitations.

## Proof Presentation Template (main text)
For each theorem/lemma in main:
1. One-sentence intuition (*why this matters*)
2. Formal statement (boxed or bold)
3. Proof sketch (3–6 lines, numbered steps)
4. Pointer: "Full proof in SI, Section Sn."

## Narrative Pattern
Each result subsection:
1. **Tension** — "X seems feasible, yet fails unless Y"
2. **Formal condition** — the theorem/lemma
3. **Stakes** — "If Y fails, then Z collapses"
4. **Testable implication** — one falsifiable prediction

## Compression Checklist
- [ ] One core claim per paragraph
- [ ] No repeated motivation after a claim is established
- [ ] Result-first prose; no narrative algebra
- [ ] Notation stable throughout; no symbol redefinition
- [ ] Only cite what directly supports each claim (~50 refs total)
- [ ] Abstract ≤250 words, significance ≤120 words
- [ ] No "Introduction" heading
- [ ] All proofs in SI, only sketches in main

## Editing Passes
1. **Structure** — split NMI content into main vs. SI files
2. **Claims** — verify each section maps to exactly one claim
3. **Density** — cut redundancy, long transitions, repeated motivation
4. **Proof-link** — every formal claim has SI anchor with section number
5. **Word count** — target ≤5,000 words main text (math article allowance)
6. **PNAS compliance** — metadata fields, significance statement, no intro heading, numbered refs, display item count, footnote count ≤13

## Deliverables
- `main.tex` — PNAS submission manuscript (pnasmathematics template)
- `si.tex` — Supporting Information (single file → single PDF)
- `ref.bib` — shared bibliography
- Figures: `figures/` directory (main figs 1–4, SI figs S1–Sn)
- This plan file

## Main ↔ SI Cross-Reference Map

### Formal objects

| Object | Main location | Main contains | SI section | SI contains |
|---|---|---|---|---|
| Lemma 1 (Hedging) | R1 | 3 assumptions + 3-step sketch | S3 | Full replicator dynamics, ESS, mutation-selection |
| Theorem 1 (Incompleteness) | R2 | Resource argument + 3-step sketch | S6.1–S6.2 | Formal setup, 5-step proof, non-saturating remark |
| Monitoring table | R2, Table | 3-row q-value table | S6.3 | Bandwidth calcs, entropy comparison |
| Theorem 2 (Stability) | R3 | 3 equations + intuition per condition | S4 | CE (S4.1), PPE (S4.2), Shapley/core (S4.3), cases (S4.4) |
| V_dynamic | R4, Eq 4 | Definition | S5 | Monte Carlo methodology, per-condition volumes |
| N\* = f/c | R4 (inline) | Formula + interpretation | S2.1 | Calculus derivation, circularity caveat |
| Range sensitivity | R5 | Classification + β_α dominance | S7, Table S2 | Full 7-param table |
| Sobol indices | R5 (mention) | Regime shift statement | S5 | Saltelli sampling, first/total-order, N-dependent |
| Theorem 3 (Weak Leviathan) | Discussion | 4 conditions + interpretation | S6.4 | Full proof, intention-action resolution |
| Falsifiability | Discussion | 4 predictions (1 sentence each) | S6.5 | Detailed conditions, violation criteria |
| Assumptions | R3 (itemize) | A_ext, A_int, A_dev, A_res | S4 preamble | Scope, alternatives |
| Parameter derivations | Methods, Table 1 | Hard vs soft bounds | S2 | A.1–A.5 full derivations |
| Functional forms | Fig 1b | Figure | S2.2 | Hyperbolic, log, parabolic derivations |

### Figures

| Main | Content | SI companions |
|---|---|---|
| Fig 1 | Phase diagram + functional forms | — |
| Fig 2 | V_dynamic vs N + marginal gains | — |
| Fig 3 | Benefit structure scaling (5 panels) | — |
| Fig 4 | Summary: 3D space + volumes + Sobol (end of Results, after R5) | — |
| — | — | Fig S1: Per-condition volumes separately |
| — | — | Fig S2: Full Sobol indices by N |
| — | — | Fig S3: Margin distributions per condition |

### Tables

| Main | Content | SI companions |
|---|---|---|
| Table (R2) | Monitoring scenarios, q values | — |
| Table 1 (Methods) | Hard vs soft parameter bounds | — |
| — | — | Table S1: Symbol glossary |
| — | — | Table S2: Full range sensitivity |
| — | — | Table S3: Physical constraint → parameter mapping |
| — | — | Table S4: Binding coalition by deviation structure |

### Reference flow (main → SI pointers)

```
R1 Lemma 1       →  "Full proof in SI, Section S3"
R2 Theorem 1     →  "Full proof in SI, Section S6.1–S6.2"
R2 Monitoring     →  "Derivations in SI, Section S6.3"
R3 Theorem 2     →  "Derivations in SI, Section S4"
R3 Assumptions    →  "Discussed in SI, Section S4 preamble"
R3 Coalition cases →  "Cases in SI, Section S4.4"
R3 Fig 1          →  "Derivations in SI, Section S2.2"
R4 V_dynamic      →  "Monte Carlo details in SI, Section S5"
R4 N* derivation  →  "Derivation in SI, Section S2.1"
R5 Sensitivity    →  "Full table in SI, Table S2"
R5 Sobol          →  "Analysis in SI, Section S5"
Disc Theorem 3    →  "Full proof in SI, Section S6.4"
Disc Predictions  →  "Detailed conditions in SI, Section S6.5"
Methods Table 1   →  "Derivations in SI, Section S2"
```

## Execution Order (updated)
1. ~~Obtain template files~~ DONE
2. ~~Scaffold main.tex with PNAS metadata~~ DONE
3. ~~Write significance statement and abstract~~ DONE
4. ~~Port introduction (3 paragraphs)~~ DONE
5. ~~Add OCAR scaffolding + placeholders for all formal objects~~ DONE
6. ~~Fill Results prose (R1 → R2 → R3 → R4 → R5)~~ DONE
7. ~~Fill Discussion prose~~ DONE
8. ~~Fill Methods prose~~ DONE
9. ~~Build `si.tex` — port all proofs from NMI SM-A through SM-E~~ DONE
10. ~~Verify all cross-references between main and SI~~ DONE (17/17 resolve)
11. ~~Word count and PNAS compliance check~~ DONE

## si.tex Status (2026-03-13)
- 667 lines, compiles cleanly via tectonic → si.pdf (477 KB)
- 8 sections (S1–S8): symbol glossary, physical constraints, Lemma 1 proof, Theorem 2 proof, sensitivity/Sobol, incompleteness/Thm 3, range sensitivity table, 8 SI figures
- 4 tables (S1–S4): symbols, range sensitivity, physical mapping, binding coalitions
- 8 figures (S1–S8): all PDFs present in figures/
- 17/17 cross-references from main.tex resolve
- Bibliography: unsrtnat style, shared ref.bib

## main.tex Status (2026-03-13)
- 278 lines, 8 pages, compiles cleanly via tectonic
- ~3,500 words (target ≤5,000)
- 0 TODO markers, 0 scaffolding comments
- 19 cite keys, all in ref.bib
- 13 labels, 9 \ref targets, all resolve
- Schimel OCAR verified: all 5 thread sentences intact (R1→R2→R3→R4→R5→Discussion)
- Story chain fixes applied: R3 O/C split, R5→Discussion thread, limitations before predictions

## SI Build Plan (Step 9)

### Source mapping: NMI → PNAS SI
| PNAS SI section | NMI source (lines) | Content |
|---|---|---|
| S1 Symbol glossary | 452–481 | Table, 16 symbols |
| S2 Physical constraints | 483–705 | A.1–A.6, 2 tables, 15+ eqs |
| S3 Lemma 1 proof | 707–798 | B.1–B.5, replicator dynamics |
| S4 Theorem 2 proof | 800–895 | C.1–C.4, Shapley, 1 table |
| S5 Sensitivity & Sobol | 897–901 + scattered | Monte Carlo, Sobol, per-N |
| S6 Incompleteness | 906–1035 | E.0–E.4, Thm 1+3, 1 table |
| S7 Range sensitivity table | from A.6 | Table S2 (±20% results) |
| S8 Additional figures | 8 PDFs | S1–S8 unreferenced in main |

### SI cross-references required by main.tex
(must exist as \label targets in si.tex)
```
main R1  Lemma 1 sketch    →  "Full proof in SI, Section S3"
main R2  Theorem 1 sketch  →  "Full proof in SI, Section S6.1–S6.2"
main R2  Monitoring table   →  "Derivations in SI, Section S6.3"
main R3  C1* intuition      →  "Derivation in SI, Section S4.1"
main R3  C1** intuition     →  "Derivation in SI, Section S4.2"
main R3  C2* intuition      →  "Full case analysis in SI, Section S4.3–S4.4"
main R3  Assumptions        →  "Full derivations in SI, Section S4"
main R3  Fig 1 caption      →  "Derivations in SI, Section S2.2"
main R4  V_dynamic          →  "Monte Carlo details in SI, Section S5"
main R4  N* derivation      →  "derivation in SI, Section S2.1"
main R4  Fig 3 caption      →  "Derivations in SI, Section S2.1"
main R5  Sensitivity        →  "full table in SI, Table S2"
main R5  Sobol              →  "SI, Section S5"
main R5  Fig 4 caption      →  "Extended Sobol analysis in SI, Section S5"
main Discussion  Thm 3      →  "full proof in SI, Section S6.4"
main Discussion  Predictions →  "detailed conditions in SI, Section S6.5"
main Methods  Table 1        →  "Derivations in SI, Section S2"
```

### SI figure assignments
| SI label | File | Content |
|---|---|---|
| Fig S1 | stability_regions.pdf | Per-condition stability volumes |
| Fig S2 | sobol_sensitivity_by_n.pdf | Full Sobol indices by N |
| Fig S3 | margin_distributions.pdf | Margin distributions per condition |
| Fig S4 | ce_convergence_validation.pdf | CE convergence validation |
| Fig S5 | ppe_sustainability_validation.pdf | PPE sustainability validation |
| Fig S6 | shapley_participation_validation.pdf | Shapley participation validation |
| Fig S7 | sensitivity_analysis.pdf | Extended sensitivity analysis |
| Fig S8 | figure_functional_structure_appendix.pdf | Appendix functional structure |

### SI table assignments
| SI label | Content | Source |
|---|---|---|
| Table S1 | Symbol glossary (16 symbols) | NMI 452–481 |
| Table S2 | Range sensitivity (±20%, 7 params) | NMI A.6 |
| Table S3 | Physical constraint → parameter mapping | NMI A.4 |
| Table S4 | Binding coalition by deviation structure | NMI C.3 |

### Build steps for si.tex
1. Create si.tex with standalone document class (article, 11pt)
2. Port S1 (symbol glossary) — relabel as Table S1
3. Port S2 (SM-A) — relabel subsections as S2.1, S2.2; tables as S3
4. Port S3 (SM-B) — relabel subsections as S3.1–S3.5
5. Port S4 (SM-C) — relabel subsections as S4.1–S4.4; table as S4
6. Port S5 (SM-D) — expand with Monte Carlo/Sobol detail
7. Port S6 (SM-E) — relabel subsections as S6.1–S6.5
8. Create S7 — extract range sensitivity table from A.6 as standalone
9. Create S8 — place 8 SI figures with captions
10. Verify every main→SI pointer resolves
11. Compile si.tex independently
