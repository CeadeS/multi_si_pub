# Range Sensitivity of V_dynamic (±20% range variation)

- Generated: 2026-07-16T21:44:54.581355+00:00
- Samples per estimate: `200000` (seed `0`)
- Baseline V_dynamic: `0.9582`
- S = sign × max(|V_narrow−V_base|, |V_wide−V_base|) / (0.2·V_base); sign negative iff narrowing reduces V_dynamic.

| Parameter | Narrow range | Wide range | V_narrow | V_wide | S |
|---|---|---|---:|---:|---:|
| beta_D | [0.145, 0.905] | [0, 1.095] | 0.9636 | 0.9509 | +0.0377 |
| beta_alpha | [0.36, 0.84] | [0.24, 0.96] | 0.9614 | 0.9547 | +0.0183 |
| beta_kappa | [0.75, 2.75] | [0.25, 3.25] | 0.9599 | 0.9557 | +0.0129 |
| beta_Omega | [0.075, 0.275] | [0.025, 0.325] | 0.9669 | 0.9395 | +0.0974 |
| beta_ell | [0.14, 0.46] | [0.06, 0.54] | 0.9598 | 0.9562 | +0.0103 |
| delta | [0.185, 0.865] | [0.015, 1] | 0.9726 | 0.9326 | +0.1333 |
| N | [3, 9] | [2, 11] | 0.9682 | 0.9604 | +0.0525 |
| q_detect | [0.03, 0.27] | [0, 0.33] | 0.9583 | 0.9574 | +0.0039 |

