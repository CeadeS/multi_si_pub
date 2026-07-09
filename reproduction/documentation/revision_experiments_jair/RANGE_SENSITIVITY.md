# Range Sensitivity of V_dynamic (±20% range variation)

- Generated: 2026-07-06T17:41:54.455616+00:00
- Samples per estimate: `200000` (seed `0`)
- Baseline V_dynamic: `0.8869`
- S = sign × max(|V_narrow−V_base|, |V_wide−V_base|) / (0.2·V_base); sign negative iff narrowing reduces V_dynamic.

| Parameter | Narrow range | Wide range | V_narrow | V_wide | S |
|---|---|---|---:|---:|---:|
| beta_D | [0.145, 0.905] | [0, 1.095] | 0.8932 | 0.8780 | +0.0503 |
| beta_alpha | [0.36, 0.84] | [0.24, 0.96] | 0.8905 | 0.8827 | +0.0238 |
| beta_kappa | [0.75, 2.75] | [0.25, 3.25] | 0.8902 | 0.8782 | +0.0487 |
| beta_Omega | [0.075, 0.275] | [0.025, 0.325] | 0.9094 | 0.8472 | +0.2239 |
| beta_ell | [0.14, 0.46] | [0.06, 0.54] | 0.8909 | 0.8823 | +0.0258 |
| delta | [0.185, 0.865] | [0.015, 1] | 0.8982 | 0.8658 | +0.1190 |
| N | [3, 9] | [2, 11] | 0.9166 | 0.8962 | +0.1675 |
| q_detect | [0.03, 0.27] | [0, 0.33] | 0.8871 | 0.8860 | +0.0051 |

