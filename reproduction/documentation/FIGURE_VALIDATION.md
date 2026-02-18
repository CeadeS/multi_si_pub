# Figure Validation Report

This report validates numeric invariants that underlie the main figures.

## Checks
- ✓ V_dynamic nondecreasing with N
- ✓ V_dynamic crosses 95% at N=4
- ✓ V_dynamic ≥ 0.97 at N=6
- ✓ V_dynamic ≤ V_C2 for all N
- ✓ Sobol: beta_Omega dominates s_dynamic at N=2
- ✓ Sobol: delta dominates s_dynamic at N=10
- ✓ Anchor C1* margin = 1.3 (>0)
- ✓ Anchor C2* margin (N=5) = 0.7 (>0)

## Summary
- N grid: [2, 3, 4, 5, 6, 7, 8, 9, 10]
- V_dynamic: ['0.758', '0.896', '0.951', '0.973', '0.979', '0.979', '0.979', '0.979', '0.979']
- V_C2: ['0.775', '0.916', '0.971', '0.994', '1.000', '1.000', '1.000', '1.000', '1.000']
- Sobol top driver at N=2: beta_Omega
- Sobol top driver at N=10: delta
- Scaling optima (saturating / linear / network): 2 / 3 / 10

## Figure 2 Legacy Comparison

This section compares the Figure 2 sensitivity margins against the legacy binary stability logic.

### Anchor
- Anchor parameters: {'beta_kappa': 1.0, 'beta_alpha': 0.7, 'beta_D': 0.4, 'beta_Omega': 1.0, 'beta_ell': 1.5, 'G0': 1.0}
- Anchor stability margin S: 0.700

### Sweep Comparison

| Parameter | Sign agreement | New crossing | Legacy crossing |
| --- | ---: | ---: | ---: |
| beta_kappa | 1.000 | n/a | n/a |
| beta_alpha | 1.000 | n/a | n/a |
| beta_D | 1.000 | n/a | n/a |
| beta_Omega | 0.000 | n/a | n/a |
| beta_ell | 1.000 | n/a | n/a |
| N | 1.000 | n/a | n/a |

Notes:
- Legacy curves are binary (0/1) and reflect C1* plus a Shapley-based C2* check with fixed synergy.
- Missing legacy panels (beta_ell, N) are reported for sign agreement but should be read as informational only.

