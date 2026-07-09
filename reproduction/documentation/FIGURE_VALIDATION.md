# Figure Validation Report

This report validates numeric invariants that underlie the main figures.

## Checks
- ✓ V_dynamic nondecreasing with N
- ✓ V_dynamic crosses 95% at N=7 (corrected model, q sampled)
- ✓ V_dynamic ≥ 0.97 at N=8
- ✓ V_dynamic ≤ V_C2 for all N
- ✓ Sobol: beta_Omega dominates s_dynamic at N=2
- ✓ Sobol: beta_Omega remains top driver at N=10 (corrected model)
- ✓ Sobol: delta total-order share grows from N=2 to N=10
- ✓ Anchor C1* margin (q=0) = 1.3 (>0)
- ✓ Anchor C2* margin (N=5) = 0.7 (>0)
- ✓ Anchor C1*(q=0.2) margin matches beta_alpha + (1-q)beta_kappa - beta_D
- ✓ delta*(q) = g/(g+(1-q)kappa_eff) verified at q=0 and q=0.2 (hand-computed 0.2941...)

## Summary
- N grid: [2, 3, 4, 5, 6, 7, 8, 9, 10]
- V_dynamic: ['0.586', '0.778', '0.867', '0.917', '0.945', '0.962', '0.971', '0.976', '0.977']
- V_C2: ['0.600', '0.797', '0.888', '0.938', '0.967', '0.985', '0.994', '0.999', '1.000']
- Sobol top driver at N=2: beta_Omega
- Sobol top driver at N=10: beta_Omega
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

