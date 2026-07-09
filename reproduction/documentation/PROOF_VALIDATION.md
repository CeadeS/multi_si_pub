# Proof Validation Report

This report validates numeric identities used in the proofs.

## Checks
- ✓ C1* margin equals beta_alpha + beta_kappa - beta_D
- ✓ C2* margin equals beta_Omega - beta_ell/N at N=5
- ✓ C1*(q) margin equals beta_alpha + (1-q)*beta_kappa - beta_D at q=0.2
- ✓ C1** threshold matches g/(g+kappa_eff) at q=0 when g>0
- ✓ C1** threshold matches g/(g+(1-q)*kappa_eff) at q=0.2 (hand-computed)
- ✓ Patience-free regime: beta_alpha >= beta_D implies delta* = 0 (robust to any q)
- ✓ Phase transition jump matches 1/kappa_eff at q=0
- ✓ Phase transition jump matches 1/((1-q)*kappa_eff) at q=0.2
- ✓ Deterrence effectiveness ratio = ((g+kappa_eff')/(g+kappa_eff))^2 = 49× at beta_kappa=0 vs 10g
- ✓ Onset width kappa_eff/9 gives delta* = 0.1 exactly at q=0
- ✓ Onset width (1-q)*kappa_eff/9 gives delta* = 0.1 at q=0.2
- ✓ Parabolic optimum N* = f/c
- ✓ Parabolic value negative beyond N > 2f/c
- ✓ Information-theoretic limits enforced (beta_alpha <= 1)

## Summary
- Example delta* (q=0): 0.231
- Hand-computed delta*(q=0.2): 0.294118
- Patience-free delta*: 0.000
- Phase jump (approx, q=0): 0.714
- Phase jump (approx, q=0.2): 0.893
- Deterrence ratio: 49.0×
- Critical width: 0.111
- Parabolic N*: 5.000

