# Revision Experiments: Robustness + Sobol

- Generated: 2026-07-17T21:21:35.115136+00:00
- Seed: `0`
- Default prior: `uniform`
- Uniform samples: `200000`
- Sobol base samples: `20000`

## Robustness Metrics (Default Prior)

- V_C1: `0.996825`
- V_C1_dynamic: `0.979120` (δ ≥ δ*)
- V_C2: `0.977815`
- V_myopic: `0.974715` (myopic benchmark: C1* & C2*)
- V_dynamic: `0.958165` (continuation equilibrium: C1** & C2*)
- V_immediate: `0.977725` (immediate enforcement: δ ≥ δ*(q,s) & C2*)
- V_cross: `0.956275` (three-way cross-regime intersection, conservative)

## Margin Quantiles (q05 / q50 / q95)

- `c1_margin`: `0.4288` / `1.5462` / `2.7499`
- `c2_margin`: `0.0251` / `0.1447` / `0.2665`
- `s_static`: `0.0228` / `0.1434` / `0.2661`
- `delta_margin`: `0.0831` / `0.4758` / `0.8844`
- `s_dynamic`: `0.0081` / `0.1335` / `0.2647`

## Worst-Case Requirements (Robust Parameter Analysis)

### Required β_α vs δ (worst-case over β_D max, β_κ min)
- δ=0.1: β_α ≥ `0.8918`
- δ=0.2: β_α ≥ `0.7766`
- δ=0.3: β_α ≥ `0.6538`
- δ=0.5: β_α ≥ `0.6500`
- δ=0.7: β_α ≥ `0.6500`
- δ=0.9: β_α ≥ `0.6500`

### Required β_Ω vs N (worst-case over β_ℓ max)
- N=2: β_Ω ≥ `0.2250`
- N=3: β_Ω ≥ `0.1500`
- N=4: β_Ω ≥ `0.1125`
- N=5: β_Ω ≥ `0.0900`
- N=6: β_Ω ≥ `0.0750`
- N=7: β_Ω ≥ `0.0643`
- N=8: β_Ω ≥ `0.0563`
- N=9: β_Ω ≥ `0.0500`
- N=10: β_Ω ≥ `0.0450`

## Sobol Sensitivity (First/Total order)

Outputs: `c1_margin`, `c2_margin`, `s_static`, `delta_margin`, `s_dynamic`.
See JSON for full matrices; here we list the top driver per output by total-order index.

- `c1_margin`: top driver `beta_kappa` (T≈0.757)
- `c2_margin`: top driver `beta_Omega` (T≈0.861)
- `s_static`: top driver `beta_Omega` (T≈0.834)
- `delta_margin`: top driver `delta` (T≈0.905)
- `s_dynamic`: top driver `beta_Omega` (T≈0.686)

## Sobol By N (Top Driver for `s_dynamic`)

- N=2: `beta_Omega` (T≈0.602)
- N=3: `beta_Omega` (T≈0.673)
- N=5: `beta_Omega` (T≈0.696)
- N=10: `beta_Omega` (T≈0.691)

## Stability Volume by Fixed N (Default Prior)

Monte Carlo estimates of V_dynamic for each fixed N (other parameters drawn from the declared prior).

- N=2: V_dynamic=`0.8629` (V_C2=`0.8777`)
- N=3: V_dynamic=`0.9337` (V_C2=`0.9519`)
- N=4: V_dynamic=`0.9607` (V_C2=`0.9801`)
- N=5: V_dynamic=`0.9725` (V_C2=`0.9925`)
- N=6: V_dynamic=`0.9773` (V_C2=`0.9977`)
- N=7: V_dynamic=`0.9788` (V_C2=`0.9994`)
- N=8: V_dynamic=`0.9792` (V_C2=`1.0000`)
- N=9: V_dynamic=`0.9793` (V_C2=`1.0000`)
- N=10: V_dynamic=`0.9793` (V_C2=`1.0000`)

