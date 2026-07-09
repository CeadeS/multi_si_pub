# Revision Experiments: Robustness + Sobol

- Generated: 2026-07-06T17:37:53.114298+00:00
- Seed: `0`
- Default prior: `uniform`
- Uniform samples: `200000`
- Sobol base samples: `20000`

## Robustness Metrics (Default Prior)

- V_C1: `0.996825`
- V_C1_dynamic: `0.979120` (δ ≥ δ*)
- V_C2: `0.907650`
- V_static: `0.904775` (C1* & C2*)
- V_dynamic: `0.886880` (C1* & C2* & C1** under δ prior)

## Margin Quantiles (q05 / q50 / q95)

- `c1_margin`: `0.4288` / `1.5462` / `2.7499`
- `c2_margin`: `-0.0302` / `0.1118` / `0.2435`
- `s_static`: `-0.0324` / `0.1107` / `0.2430`
- `delta_margin`: `0.0831` / `0.4758` / `0.8844`
- `s_dynamic`: `-0.0452` / `0.1028` / `0.2375`

## Worst-Case Requirements (Robust Parameter Analysis)

### Required β_α vs δ (worst-case over β_D max, β_κ min)
- δ=0.1: β_α ≥ `0.8918`
- δ=0.2: β_α ≥ `0.7766`
- δ=0.3: β_α ≥ `0.6538`
- δ=0.5: β_α ≥ `0.6500`
- δ=0.7: β_α ≥ `0.6500`
- δ=0.9: β_α ≥ `0.6500`

### Required β_Ω vs N (worst-case over β_ℓ max)
- N=2: β_Ω ≥ `0.2500`
- N=3: β_Ω ≥ `0.1667`
- N=4: β_Ω ≥ `0.1250`
- N=5: β_Ω ≥ `0.1000`
- N=6: β_Ω ≥ `0.0833`
- N=7: β_Ω ≥ `0.0714`
- N=8: β_Ω ≥ `0.0625`
- N=9: β_Ω ≥ `0.0556`
- N=10: β_Ω ≥ `0.0500`

## Sobol Sensitivity (First/Total order)

Outputs: `c1_margin`, `c2_margin`, `s_static`, `delta_margin`, `s_dynamic`.
See JSON for full matrices; here we list the top driver per output by total-order index.

- `c1_margin`: top driver `beta_kappa` (T≈0.757)
- `c2_margin`: top driver `beta_Omega` (T≈0.705)
- `s_static`: top driver `beta_Omega` (T≈0.693)
- `delta_margin`: top driver `delta` (T≈0.905)
- `s_dynamic`: top driver `beta_Omega` (T≈0.612)

## Sobol By N (Top Driver for `s_dynamic`)

- N=2: `beta_Omega` (T≈0.579)
- N=3: `beta_Omega` (T≈0.694)
- N=5: `beta_Omega` (T≈0.733)
- N=10: `beta_Omega` (T≈0.712)

## Stability Volume by Fixed N (Default Prior)

Monte Carlo estimates of V_dynamic for each fixed N (other parameters drawn from the declared prior).

- N=2: V_dynamic=`0.5860` (V_C2=`0.5995`)
- N=3: V_dynamic=`0.7784` (V_C2=`0.7968`)
- N=4: V_dynamic=`0.8674` (V_C2=`0.8877`)
- N=5: V_dynamic=`0.9169` (V_C2=`0.9382`)
- N=6: V_dynamic=`0.9453` (V_C2=`0.9671`)
- N=7: V_dynamic=`0.9623` (V_C2=`0.9845`)
- N=8: V_dynamic=`0.9714` (V_C2=`0.9938`)
- N=9: V_dynamic=`0.9761` (V_C2=`0.9987`)
- N=10: V_dynamic=`0.9775` (V_C2=`1.0000`)

