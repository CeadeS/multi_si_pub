# Revision Experiments: Robustness + Sobol

- Generated: 2026-01-17T09:20:46.976434+00:00
- Seed: `42`
- Default prior: `uniform`
- Uniform samples: `200000`
- Sobol base samples: `8192`

## Robustness Metrics (Default Prior)

- V_C1: `0.998440`
- V_C1_dynamic: `0.980645` (δ ≥ δ*)
- V_C2: `0.961185`
- V_static: `0.959680` (C1* & C2*)
- V_dynamic: `0.941850` (C1* & C2* & C1** under δ prior)

## Margin Quantiles (q05 / q50 / q95)

- `c1_margin`: `0.6713` / `2.0769` / `3.4739`
- `c2_margin`: `0.0764` / `1.0962` / `2.1001`
- `s_static`: `0.0606` / `0.9700` / `1.9801`
- `delta_margin`: `0.1063` / `0.4878` / `0.8945`
- `s_dynamic`: `-0.0300` / `0.4056` / `0.8593`

## Worst-Case Requirements (Robust Parameter Analysis)

### Required β_α vs δ (worst-case over β_D max, β_κ min)
- δ=0.1: β_α ≥ `0.9444`
- δ=0.2: β_α ≥ `0.8750`
- δ=0.3: β_α ≥ `0.7857`
- δ=0.5: β_α ≥ `0.5000`
- δ=0.7: β_α ≥ `0.5000`
- δ=0.9: β_α ≥ `0.5000`

### Required β_Ω vs N (worst-case over β_ℓ max)
- N=2: β_Ω ≥ `1.5000`
- N=3: β_Ω ≥ `1.0000`
- N=4: β_Ω ≥ `0.7500`
- N=5: β_Ω ≥ `0.6000`
- N=6: β_Ω ≥ `0.5000`
- N=7: β_Ω ≥ `0.4286`
- N=8: β_Ω ≥ `0.3750`
- N=9: β_Ω ≥ `0.3333`
- N=10: β_Ω ≥ `0.3000`

## Sobol Sensitivity (First/Total order)

Outputs: `c1_margin`, `c2_margin`, `s_static`, `delta_margin`, `s_dynamic`.
See JSON for full matrices; here we list the top driver per output by total-order index.

- `c1_margin`: top driver `beta_kappa` (T≈0.710)
- `c2_margin`: top driver `beta_Omega` (T≈0.799)
- `s_static`: top driver `beta_Omega` (T≈0.721)
- `delta_margin`: top driver `delta` (T≈0.895)
- `s_dynamic`: top driver `delta` (T≈0.571)

## Sobol By N (Top Driver for `s_dynamic`)

- N=2: `beta_Omega` (T≈0.729)
- N=3: `beta_Omega` (T≈0.588)
- N=5: `delta` (T≈0.717)
- N=10: `delta` (T≈0.835)

## Stability Volume by Fixed N (Default Prior)

Monte Carlo estimates of V_dynamic for each fixed N (other parameters drawn from the declared prior).

- N=2: V_dynamic=`0.7580` (V_C2=`0.7746`)
- N=3: V_dynamic=`0.8963` (V_C2=`0.9160`)
- N=4: V_dynamic=`0.9505` (V_C2=`0.9713`)
- N=5: V_dynamic=`0.9730` (V_C2=`0.9942`)
- N=6: V_dynamic=`0.9786` (V_C2=`1.0000`)
- N=7: V_dynamic=`0.9786` (V_C2=`1.0000`)
- N=8: V_dynamic=`0.9786` (V_C2=`1.0000`)
- N=9: V_dynamic=`0.9786` (V_C2=`1.0000`)
- N=10: V_dynamic=`0.9786` (V_C2=`1.0000`)

