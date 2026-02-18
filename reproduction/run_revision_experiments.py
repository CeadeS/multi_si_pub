#!/usr/bin/env python3
"""Run robustness metrics + Sobol sensitivity for the mediator stability model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from repro.model import ParamRanges, plausible_ranges
from repro.robustness_metrics import (
    PriorSpec,
    PriorFamily,
    run_metrics,
    sobol_indices,
    volume_metrics_by_fixed_N,
    worst_case_beta_alpha_required,
    worst_case_beta_omega_required,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prior_markdown(prior: PriorSpec) -> str:
    ranges = prior.ranges
    lines: list[str] = []
    lines.append(f"# Prior Specification: `{prior.name}`")
    lines.append("")
    lines.append("All probability/sensitivity statements are conditional on this declared prior.")
    lines.append("")
    lines.append("| Parameter | Range | Family | Notes |")
    lines.append("|---|---:|---|---|")

    def family_for(key: str) -> str:
        return prior.families.get(key, PriorFamily.UNIFORM).value

    lines.append(f"| β_D | [{ranges.beta_D[0]}, {ranges.beta_D[1]}] | {family_for('beta_D')} | defection temptation |")
    lines.append(f"| β_κ | [{ranges.beta_kappa[0]}, {ranges.beta_kappa[1]}] | {family_for('beta_kappa')} | conflict / deterrence |")
    lines.append(f"| β_α | [{ranges.beta_alpha[0]}, {ranges.beta_alpha[1]}] | {family_for('beta_alpha')} | coordination / signal value |")
    lines.append(f"| β_Ω | [{ranges.beta_Omega[0]}, {ranges.beta_Omega[1]}] | {family_for('beta_Omega')} | oversight / membership value |")
    lines.append(f"| β_ℓ | [{ranges.beta_ell[0]}, {ranges.beta_ell[1]}] | {family_for('beta_ell')} | removal / outside option |")
    lines.append(
        f"| N | [{ranges.n_agents[0]}, {ranges.n_agents[1]}] | uniform | sampled as discrete-uniform over integers (rounded after sampling on [{ranges.n_agents[0]-0.5}, {ranges.n_agents[1]+0.5}]) |"
    )
    lines.append(f"| δ | [{prior.delta_range[0]}, {prior.delta_range[1]}] | uniform | discount factor prior |")
    lines.append("")
    lines.append("Uniform priors are the maximum-entropy choice given bounds; log-uniform priors stress-test order-of-magnitude uncertainty.")
    return "\n".join(lines) + "\n"


def _write_md(path: Path, payload: dict) -> None:
    default = payload["runs"][payload["meta"]["default_prior"]]
    volume = default["metrics"]["volume"]
    quantiles = default["metrics"]["quantiles"]

    lines: list[str] = []
    lines.append("# Revision Experiments: Robustness + Sobol")
    lines.append("")
    lines.append(f"- Generated: {payload['meta']['generated_at']}")
    lines.append(f"- Seed: `{payload['meta']['seed']}`")
    lines.append(f"- Default prior: `{payload['meta']['default_prior']}`")
    lines.append(f"- Uniform samples: `{default['metrics']['prior']['n_samples']}`")
    lines.append(f"- Sobol base samples: `{default['sobol']['n_base']}`")
    lines.append("")
    lines.append("## Robustness Metrics (Default Prior)")
    lines.append("")
    lines.append(f"- V_C1: `{volume['V_C1']:.6f}`")
    lines.append(f"- V_C1_dynamic: `{volume['V_C1_dynamic']:.6f}` (δ ≥ δ*)")
    lines.append(f"- V_C2: `{volume['V_C2']:.6f}`")
    lines.append(f"- V_static: `{volume['V_static']:.6f}` (C1* & C2*)")
    lines.append(f"- V_dynamic: `{volume['V_dynamic']:.6f}` (C1* & C2* & C1** under δ prior)")
    lines.append("")
    lines.append("## Margin Quantiles (q05 / q50 / q95)")
    lines.append("")
    for key in ("c1_margin", "c2_margin", "s_static", "delta_margin", "s_dynamic"):
        q = quantiles[key]
        lines.append(f"- `{key}`: `{q['q05']:.4f}` / `{q['q50']:.4f}` / `{q['q95']:.4f}`")
    lines.append("")
    lines.append("## Worst-Case Requirements (Robust Parameter Analysis)")
    lines.append("")
    lines.append("### Required β_α vs δ (worst-case over β_D max, β_κ min)")
    for delta_str, value in payload["worst_case"]["beta_alpha_required_by_delta"].items():
        lines.append(f"- δ={delta_str}: β_α ≥ `{value:.4f}`")
    lines.append("")
    lines.append("### Required β_Ω vs N (worst-case over β_ℓ max)")
    for n_str, value in payload["worst_case"]["beta_omega_required_by_N"].items():
        lines.append(f"- N={n_str}: β_Ω ≥ `{value:.4f}`")
    lines.append("")
    lines.append("## Sobol Sensitivity (First/Total order)")
    lines.append("")
    lines.append("Outputs: `c1_margin`, `c2_margin`, `s_static`, `delta_margin`, `s_dynamic`.")
    lines.append("See JSON for full matrices; here we list the top driver per output by total-order index.")
    lines.append("")
    sobol = default["sobol"]
    parameter_names = sobol["parameter_names"]
    output_names = sobol["output_names"]
    s_total = np.array(sobol["S_total"], dtype=float)  # shape: (d, outputs)
    for output_idx, output_name in enumerate(output_names):
        column = s_total[:, output_idx]
        best_idx = int(np.nanargmax(column))
        lines.append(f"- `{output_name}`: top driver `{parameter_names[best_idx]}` (T≈{column[best_idx]:.3f})")
    lines.append("")

    if "sobol_per_N" in default:
        lines.append("## Sobol By N (Top Driver for `s_dynamic`)")
        lines.append("")
        for n_str, sobol_n in default["sobol_per_N"].items():
            param_names_n = sobol_n["parameter_names"]
            out_names_n = sobol_n["output_names"]
            st_n = np.array(sobol_n["S_total"], dtype=float)
            idx_dyn = out_names_n.index("s_dynamic")
            order = np.argsort(-st_n[:, idx_dyn])
            # Skip the trivial fixed-N entry if present.
            top = next((i for i in order if param_names_n[i] != "N"), int(order[0]))
            lines.append(f"- N={n_str}: `{param_names_n[top]}` (T≈{st_n[top, idx_dyn]:.3f})")
        lines.append("")

    if "volume_by_N" in default:
        lines.append("## Stability Volume by Fixed N (Default Prior)")
        lines.append("")
        lines.append("Monte Carlo estimates of V_dynamic for each fixed N (other parameters drawn from the declared prior).")
        lines.append("")
        for n_str in map(str, range(2, 11)):
            vol = default["volume_by_N"][n_str]
            lines.append(f"- N={n_str}: V_dynamic=`{vol['V_dynamic']:.4f}` (V_C2=`{vol['V_C2']:.4f}`)")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="documentation/revision_experiments", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--n-samples", type=int, default=200_000, help="Uniform Monte Carlo samples")
    parser.add_argument("--sobol-n-base", type=int, default=20_000, help="Sobol base sample size")
    parser.add_argument("--delta-min", type=float, default=0.1, help="δ prior min (inclusive)")
    parser.add_argument("--delta-max", type=float, default=0.95, help="δ prior max (inclusive)")
    parser.add_argument(
        "--prior",
        action="append",
        choices=["uniform", "log_uniform"],
        default=["uniform"],
        help="Prior configuration to run (can be repeated)",
    )
    parser.add_argument("--skip-sobol", action="store_true", help="Skip Sobol computation")
    parser.add_argument(
        "--sobol-per-n",
        default="2,3,5,10",
        help="Comma-separated N values for per-N Sobol (default: 2,3,5,10)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranges = plausible_ranges()
    prior_defs: dict[str, PriorSpec] = {}
    for prior_name in args.prior:
        families: dict[str, PriorFamily] = {}
        if prior_name == "log_uniform":
            families = {
                "beta_D": PriorFamily.LOG_UNIFORM,
                "beta_ell": PriorFamily.LOG_UNIFORM,
            }
        prior_defs[prior_name] = PriorSpec(
            name=prior_name,
            ranges=ranges,
            delta_range=(args.delta_min, args.delta_max),
            families=families,
            seed=args.seed,
        )

    delta_grid = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    beta_alpha_required = worst_case_beta_alpha_required(ranges, delta_grid)
    beta_omega_required = worst_case_beta_omega_required(ranges, range(ranges.n_agents[0], ranges.n_agents[1] + 1))

    runs: dict[str, dict] = {}
    for prior_name, prior in prior_defs.items():
        metrics = run_metrics(prior=prior, n_samples=args.n_samples)
        sobol = {"n_base": 0, "parameter_names": [], "output_names": [], "S_first": [], "S_total": []}
        if not args.skip_sobol:
            sobol = sobol_indices(prior=prior, n_base=args.sobol_n_base, seed=args.seed + 1)
        run_payload: dict = {
            "metrics": metrics,
            "sobol": sobol,
        }
        if prior_name == args.prior[0]:
            run_payload["volume_by_N"] = volume_metrics_by_fixed_N(
                prior=prior,
                n_samples=max(50_000, args.n_samples // 4),
                seed=args.seed + 3,
            )
        if not args.skip_sobol and prior_name == args.prior[0]:
            # Per-N Sobol view for the default prior only.
            n_values = [int(tok) for tok in args.sobol_per_n.split(",") if tok.strip()]
            sobol_per_n: dict[str, dict] = {}
            for n_val in n_values:
                fixed_ranges = ParamRanges(
                    beta_kappa=ranges.beta_kappa,
                    beta_alpha=ranges.beta_alpha,
                    beta_D=ranges.beta_D,
                    beta_Omega=ranges.beta_Omega,
                    beta_ell=ranges.beta_ell,
                    n_agents=(n_val, n_val),
                )
                prior_n = PriorSpec(
                    name=f"{prior.name}_N{n_val}",
                    ranges=fixed_ranges,
                    delta_range=prior.delta_range,
                    families=prior.families,
                    seed=prior.seed,
                )
                sobol_per_n[str(n_val)] = sobol_indices(prior=prior_n, n_base=max(1024, args.sobol_n_base // 2), seed=args.seed + 2)
            run_payload["sobol_per_N"] = sobol_per_n
        runs[prior_name] = run_payload

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "cwd": os.getcwd(),
            "default_prior": args.prior[0],
        },
        "worst_case": {
            "beta_alpha_required_by_delta": {f"{k:.1f}": v for k, v in beta_alpha_required.items()},
            "beta_omega_required_by_N": {str(k): v for k, v in beta_omega_required.items()},
        },
        "runs": runs,
    }

    _write_json(out_dir / "RESULTS.json", payload)
    _write_md(out_dir / "RESULTS.md", payload)
    for prior_name, prior in prior_defs.items():
        (out_dir / f"PRIOR_{prior_name}.md").write_text(_prior_markdown(prior), encoding="utf-8")


if __name__ == "__main__":
    main()
