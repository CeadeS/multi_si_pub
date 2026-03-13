#!/usr/bin/env python3
"""Validate numeric invariants behind the main figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from dataclasses import asdict

import numpy as np

from repro.model import anchor_params, c1_margin, c2_margin, plausible_ranges
from repro.cost_model import default_benefit_params, default_cost_params, net_values


def _load_results() -> dict:
    results_path = Path(__file__).resolve().parent / "documentation" / "revision_experiments_final" / "RESULTS.json"
    return json.loads(results_path.read_text(encoding="utf-8"))


def _top_driver_s_dynamic(sobol: dict, n_key: str) -> str:
    sobol_n = sobol[n_key]
    params = sobol_n["parameter_names"]
    outputs = sobol_n["output_names"]
    st = np.asarray(sobol_n["S_total"], dtype=float)
    idx = outputs.index("s_dynamic")
    order = np.argsort(-st[:, idx])
    return params[int(order[0])]


def validate() -> Dict[str, object]:
    results = _load_results()
    uniform = results["runs"]["uniform"]
    volume_by_n = uniform["volume_by_N"]
    sobol_by_n = uniform["sobol_per_N"]

    n_values = sorted(int(n) for n in volume_by_n.keys())
    v_dynamic = np.array([volume_by_n[str(n)]["V_dynamic"] for n in n_values], dtype=float)
    v_c2 = np.array([volume_by_n[str(n)]["V_C2"] for n in n_values], dtype=float)

    checks: List[str] = []
    warnings: List[str] = []

    # Figure: v_dynamic_by_n
    if np.all(np.diff(v_dynamic) >= -1e-9):
        checks.append("V_dynamic nondecreasing with N")
    else:
        warnings.append("V_dynamic is not monotonic nondecreasing")

    n4_idx = n_values.index(4) if 4 in n_values else None
    if n4_idx is not None and v_dynamic[n4_idx] >= 0.95:
        checks.append("V_dynamic crosses 95% at N=4")
    else:
        warnings.append("V_dynamic does not reach 95% at N=4")

    n6_idx = n_values.index(6) if 6 in n_values else None
    if n6_idx is not None and v_dynamic[n6_idx] >= 0.97:
        checks.append("V_dynamic ≥ 0.97 at N=6")
    else:
        warnings.append("V_dynamic < 0.97 at N=6")

    if np.all(v_dynamic <= v_c2 + 1e-9):
        checks.append("V_dynamic ≤ V_C2 for all N")
    else:
        warnings.append("V_dynamic exceeds V_C2 for some N")

    # Figure: sobol_sensitivity_by_n
    driver_n2 = _top_driver_s_dynamic(sobol_by_n, "2")
    driver_n10 = _top_driver_s_dynamic(sobol_by_n, "10")
    if driver_n2 == "beta_Omega":
        checks.append("Sobol: beta_Omega dominates s_dynamic at N=2")
    else:
        warnings.append(f"Sobol: expected beta_Omega at N=2, got {driver_n2}")

    if driver_n10 == "delta":
        checks.append("Sobol: delta dominates s_dynamic at N=10")
    else:
        warnings.append(f"Sobol: expected delta at N=10, got {driver_n10}")

    # Figure: stability_regions / sensitivity anchor checks
    params = anchor_params()
    c1 = c1_margin(params)
    c2 = c2_margin(params, 5)
    if c1 > 0 and abs(c1 - 1.3) < 1e-6:
        checks.append("Anchor C1* margin = 1.3 (>0)")
    else:
        warnings.append(f"Anchor C1* margin unexpected: {c1:.3f}")

    if c2 > 0 and abs(c2 - 0.7) < 1e-6:
        checks.append("Anchor C2* margin (N=5) = 0.7 (>0)")
    else:
        warnings.append(f"Anchor C2* margin unexpected: {c2:.3f}")

    # Figure: scaling_cost_model (record optima)
    cost_params = default_cost_params()
    benefit_params = default_benefit_params()
    n_grid = np.arange(2, 21)
    net_saturating = np.array([net_values(int(n), cost_params, benefit_params)["saturating"] for n in n_grid])
    net_linear = np.array([net_values(int(n), cost_params, benefit_params)["linear"] for n in n_grid])
    net_network = np.array([net_values(int(n), cost_params, benefit_params)["network"] for n in n_grid])
    n_star_sat = int(n_grid[int(np.argmax(net_saturating))])
    n_star_lin = int(n_grid[int(np.argmax(net_linear))])
    n_star_net = int(n_grid[int(np.argmax(net_network))])

    figure2 = _validate_figure2()

    summary = {
        "n_values": n_values,
        "v_dynamic": v_dynamic.tolist(),
        "v_c2": v_c2.tolist(),
        "sobol_top_n2": driver_n2,
        "sobol_top_n10": driver_n10,
        "n_star_saturating": n_star_sat,
        "n_star_linear": n_star_lin,
        "n_star_network": n_star_net,
    }

    return {
        "checks": checks,
        "warnings": warnings,
        "summary": summary,
        "figure2": figure2,
    }


def write_report(report: Dict[str, object]) -> Path:
    out_path = Path(__file__).resolve().parent / "documentation" / "FIGURE_VALIDATION.md"
    lines: List[str] = []
    lines.append("# Figure Validation Report")
    lines.append("")
    lines.append("This report validates numeric invariants that underlie the main figures.")
    lines.append("")
    lines.append("## Checks")
    for item in report["checks"]:
        lines.append(f"- ✓ {item}")
    if report["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        for item in report["warnings"]:
            lines.append(f"- ⚠ {item}")
    lines.append("")
    lines.append("## Summary")
    summary = report["summary"]
    lines.append(f"- N grid: {summary['n_values']}")
    lines.append(f"- V_dynamic: {['{:.3f}'.format(v) for v in summary['v_dynamic']]}")
    lines.append(f"- V_C2: {['{:.3f}'.format(v) for v in summary['v_c2']]}")
    lines.append(f"- Sobol top driver at N=2: {summary['sobol_top_n2']}")
    lines.append(f"- Sobol top driver at N=10: {summary['sobol_top_n10']}")
    lines.append(f"- Scaling optima (saturating / linear / network): {summary['n_star_saturating']} / {summary['n_star_linear']} / {summary['n_star_network']}")
    lines.append("")
    lines.append("## Figure 2 Legacy Comparison")
    lines.append("")
    figure2 = report["figure2"]
    lines.append("This section compares the Figure 2 sensitivity margins against the legacy binary stability logic.")
    lines.append("")
    lines.append("### Anchor")
    lines.append(f"- Anchor parameters: {figure2['anchor_params']}")
    lines.append(f"- Anchor stability margin S: {figure2['anchor_margin']:.3f}")
    lines.append("")
    lines.append("### Sweep Comparison")
    lines.append("")
    lines.append("| Parameter | Sign agreement | New crossing | Legacy crossing |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, data in figure2["sweeps"].items():
        new_cross = data["crossing_new"]
        legacy_cross = data["crossing_legacy"]
        def fmt(val: float) -> str:
            return "n/a" if np.isnan(val) else f"{val:.3g}"
        lines.append(
            f"| {key} | {data['sign_agreement']:.3f} | {fmt(new_cross)} | {fmt(legacy_cross)} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Legacy curves are binary (0/1) and reflect C1* plus a Shapley-based C2* check with fixed synergy.")
    lines.append("- Missing legacy panels (beta_ell, N) are reported for sign agreement but should be read as informational only.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _figure2_c1_margin(beta_alpha: float, beta_kappa: float, beta_D: float) -> float:
    return beta_alpha + beta_kappa - beta_D


def _figure2_c2_margin(beta_Omega: float, beta_ell: float, n_agents: int) -> float:
    return beta_Omega - beta_ell / float(n_agents)


def _figure2_stability_margin(
    beta_alpha: float,
    beta_kappa: float,
    beta_D: float,
    beta_Omega: float,
    beta_ell: float,
    n_agents: int,
) -> float:
    return min(
        _figure2_c1_margin(beta_alpha, beta_kappa, beta_D),
        _figure2_c2_margin(beta_Omega, beta_ell, n_agents),
    )


def _figure2_legacy_stable_score(
    beta_alpha: float,
    beta_kappa: float,
    beta_D: float,
    beta_Omega: float,
    beta_ell: float,
    n_agents: int,
) -> float:
    beta_synergy = 0.4
    shapley_per_agent = beta_Omega + beta_synergy * (n_agents - 1) / 2.0
    c1 = beta_alpha + beta_kappa >= beta_D
    c2 = shapley_per_agent >= beta_ell / float(n_agents)
    return 1.0 if (c1 and c2) else 0.0


def _figure2_crossing_point(x: np.ndarray, y: np.ndarray) -> float:
    signs = np.sign(y)
    for i in range(1, len(x)):
        if signs[i] == 0:
            return float(x[i])
        if signs[i - 1] == 0:
            return float(x[i - 1])
        if signs[i - 1] * signs[i] < 0:
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]
            return float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    return float("nan")


def _figure2_sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a > 0) == (b > 0.5)))


def _validate_figure2() -> Dict[str, object]:
    params = anchor_params()
    n_agents = 5

    ranges = plausible_ranges()
    beta_kappa_vals = np.logspace(np.log10(ranges.beta_kappa[0]), np.log10(ranges.beta_kappa[1]), 120)
    beta_alpha_vals = np.logspace(np.log10(ranges.beta_alpha[0]), np.log10(ranges.beta_alpha[1]), 120)
    beta_D_vals = np.logspace(np.log10(ranges.beta_D[0]), np.log10(ranges.beta_D[1]), 120)
    beta_Omega_vals = np.logspace(np.log10(ranges.beta_Omega[0]), np.log10(ranges.beta_Omega[1]), 120)
    beta_ell_vals = np.logspace(np.log10(ranges.beta_ell[0]), np.log10(ranges.beta_ell[1]), 120)
    n_vals = np.arange(ranges.n_agents[0], ranges.n_agents[1] + 1)

    def sweep(values: np.ndarray, key: str) -> Tuple[np.ndarray, np.ndarray]:
        s_vals = []
        legacy_vals = []
        for v in values:
            beta_alpha = params.beta_alpha
            beta_kappa = params.beta_kappa
            beta_D = params.beta_D
            beta_Omega = params.beta_Omega
            beta_ell = params.beta_ell
            n = n_agents
            if key == "beta_kappa":
                beta_kappa = float(v)
            elif key == "beta_alpha":
                beta_alpha = float(v)
            elif key == "beta_D":
                beta_D = float(v)
            elif key == "beta_Omega":
                beta_Omega = float(v)
            elif key == "beta_ell":
                beta_ell = float(v)
            elif key == "N":
                n = int(v)
            s_vals.append(_figure2_stability_margin(beta_alpha, beta_kappa, beta_D, beta_Omega, beta_ell, n))
            legacy_vals.append(_figure2_legacy_stable_score(beta_alpha, beta_kappa, beta_D, beta_Omega, beta_ell, n))
        return np.asarray(s_vals), np.asarray(legacy_vals)

    sweeps = {
        "beta_kappa": beta_kappa_vals,
        "beta_alpha": beta_alpha_vals,
        "beta_D": beta_D_vals,
        "beta_Omega": beta_Omega_vals,
        "beta_ell": beta_ell_vals,
        "N": n_vals,
    }

    results: Dict[str, object] = {
        "anchor_params": asdict(params),
        "anchor_margin": _figure2_stability_margin(
            params.beta_alpha,
            params.beta_kappa,
            params.beta_D,
            params.beta_Omega,
            params.beta_ell,
            n_agents,
        ),
        "sweeps": {},
    }

    for key, values in sweeps.items():
        s_vals, legacy_vals = sweep(values, key)
        results["sweeps"][key] = {
            "sign_agreement": _figure2_sign_agreement(s_vals, legacy_vals),
            "crossing_new": _figure2_crossing_point(values.astype(float), s_vals),
            "crossing_legacy": _figure2_crossing_point(values.astype(float), legacy_vals - 0.5),
        }

    return results


def main() -> None:
    report = validate()
    path = write_report(report)
    print(path)


if __name__ == "__main__":
    main()
