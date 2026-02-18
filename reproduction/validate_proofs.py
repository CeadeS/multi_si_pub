#!/usr/bin/env python3
"""Validate core equation statements referenced in the proofs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from repro.model import (
    ModelParams,
    c1_margin,
    c2_margin,
    critical_width_90_percent,
    delta_crit,
    gap_from_patience_free,
    group_value_parabolic,
    is_patience_free,
    optimal_group_size,
    patience_effectiveness,
    phase_transition_jump,
    validate_params,
)


def _near(value: float, target: float, tol: float = 1e-6) -> bool:
    return abs(value - target) <= tol


def validate() -> Dict[str, object]:
    checks: List[str] = []
    warnings: List[str] = []

    # C1* and C2* margins (definition-level checks).
    params = ModelParams(
        beta_kappa=1.0,
        beta_alpha=0.7,
        beta_D=0.4,
        beta_Omega=1.0,
        beta_ell=1.5,
        G0=1.0,
    )
    if _near(c1_margin(params), 1.3):
        checks.append("C1* margin equals beta_alpha + beta_kappa - beta_D")
    else:
        warnings.append(f"C1* margin unexpected: {c1_margin(params):.3f}")

    if _near(c2_margin(params, 5), 0.7):
        checks.append("C2* margin equals beta_Omega - beta_ell/N at N=5")
    else:
        warnings.append(f"C2* margin unexpected: {c2_margin(params, 5):.3f}")

    # C1** discount threshold formula and patience-free boundary.
    g_positive = 0.3
    params_positive = ModelParams(
        beta_kappa=0.8,
        beta_alpha=0.2,
        beta_D=0.5,
        beta_Omega=1.0,
        beta_ell=1.0,
        G0=1.0,
    )
    g_calc = gap_from_patience_free(params_positive)
    expected_delta = g_calc / (g_calc + params_positive.beta_kappa)
    if _near(delta_crit(params_positive), expected_delta):
        checks.append("C1** threshold matches g/(g+beta_kappa) when g>0")
    else:
        warnings.append("C1** threshold mismatch for g>0")

    params_free = ModelParams(
        beta_kappa=0.5,
        beta_alpha=0.8,
        beta_D=0.4,
        beta_Omega=1.0,
        beta_ell=1.0,
        G0=1.0,
    )
    if _near(delta_crit(params_free), 0.0) and is_patience_free(params_free):
        checks.append("Patience-free regime: beta_alpha >= beta_D implies delta* = 0")
    else:
        warnings.append("Patience-free boundary check failed")

    # Phase transition derivative jump at g -> 0+.
    eps = 1e-6
    beta_kappa = 0.9
    params_eps = ModelParams(
        beta_kappa=beta_kappa,
        beta_alpha=0.5,
        beta_D=0.5 + eps,
        beta_Omega=1.0,
        beta_ell=1.0,
        G0=1.0,
    )
    derivative_right = delta_crit(params_eps) / eps
    if abs(derivative_right - (1.0 / beta_kappa)) < 1e-3:
        checks.append("Phase transition jump matches 1/beta_kappa")
    else:
        warnings.append(f"Phase transition jump off: {derivative_right:.3f}")

    # Diminishing returns ratio (first unit vs 10g).
    g = g_positive
    params_k0 = ModelParams(
        beta_kappa=0.0,
        beta_alpha=0.2,
        beta_D=0.5,
        beta_Omega=1.0,
        beta_ell=1.0,
        G0=1.0,
    )
    params_k10 = ModelParams(
        beta_kappa=10.0 * g,
        beta_alpha=0.2,
        beta_D=0.5,
        beta_Omega=1.0,
        beta_ell=1.0,
        G0=1.0,
    )
    eff0 = patience_effectiveness(params_k0)
    eff10 = patience_effectiveness(params_k10)
    if eff10 > 0 and abs((eff0 / eff10) - 121.0) < 1e-1:
        checks.append("Deterrence effectiveness ratio ≈ 121× at beta_kappa=0 vs 10g")
    else:
        warnings.append(f"Deterrence ratio unexpected: {eff0 / eff10:.1f}×")

    # Critical width for transition (90% drop).
    width = critical_width_90_percent(params_positive)
    if _near(width, params_positive.beta_kappa / 9.0):
        checks.append("Critical width matches beta_kappa/9 definition")
    else:
        warnings.append(f"Critical width mismatch: {width:.3f}")

    # Parabolic group value optimum.
    f = 5.0
    c = 1.0
    n_star = optimal_group_size(f, c)
    if _near(n_star, 5.0):
        checks.append("Parabolic optimum N* = f/c")
    else:
        warnings.append(f"Parabolic optimum mismatch: {n_star:.3f}")

    if group_value_parabolic(11, f, c) < 0:
        checks.append("Parabolic value negative beyond N > 2f/c")
    else:
        warnings.append("Parabolic negativity check failed")

    # Information-theoretic limit validation.
    try:
        validate_params(params_free)
        checks.append("Information-theoretic limits enforced (beta_alpha <= 1)")
    except ValueError as exc:
        warnings.append(f"Information-theoretic check failed: {exc}")

    summary = {
        "delta_star_example": delta_crit(params_positive),
        "delta_star_patience_free": delta_crit(params_free),
        "phase_jump": derivative_right,
        "deterrence_ratio": eff0 / eff10,
        "critical_width": width,
        "n_star": n_star,
    }

    return {
        "checks": checks,
        "warnings": warnings,
        "summary": summary,
    }


def write_report(report: Dict[str, object]) -> Path:
    out_path = Path(__file__).resolve().parent / "documentation" / "PROOF_VALIDATION.md"
    lines: List[str] = []
    lines.append("# Proof Validation Report")
    lines.append("")
    lines.append("This report validates numeric identities used in the proofs.")
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
    lines.append(f"- Example delta*: {summary['delta_star_example']:.3f}")
    lines.append(f"- Patience-free delta*: {summary['delta_star_patience_free']:.3f}")
    lines.append(f"- Phase jump (approx): {summary['phase_jump']:.3f}")
    lines.append(f"- Deterrence ratio: {summary['deterrence_ratio']:.1f}×")
    lines.append(f"- Critical width: {summary['critical_width']:.3f}")
    lines.append(f"- Parabolic N*: {summary['n_star']:.3f}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    report = validate()
    path = write_report(report)
    print(path)


if __name__ == "__main__":
    main()
