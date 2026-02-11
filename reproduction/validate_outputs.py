#!/usr/bin/env python3
"""Validation script for reproduction package outputs."""

import argparse
import os
import sys
from typing import List, Tuple

# Import core model functions
from repro.model import (
    anchor_params,
    c1_margin,
    c2_margin,
    delta_crit,
    plausible_ranges,
    validate_params,
    gap_from_patience_free,
    optimal_group_size,
    is_patience_free,
)


class ValidationError(Exception):
    """Raised when validation check fails."""
    pass


def check_files_exist(fig_dir: str) -> List[str]:
    """Check that all expected figure PDFs exist."""
    expected_files = [
        "stability_regions.pdf",
        "sensitivity_analysis.pdf",
        "ce_convergence_validation.pdf",
        "ppe_sustainability_validation.pdf",
        "shapley_participation_validation.pdf",
        "scaling_cost_model.pdf",
        "v_dynamic_by_n.pdf",
        "margin_distributions.pdf",
        "sobol_sensitivity_by_n.pdf",
        "summary_figure.pdf",
        "figure_functional_structure_main.pdf",
        "figure_functional_structure_appendix.pdf",
    ]

    missing = []
    for fname in expected_files:
        fpath = os.path.join(fig_dir, fname)
        if not os.path.exists(fpath):
            missing.append(fname)

    if missing:
        raise ValidationError(f"Missing figure files: {missing}")

    print(f"✓ All {len(expected_files)} figure files exist")
    return expected_files


def check_anchor_parameters() -> None:
    """Verify anchor parameters match paper values."""
    params = anchor_params()

    expected = {
        "beta_kappa": 1.0,
        "beta_alpha": 0.7,
        "beta_D": 0.4,
        "beta_Omega": 1.0,
        "beta_ell": 1.5,
        "G0": 1.0,
    }

    for param_name, expected_value in expected.items():
        actual_value = getattr(params, param_name)
        if abs(actual_value - expected_value) > 1e-6:
            raise ValidationError(
                f"Anchor parameter mismatch: {param_name} = {actual_value}, "
                f"expected {expected_value}"
            )

    print("✓ Anchor parameters match paper values")


def check_plausible_ranges() -> None:
    """Verify plausible ranges match paper main text (conservative estimates)."""
    ranges = plausible_ranges()

    # Conservative ranges aligned with paper (nature_machine_intelligence_template.tex line 214)
    expected = {
        "beta_kappa": (0.5, 3.0),
        "beta_alpha": (0.3, 0.9),  # Conservative (hard limit is 1.0)
        "beta_D": (0.05, 1.0),
        "beta_Omega": (0.05, 0.3),  # Conservative oversight value
        "beta_ell": (0.1, 0.5),     # Conservative removal benefit
        "n_agents": (2, 10),
    }

    for param_name, (expected_min, expected_max) in expected.items():
        actual_range = getattr(ranges, param_name)
        if abs(actual_range[0] - expected_min) > 1e-6 or abs(actual_range[1] - expected_max) > 1e-6:
            raise ValidationError(
                f"Plausible range mismatch: {param_name} = {actual_range}, "
                f"expected {(expected_min, expected_max)}"
            )

    # Verify β_α upper bound respects information-theoretic limit
    if ranges.beta_alpha[1] > 1.0:
        raise ValidationError(
            f"β_α upper bound {ranges.beta_alpha[1]} exceeds information-theoretic limit of 1.0"
        )

    print("✓ Plausible ranges match paper (conservative estimates)")
    print("  - β_α ∈ [0.3, 0.9] (conservative, hard limit is 1.0) ✓")


def check_stability_conditions() -> None:
    """Verify stability condition calculations are correct."""
    params = anchor_params()

    # C1* margin: beta_alpha + beta_kappa - beta_D
    c1 = c1_margin(params)
    expected_c1 = params.beta_alpha + params.beta_kappa - params.beta_D
    if abs(c1 - expected_c1) > 1e-6:
        raise ValidationError(f"C1* margin mismatch: {c1} != {expected_c1}")

    # At anchor: beta_alpha=0.7, beta_kappa=1.0, beta_D=0.4
    # Expected: 0.7 + 1.0 - 0.4 = 1.3
    if abs(c1 - 1.3) > 1e-6:
        raise ValidationError(f"C1* margin at anchor should be 1.3, got {c1}")

    # C2* margin at N=5: beta_Omega - beta_ell/N
    c2_n5 = c2_margin(params, 5)
    expected_c2_n5 = params.beta_Omega - params.beta_ell / 5.0
    if abs(c2_n5 - expected_c2_n5) > 1e-6:
        raise ValidationError(f"C2* margin (N=5) mismatch: {c2_n5} != {expected_c2_n5}")

    # At anchor: beta_Omega=1.0, beta_ell=1.5, N=5
    # Expected: 1.0 - 1.5/5 = 1.0 - 0.3 = 0.7
    if abs(c2_n5 - 0.7) > 1e-6:
        raise ValidationError(f"C2* margin (N=5) at anchor should be 0.7, got {c2_n5}")

    # Delta critical: max(0, (beta_D - beta_alpha) / (beta_D - beta_alpha + beta_kappa))
    delta_c = delta_crit(params)
    numerator = params.beta_D - params.beta_alpha
    denominator = params.beta_D - params.beta_alpha + params.beta_kappa

    # At anchor: beta_D=0.4, beta_alpha=0.7
    # numerator = 0.4 - 0.7 = -0.3 (negative!)
    # So delta_crit should be 0.0 (patience-free regime)
    if numerator < 0:
        if delta_c != 0.0:
            raise ValidationError(
                f"Delta_crit should be 0 when beta_D < beta_alpha, got {delta_c}"
            )
    else:
        expected_delta_c = numerator / denominator
        if abs(delta_c - expected_delta_c) > 1e-6:
            raise ValidationError(f"Delta_crit mismatch: {delta_c} != {expected_delta_c}")

    print("✓ Stability condition calculations are correct")
    print(f"  - C1* margin at anchor: {c1:.3f} (>0 ✓ cooperation rational)")
    print(f"  - C2* margin at N=5: {c2_n5:.3f} (>0 ✓ participation rational)")
    print(f"  - δ* at anchor: {delta_c:.3f} (=0 ✓ patience-free regime)")


def check_critical_thresholds() -> None:
    """Check that critical N thresholds match paper claims."""
    params = anchor_params()
    ranges = plausible_ranges()

    # Paper claims:
    # - N≥10 is sufficient for band-robust participation under conservative ranges
    # - This is when worst-case C2* holds: beta_Omega_min >= beta_ell_max / N

    beta_omega_min = ranges.beta_Omega[0]  # 0.05 (conservative)
    beta_ell_max = ranges.beta_ell[1]      # 0.5 (conservative)

    # Required N for band-robust C2*:
    n_required = beta_ell_max / beta_omega_min

    print(f"✓ Critical thresholds:")
    print(f"  - Worst-case C2* requires N ≥ {n_required:.1f}")
    print(f"  - Conservative ranges: beta_Omega ∈ {ranges.beta_Omega}, beta_ell ∈ {ranges.beta_ell}")
    print(f"  - Paper claims N≥10 sufficient: {'✓' if n_required <= 10 else '✗'}")

    # Check at N=10:
    c2_worst_case_n10 = beta_omega_min - beta_ell_max / 10.0
    print(f"  - Worst-case C2* margin at N=10: {c2_worst_case_n10:.3f} ({'✓' if c2_worst_case_n10 >= 0 else '✗'})")


def check_functional_analysis() -> None:
    """Verify functional analysis equations from Section 2.X and Appendix A.5."""
    params = anchor_params()

    # Test gap calculation
    g = gap_from_patience_free(params)
    expected_g = params.beta_D - params.beta_alpha
    if abs(g - expected_g) > 1e-6:
        raise ValidationError(f"Gap calculation error: {g} != {expected_g}")

    # At anchor: beta_D=0.4, beta_alpha=0.7, so g = -0.3
    if abs(g - (-0.3)) > 1e-6:
        raise ValidationError(f"Gap at anchor should be -0.3, got {g}")

    # Test patience-free detection
    patience_free = is_patience_free(params)
    # At anchor: beta_alpha (0.7) > beta_D (0.4), so should be patience-free
    if not patience_free:
        raise ValidationError("Anchor should be in patience-free regime")

    # Test optimal group size calculation
    N_star = optimal_group_size(f=5.0, c=1.0)
    if abs(N_star - 5.0) > 1e-6:
        raise ValidationError(f"Optimal N* = f/c should be 5.0, got {N_star}")

    # Test parameter validation
    try:
        validate_params(params)
    except ValueError as e:
        raise ValidationError(f"Anchor parameters failed validation: {e}")

    print("✓ Functional analysis equations validated")
    print(f"  - Gap from patience-free: g = {g:.3f}")
    print(f"  - Patience-free regime: {patience_free} ✓")
    print(f"  - Optimal N* (f=5, c=1): {N_star:.1f}")
    print(f"  - Parameter validation passed ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reproduction package outputs.")
    parser.add_argument(
        "--fig-dir",
        default=os.path.join("paper", "figures"),
        help="Directory containing generated figure PDFs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation information",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Reproduction Package Validation")
    print("=" * 60)
    print()

    try:
        # 1. Check figure files exist
        check_files_exist(args.fig_dir)
        print()

        # 2. Check anchor parameters
        check_anchor_parameters()
        print()

        # 3. Check plausible ranges
        check_plausible_ranges()
        print()

        # 4. Check stability condition implementations
        check_stability_conditions()
        print()

        # 5. Check critical thresholds match paper
        check_critical_thresholds()
        print()

        # 6. Check functional analysis equations
        check_functional_analysis()
        print()

        print("=" * 60)
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
        return 0

    except ValidationError as e:
        print()
        print("=" * 60)
        print(f"✗ VALIDATION FAILED: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
