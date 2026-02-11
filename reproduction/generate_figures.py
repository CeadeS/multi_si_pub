#!/usr/bin/env python3
"""Generate all figures for paper/main.tex."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from repro.figures import (
    figure_ce_convergence,
    figure_ppe_sustainability,
    figure_scaling_cost_model,
    figure_sensitivity_analysis,
    figure_shapley_participation,
    figure_stability_regions,
    figure_v_dynamic_by_n,
    figure_margin_distributions,
    figure_sobol_by_n,
    figure_summary,
)
from repro.functional_figures import (
    generate_functional_structure_main,
    generate_functional_structure_appendix,
)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate reproduction figures.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("paper", "figures"),
        help="Output directory for PDF figures (default: paper/figures)",
    )
    parser.add_argument(
        "--results",
        default=os.path.join(
            script_dir,
            "documentation",
            "revision_experiments_final",
            "RESULTS.json",
        ),
        help="Path to revision experiments results JSON",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.abspath(args.results)

    print("Generating core validation figures...")
    figure_stability_regions(os.path.join(out_dir, "stability_regions.pdf"))
    figure_sensitivity_analysis(os.path.join(out_dir, "sensitivity_analysis.pdf"))
    figure_ce_convergence(os.path.join(out_dir, "ce_convergence_validation.pdf"))
    figure_ppe_sustainability(os.path.join(out_dir, "ppe_sustainability_validation.pdf"))
    figure_shapley_participation(os.path.join(out_dir, "shapley_participation_validation.pdf"))
    figure_scaling_cost_model(os.path.join(out_dir, "scaling_cost_model.pdf"))

    print("Generating robustness analysis figures...")
    figure_v_dynamic_by_n(os.path.join(out_dir, "v_dynamic_by_n.pdf"), results_path)
    figure_margin_distributions(os.path.join(out_dir, "margin_distributions.pdf"), results_path)
    figure_sobol_by_n(os.path.join(out_dir, "sobol_sensitivity_by_n.pdf"), results_path)

    print("Generating summary figure...")
    figure_summary(os.path.join(out_dir, "summary_figure.pdf"), results_path)

    print("Generating functional structure figures...")
    generate_functional_structure_main(out_dir)
    generate_functional_structure_appendix(out_dir)

    print(f"\nAll figures saved to {out_dir}/")
    print("\nFigures generated:")
    print("  Core validation (6 figures)")
    print("  Robustness analysis (4 figures)")
    print("  Functional structure (2 figures)")
    print("  Total: 12 figures")


if __name__ == "__main__":
    main()
