"""Functional structure figures for Section 2.X and Appendix A.5."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from pathlib import Path

from .model import ModelParams, anchor_params, plausible_ranges


def generate_functional_structure_main(out_dir: str = "figures") -> None:
    """Generate 2-panel figure showing functional structure (main text).

    Panel A: Phase diagram (β_α vs β_D showing regimes)
    Panel B: Functional forms overlay (normalized)

    Saves to: {out_dir}/figure_functional_structure_main.pdf
    """
    # Set publication-quality defaults
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.figsize'] = (12, 5)

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ========================================================================
    # PANEL A: Phase Diagram
    # ========================================================================

    # Axes ranges
    beta_D_range = np.linspace(0, 1.5, 300)
    beta_alpha_range = np.linspace(0, 1.2, 300)

    # Define regions
    beta_kappa_values = [0, 0.3, 0.6]
    colors_kappa = ['red', 'orange', 'purple']
    labels_kappa = [r'$\beta_\kappa = 0$', r'$\beta_\kappa = 0.3$', r'$\beta_\kappa = 0.6$']

    # Fill patience-free region (green)
    ax1.fill_between(beta_D_range, beta_D_range, 1.2,
                      alpha=0.3, color='green', label='Patience-free')

    # Fill patience-required region (for β_κ = 0.3 as example)
    beta_kappa_example = 0.3
    patience_req_lower = beta_D_range
    patience_req_upper = np.maximum(beta_D_range - beta_kappa_example, patience_req_lower)
    ax1.fill_between(beta_D_range, patience_req_lower, patience_req_upper,
                      where=(patience_req_lower < patience_req_upper),
                      alpha=0.3, color='yellow', label=f'Patience-required (β_κ={beta_kappa_example})')

    # Fill infeasible region (for β_κ = 0.3)
    ax1.fill_between(beta_D_range, 0, patience_req_upper,
                      alpha=0.3, color='red', label='Infeasible')

    # Draw C1* boundaries for different β_κ
    for bk, color, label in zip(beta_kappa_values, colors_kappa, labels_kappa):
        beta_alpha_boundary = beta_D_range - bk
        beta_alpha_boundary = np.clip(beta_alpha_boundary, 0, 1.2)
        ax1.plot(beta_D_range, beta_alpha_boundary, '--', color=color,
                 linewidth=2, label=label + ' (C1*)')

    # Draw patience-free boundary (diagonal)
    ax1.plot(beta_D_range, beta_D_range, 'k-', linewidth=3,
             label=r'$\beta_\alpha = \beta_D$ (phase transition)')

    # Draw information-theoretic limit
    ax1.axhline(y=1.0, color='blue', linestyle=':', linewidth=2,
                label=r'$\beta_\alpha = 1$ (info limit)')

    # Mark typical operating point
    params = anchor_params()
    ax1.plot(params.beta_D, params.beta_alpha, 'ko', markersize=10,
             markerfacecolor='white', markeredgewidth=2,
             label='Typical operating point')

    # Labels and formatting
    ax1.set_xlabel(r'$\beta_D$ (Defection Temptation)', fontsize=11)
    ax1.set_ylabel(r'$\beta_\alpha$ (Coordination Quality)', fontsize=11)
    ax1.set_title('(A) Phase Diagram', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Add annotations
    ax1.annotate('Patience-Free\n(Structurally Stable)',
                 xy=(0.3, 0.8), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    ax1.annotate('Patience-Required\n(Psychology-Dependent)',
                 xy=(0.8, 0.5), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    ax1.annotate('Infeasible',
                 xy=(1.2, 0.2), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

    # ========================================================================
    # PANEL B: Functional Forms Overlay (Normalized)
    # ========================================================================

    # X-axis: normalized control variable [0, 1]
    x_norm = np.linspace(0, 1, 300)

    # Hyperbolic: δ* = g/(g+β_κ)
    g = 1.0
    beta_kappa_norm = x_norm * 3 * g
    delta_star_hyp = g / (g + beta_kappa_norm)

    # Logarithmic: β_α ~ log(1+SNR)
    SNR_max = 20
    SNR_norm = x_norm * SNR_max
    beta_alpha_log = np.log2(1 + SNR_norm) / np.log2(1 + SNR_max)

    # Parabolic: V(N) = (N-1)f - cN²/2
    N_norm = 1 + x_norm * 9
    f_over_c = 5.0
    V_parab = (N_norm - 1) * f_over_c - 0.5 * N_norm**2
    V_max = np.max(V_parab)
    V_min = np.min(V_parab)
    V_parab_norm = (V_parab - V_min) / (V_max - V_min)

    # Plot curves
    ax2.plot(x_norm, delta_star_hyp, 'r-', linewidth=2.5,
             label=r'Hyperbolic: $\delta^* = g/(g+\beta_\kappa)$ (deterrence)')
    ax2.plot(x_norm, beta_alpha_log, 'b--', linewidth=2.5,
             label=r'Logarithmic: $\beta_\alpha \sim \log(1+\mathrm{SNR})$ (coordination)')
    ax2.plot(x_norm, V_parab_norm, 'g:', linewidth=2.5,
             label=r'Parabolic: $V(N) = fN - cN^2/2$ (group size)')

    # Mark key points
    x_g = 1.0 / 3.0
    y_g = g / (g + g)
    ax2.plot(x_g, y_g, 'ro', markersize=8, label=r'$\beta_\kappa = g$ (4× drop)')

    # Parabolic optimum
    x_opt = (f_over_c - 1) / 9.0
    ax2.plot(x_opt, 1.0, 'go', markersize=8, label=r'$N^* = f/c$ (optimum)')

    # Saturation line
    ax2.axhline(y=1.0, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    ax2.text(0.5, 1.05, 'Saturation limit', ha='center', fontsize=9, color='blue')

    # Labels and formatting
    ax2.set_xlabel('Normalized Control Variable', fontsize=11)
    ax2.set_ylabel('Normalized Response', fontsize=11)
    ax2.set_title('(B) Three Functional Forms', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Add annotations for functional properties
    ax2.annotate('Convex\n(Dim. Returns)',
                 xy=(0.6, 0.4), xytext=(0.75, 0.25),
                 fontsize=8, ha='center', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax2.annotate('Concave\n(Dim. Returns)',
                 xy=(0.7, 0.85), xytext=(0.85, 0.7),
                 fontsize=8, ha='center', color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax2.annotate('Unique\nMaximum',
                 xy=(x_opt, 1.0), xytext=(0.3, 0.85),
                 fontsize=8, ha='center', color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Overall figure adjustments
    plt.tight_layout()

    # Save figure
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "figure_functional_structure_main.pdf"
    png_path = out_path / "figure_functional_structure_main.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"✓ Generated: {pdf_path}")
    plt.close()


def generate_functional_structure_appendix(out_dir: str = "figures") -> None:
    """Generate 4-panel figure showing detailed functional structure (appendix).

    Panel A: Hyperbolic deterrence (δ* vs β_κ)
    Panel B: Logarithmic coordination (β_α vs SNR)
    Panel C: Parabolic group value (V vs N)
    Panel D: Full phase diagram

    Saves to: {out_dir}/figure_functional_structure_appendix.pdf
    """
    # Set publication-quality defaults
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 9
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 11
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 8
    rcParams['figure.figsize'] = (14, 10)

    # Create figure with 2x2 panels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # ========================================================================
    # PANEL A: Hyperbolic Deterrence Effect
    # ========================================================================

    beta_kappa_range = np.linspace(0, 3, 300)

    # Different g values
    g_values = [0.1, 0.3, 0.5]
    colors_g = ['green', 'blue', 'red']
    labels_g = [r'$g = 0.1$ (near patience-free)',
                r'$g = 0.3$ (moderate gap)',
                r'$g = 0.5$ (large gap)']

    for g, color, label in zip(g_values, colors_g, labels_g):
        delta_star = g / (g + beta_kappa_range)
        ax1.plot(beta_kappa_range, delta_star, color=color,
                 linewidth=2.5, label=label)

        # Mark β_κ = g point
        beta_k_at_g = g
        delta_at_g = 0.5
        ax1.plot(beta_k_at_g, delta_at_g, 'o', color=color,
                 markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Add effectiveness annotation
    ax1.annotate(r'Effectiveness: $\varepsilon(\beta_\kappa) = \frac{g}{(g+\beta_\kappa)^2}$',
                 xy=(1.5, 0.85), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

    # Labels and formatting
    ax1.set_xlabel(r'$\beta_\kappa$ (Deterrence)', fontsize=10)
    ax1.set_ylabel(r'$\delta^*$ (Required Patience)', fontsize=10)
    ax1.set_title('(A) Hyperbolic Deterrence Effect', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=8)

    # Add vertical lines at β_κ = g
    for g in g_values:
        ax1.axvline(x=g, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # ========================================================================
    # PANEL B: Logarithmic Coordination Saturation
    # ========================================================================

    SNR_range = np.linspace(0, 20, 300)
    beta_alpha_full = np.log2(1 + SNR_range) / np.log2(1 + 20)

    ax2.plot(SNR_range, beta_alpha_full, 'b-', linewidth=2.5,
             label=r'$\beta_\alpha \sim \log_2(1+\mathrm{SNR})$')

    # Mark saturation line
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label=r'$\beta_\alpha = 1$ (information-theoretic limit)')

    # Fill impossible region
    ax2.fill_between(SNR_range, 1.0, 1.2, alpha=0.3, color='red',
                      label='Impossible Region')

    # Annotations
    ax2.annotate('Concave:\n' + r'$\frac{\partial^2 \beta_\alpha}{\partial \mathrm{SNR}^2} < 0$',
                 xy=(10, 0.7), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    ax2.annotate('Cannot Exceed\n(Info Theory)',
                 xy=(15, 1.1), fontsize=9, ha='center', color='red')

    # Labels and formatting
    ax2.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=10)
    ax2.set_ylabel(r'$\beta_\alpha$ (Coordination Quality)', fontsize=10)
    ax2.set_title('(B) Logarithmic Coordination Saturation', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=8)

    # ========================================================================
    # PANEL C: Parabolic Group Value
    # ========================================================================

    N_range = np.linspace(1, 15, 300)

    # Different f/c ratios
    f_over_c_values = [3, 5, 7]
    colors_fc = ['purple', 'green', 'orange']
    labels_fc = [r'$f/c = 3$ ($N^* = 3$)',
                 r'$f/c = 5$ ($N^* = 5$, empirical)',
                 r'$f/c = 7$ ($N^* = 7$)']

    for fc, color, label in zip(f_over_c_values, colors_fc, labels_fc):
        V_N = (N_range - 1) * fc - 0.5 * N_range**2
        ax3.plot(N_range, V_N, color=color, linewidth=2.5, label=label)

        # Mark optimum
        N_star = fc
        V_star = (N_star - 1) * fc - 0.5 * N_star**2
        ax3.plot(N_star, V_star, 'o', color=color, markersize=10,
                 markerfacecolor='white', markeredgewidth=2)
        ax3.axvline(x=N_star, color=color, linestyle=':', alpha=0.5, linewidth=1)

    # Mark zero-crossing
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Shade negative region
    ax3.fill_between(N_range, -5, 0, alpha=0.2, color='red',
                      label='Negative Value (harmful)')

    # Show empirical observation
    ax3.axvspan(4, 6, alpha=0.2, color='green', label='Empirical $N \\sim 4-6$')

    # Annotations
    ax3.annotate(r'$V(N) = (N-1)f - \frac{cN^2}{2}$' + '\n' +
                 r'Optimum: $\frac{\partial V}{\partial N} = 0 \Rightarrow N^* = \frac{f}{c}$',
                 xy=(10, 5), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

    # Labels and formatting
    ax3.set_xlabel(r'$N$ (Group Size)', fontsize=10)
    ax3.set_ylabel(r'$V(N)$ (Net Value)', fontsize=10)
    ax3.set_title('(C) Parabolic Group Value Function', fontsize=11, fontweight='bold')
    ax3.set_xlim(1, 15)
    ax3.set_ylim(-5, 15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=8)

    # ========================================================================
    # PANEL D: Full Phase Diagram
    # ========================================================================

    beta_D_range_full = np.linspace(0, 1.5, 300)

    # Fill patience-free region
    ax4.fill_between(beta_D_range_full, beta_D_range_full, 1.2,
                      alpha=0.3, color='green', label='Patience-Free')

    # C1* boundaries for different β_κ
    beta_kappa_lines = [0, 0.2, 0.4, 0.6]
    colors_lines = ['red', 'orange', 'purple', 'brown']
    for bk, color in zip(beta_kappa_lines, colors_lines):
        beta_alpha_boundary = beta_D_range_full - bk
        beta_alpha_boundary = np.clip(beta_alpha_boundary, 0, 1.2)
        ax4.plot(beta_D_range_full, beta_alpha_boundary, '--',
                 color=color, linewidth=2,
                 label=f'C1* boundary (β_κ={bk})')

    # Phase transition diagonal
    ax4.plot(beta_D_range_full, beta_D_range_full, 'k-', linewidth=3,
             label=r'$\beta_\alpha = \beta_D$ (1st-order transition)')

    # Information limit
    ax4.axhline(y=1.0, color='blue', linestyle=':', linewidth=2,
                label=r'$\beta_\alpha = 1$ (info limit)')

    # Typical operating point
    params = anchor_params()
    ax4.plot(params.beta_D, params.beta_alpha, 'ko', markersize=12,
             markerfacecolor='gold', markeredgewidth=2,
             label='Typical operating point', zorder=10)

    # Labels and formatting
    ax4.set_xlabel(r'$\beta_D$ (Defection Temptation)', fontsize=10)
    ax4.set_ylabel(r'$\beta_\alpha$ (Coordination Quality)', fontsize=10)
    ax4.set_title('(D) Complete Phase Diagram', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 1.5)
    ax4.set_ylim(0, 1.2)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper left', fontsize=7, framealpha=0.95, ncol=2)

    # Add region labels
    ax4.text(0.3, 0.9, 'Patience-Free\n(Stable)', ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    ax4.text(0.9, 0.6, 'Patience-\nRequired', ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    ax4.text(1.3, 0.2, 'Infeasible', ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

    # Overall figure adjustments
    plt.tight_layout()

    # Save figure
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / "figure_functional_structure_appendix.pdf"
    png_path = out_path / "figure_functional_structure_appendix.png"

    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    print(f"✓ Generated: {pdf_path}")
    plt.close()
