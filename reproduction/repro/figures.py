"""Figure generation for the reproduction package."""

from dataclasses import replace
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .cost_model import (
    benefit_linear,
    benefit_network,
    benefit_saturating,
    cost_comm,
    cost_consensus,
    cost_hedging,
    cost_latency,
    cost_reliability,
    default_benefit_params,
    default_cost_params,
    net_values,
)
from .model import (
    anchor_params,
    coalition_value_by_size,
    delta_crit,
    expected_welfare,
    plausible_ranges,
    replace_param,
    stability_margin,
    symmetric_shapley_per_agent,
)
from .plot_style import apply_style


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _log_pad(min_val: float, max_val: float, factor: float = 1.3) -> tuple[float, float]:
    return min_val / factor, max_val * factor


def _normal_sigma(mean: float, min_val: float, max_val: float) -> float:
    width = min(mean - min_val, max_val - mean)
    return max(width / 2.0, mean * 0.05)


def _lognormal_sigma(mean: float, min_val: float, max_val: float) -> float:
    lo = np.log(mean / min_val)
    hi = np.log(max_val / mean)
    width = min(lo, hi)
    return max(width / 2.0, 0.08)


def _sample_param_range(
    rng: np.random.Generator,
    min_val: float,
    max_val: float,
    mean: float,
    dist: str,
    n_samples: int,
) -> np.ndarray:
    if dist == "uniform":
        return rng.uniform(min_val, max_val, size=n_samples)
    if dist == "log-uniform":
        return 10 ** rng.uniform(np.log10(min_val), np.log10(max_val), size=n_samples)
    if dist == "normal":
        sigma = _normal_sigma(mean, min_val, max_val)
        draws = rng.normal(mean, sigma, size=n_samples)
        return np.clip(draws, min_val, max_val)
    if dist == "log-normal":
        sigma = _lognormal_sigma(mean, min_val, max_val)
        mu = np.log(mean)
        draws = rng.lognormal(mean=mu, sigma=sigma, size=n_samples)
        return np.clip(draws, min_val, max_val)
    raise ValueError(f"Unsupported distribution: {dist}")


def figure_stability_regions(path: str) -> None:
    """Stability region for C1* (beta_alpha + beta_kappa >= beta_D)."""

    apply_style()
    params = anchor_params()
    ranges = plausible_ranges()
    kappa_plaus_min, kappa_plaus_max = ranges.beta_kappa
    d_plaus_min, d_plaus_max = ranges.beta_D
    alpha_plaus_min, alpha_plaus_max = ranges.beta_alpha

    kappa_min, kappa_max = _log_pad(kappa_plaus_min, kappa_plaus_max, factor=2.0)
    d_min, d_max = _log_pad(d_plaus_min, d_plaus_max, factor=2.0)
    d_max = max(d_max, params.beta_alpha + kappa_max)

    beta_kappa = np.logspace(np.log10(kappa_min), np.log10(kappa_max), 240)
    beta_D = np.logspace(np.log10(d_min), np.log10(d_max), 240)

    margin_value = params.beta_alpha + params.beta_kappa - params.beta_D
    margin_top = params.beta_D + margin_value

    cmap = LinearSegmentedColormap.from_list("stability", ["#f4b6b6", "#b7e3b3"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): stability region in (beta_kappa, beta_D) for anchored beta_alpha.
    ax = axes[0]
    grid_kappa, grid_D = np.meshgrid(beta_kappa, beta_D)
    margin = params.beta_alpha + grid_kappa - grid_D
    margin_min = float(np.min(margin))
    margin_max = float(np.max(margin))
    stable_color = "#b7e3b3"
    unstable_color = "#f4b6b6"
    ax.contourf(
        grid_kappa,
        grid_D,
        margin,
        levels=[margin_min, 0.0, margin_max],
        colors=[unstable_color, stable_color],
        alpha=0.85,
        extend="both",
    )
    ax.contour(grid_kappa, grid_D, margin, levels=[0.0], colors="black", linewidths=1.2)

    boundary = params.beta_alpha + beta_kappa
    ax.plot(
        beta_kappa,
        boundary,
        color="black",
        linewidth=1.4,
        label=r"C1* indifference ($\beta_D = \beta_\alpha + \beta_\kappa$)",
    )

    # Patience-free boundary: where β_α ≥ β_D (δ*=0)
    patience_free_boundary = 0.7  # Mid-range plausible value
    ax.axhline(
        patience_free_boundary,
        color="#10b981",
        linewidth=2.5,
        linestyle="--",
        label=r"Patience-free boundary ($\delta^*=0$)",
        zorder=10,
    )
    # Shade region where δ*=0 (below the boundary)
    ax.axhspan(
        d_min,
        patience_free_boundary,
        color="#10b981",
        alpha=0.15,
        zorder=0,
    )
    ax.text(
        0.72,
        0.25,
        r"Patience-free regime ($\delta^*=0$)",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#047857",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.text(
        0.06,
        0.9,
        "C1* violated\n(defect rational)",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#b91c1c",
        va="top",
    )
    ax.text(
        0.62,
        0.18,
        "C1* holds\n(cooperate rational)",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#166534",
        va="bottom",
    )
    plausible_box = Rectangle(
        (kappa_plaus_min, d_plaus_min),
        kappa_plaus_max - kappa_plaus_min,
        d_plaus_max - d_plaus_min,
        facecolor="#fef3c7",
        edgecolor="#f59e0b",
        linewidth=1.0,
        alpha=0.25,
        label="Plausible range",
    )
    ax.add_patch(plausible_box)

    # Plot anchor point as red star
    ax.plot(
        params.beta_kappa,
        params.beta_D,
        marker="*",
        color="red",
        markersize=15,
        markeredgecolor="darkred",
        markeredgewidth=0.8,
        zorder=15,
        label="Anchor point",
    )

    ax.set_xlabel(r"$\beta_\kappa$ (deterrence strength)")
    ax.set_ylabel(r"$\beta_D$ (defection temptation)")
    ax.set_title("(a) C1* Region (Rationality)")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor=stable_color, edgecolor="none", label="C1* holds"),
            Patch(facecolor=unstable_color, edgecolor="none", label="C1* violated"),
            Patch(facecolor="#fef3c7", edgecolor="#f59e0b", label="Plausible range"),
        ]
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([kappa_min, kappa_max])
    ax.set_ylim([d_min, d_max])

    # Panel (b): sensitivity to beta_alpha (boundary shifts).
    ax = axes[1]
    alpha_values = [
        alpha_plaus_min,
        0.5 * (alpha_plaus_min + params.beta_alpha),
        params.beta_alpha,
        alpha_plaus_max,
    ]
    colors_alpha = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_values)))
    for alpha_val, color in zip(alpha_values, colors_alpha):
        boundary = alpha_val + beta_kappa
        ax.plot(beta_kappa, boundary, color=color, linewidth=2, label=f"$\\beta_\\alpha$={alpha_val}")
        ax.fill_between(beta_kappa, d_min, boundary, color=color, alpha=0.08)
    # Patience-free boundary: mid-range threshold
    patience_free_alpha = 0.7
    patience_free_beta_D = 0.7
    patience_free_boundary_line = patience_free_alpha + beta_kappa
    ax.plot(
        beta_kappa,
        patience_free_boundary_line,
        color="#10b981",
        linewidth=2.5,
        linestyle="--",
        label=f"Patience-free threshold ($\\beta_\\alpha$={patience_free_alpha})",
    )
    ax.axhline(
        patience_free_beta_D,
        color="#10b981",
        linewidth=1.5,
        linestyle=":",
        alpha=0.6,
    )
    ax.axvspan(kappa_plaus_min, kappa_plaus_max, color="#fef3c7", alpha=0.2, label="_nolegend_")
    ax.axhspan(d_plaus_min, d_plaus_max, color="#fde68a", alpha=0.12, label="_nolegend_")
    ax.set_xlabel(r"$\beta_\kappa$ (deterrence strength)")
    ax.set_ylabel(r"Max $\beta_D$ with C1* holding")
    ax.set_title("(b) Boundary Shifts with $\\beta_\\alpha$")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([kappa_min, kappa_max])
    ax.set_ylim([d_min, d_max])
    ax.text(
        0.02,
        0.08,
        "Shaded bands: plausible ranges",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Panel (c): stability fraction vs beta_alpha under different distributional assumptions.
    ax = axes[2]

    def fraction_stable(
        alpha_values: np.ndarray,
        kappa_vals: np.ndarray,
        d_vals: np.ndarray,
        w_kappa: np.ndarray,
        w_d: np.ndarray,
    ) -> np.ndarray:
        grid_kappa, grid_d = np.meshgrid(kappa_vals, d_vals)
        weights = np.outer(w_d, w_kappa)
        total = float(np.sum(weights))
        fractions = []
        delta = grid_d - grid_kappa
        for alpha in alpha_values:
            mask = delta <= alpha
            fractions.append(float(np.sum(weights[mask]) / total))
        return np.array(fractions)

    def normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) / (x * sigma * np.sqrt(2.0 * np.pi))

    def uniform_grid(min_val: float, max_val: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        vals = np.linspace(min_val, max_val, grid_size)
        weights = np.ones_like(vals)
        return vals, weights

    def log_uniform_grid(min_val: float, max_val: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        vals = np.logspace(np.log10(min_val), np.log10(max_val), grid_size)
        weights = 1.0 / vals
        return vals, weights

    def normal_grid(mean: float, min_val: float, max_val: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        sigma = _normal_sigma(mean, min_val, max_val)
        vals = np.linspace(min_val, max_val, grid_size)
        weights = normal_pdf(vals, mean, sigma)
        return vals, weights

    def log_normal_grid(mean: float, min_val: float, max_val: float, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        sigma = _lognormal_sigma(mean, min_val, max_val)
        mu = np.log(mean)
        vals = np.logspace(np.log10(min_val), np.log10(max_val), grid_size)
        weights = lognormal_pdf(vals, mu, sigma)
        return vals, weights

    alpha_low = max(alpha_plaus_min * 0.7, 0.05)
    alpha_high = alpha_plaus_max * 1.3
    alpha_values = np.linspace(alpha_low, alpha_high, 160)

    grid_size = 220

    def curve_for_distribution(
        grid_fn,
        mean_kappa: float | None = None,
        mean_d: float | None = None,
    ) -> np.ndarray:
        if mean_kappa is None or mean_d is None:
            kappa_vals, w_kappa = grid_fn(kappa_plaus_min, kappa_plaus_max, grid_size)
            d_vals, w_d = grid_fn(d_plaus_min, d_plaus_max, grid_size)
        else:
            kappa_vals, w_kappa = grid_fn(mean_kappa, kappa_plaus_min, kappa_plaus_max, grid_size)
            d_vals, w_d = grid_fn(mean_d, d_plaus_min, d_plaus_max, grid_size)
        return fraction_stable(alpha_values, kappa_vals, d_vals, w_kappa, w_d)

    uniform_curve = curve_for_distribution(uniform_grid)
    log_uniform_curve = curve_for_distribution(log_uniform_grid)
    normal_curve = curve_for_distribution(normal_grid, params.beta_kappa, params.beta_D)
    log_normal_curve = curve_for_distribution(log_normal_grid, params.beta_kappa, params.beta_D)

    ax.plot(alpha_values, uniform_curve, color="#d97706", linewidth=2, linestyle="--", label="Uniform")
    ax.plot(alpha_values, log_uniform_curve, color="#1f5a99", linewidth=2, label="Log-uniform")
    ax.plot(alpha_values, normal_curve, color="#15803d", linewidth=2, linestyle=":", label="Normal")
    ax.plot(alpha_values, log_normal_curve, color="#7c3aed", linewidth=2, linestyle="-.", label="Log-normal")

    ax.axvspan(alpha_plaus_min, alpha_plaus_max, color="#fef3c7", alpha=0.2, label="Plausible range")
    # Patience-free boundary: where β_α = β_D = 0.7
    patience_free_alpha_c = 0.7
    ax.axvline(
        patience_free_alpha_c,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        label=r"Patience-free boundary ($\beta_\alpha=\beta_D$)",
    )
    ax.set_xlabel(r"$\beta_\alpha$ (coordination value)")
    ax.set_ylabel("Fraction where C1* holds")
    ax.set_title("(c) Distribution-Sensitive C1* Hold Fraction")
    ax.set_ylim([0.5, 1.1])
    ax.set_xlim([alpha_low, alpha_high])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.15,
        "Shaded region marks\nplausible $\\beta_\\alpha$ range",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.suptitle("Equilibrium Stability Analysis - C1* (Correlated Equilibrium)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, path)


def _set_margin_ylim(ax: plt.Axes, *series: Sequence[float], cap: float = 2.0) -> None:
    values = np.concatenate([np.asarray(s) for s in series] + [np.asarray([0.0])])
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 0.5:
        center = 0.5 * (max_val + min_val)
        min_val = center - 0.25
        max_val = center + 0.25
    min_val = max(min_val, -cap)
    max_val = min(max_val, cap)
    if min_val >= max_val:
        min_val, max_val = -cap, cap
    ax.set_ylim(min_val, max_val)


def _sensitivity_panel(
    ax: plt.Axes,
    values: Sequence[float],
    margin_min: Sequence[float],
    margin_c1: Sequence[float],
    margin_c2: Sequence[float],
    anchor_value: float,
    xlabel: str,
    title: str,
    log_scale: bool = True,
    plausible_range: tuple[float, float] | None = None,
) -> None:
    margins_arr = np.asarray(margin_min)
    c1_arr = np.asarray(margin_c1)
    c2_arr = np.asarray(margin_c2)
    ax.plot(values, margins_arr, color="#1f5a99", linewidth=2, label="Overall margin S")
    ax.plot(values, c1_arr, color="#64748b", linestyle="--", linewidth=1.2, label="C1* margin")
    ax.plot(values, c2_arr, color="#94a3b8", linestyle=":", linewidth=1.4, label="C2* margin")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="Threshold (S=0)")
    ax.fill_between(values, 0.0, margins_arr, where=margins_arr >= 0.0, color="#b7e3b3", alpha=0.6, label="C1* & C2* hold")
    ax.fill_between(values, 0.0, margins_arr, where=margins_arr < 0.0, color="#f5c2c2", alpha=0.6, label="C1* or C2* violated")
    ax.axvline(anchor_value, color="#cc0000", linestyle="--", linewidth=1, label="Operating point")
    if plausible_range:
        ax.axvspan(plausible_range[0], plausible_range[1], color="#fef3c7", alpha=0.18, label="Plausible range")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if log_scale:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    _set_margin_ylim(ax, margins_arr, c1_arr, c2_arr)


def figure_sensitivity_analysis(path: str) -> None:
    """Single-parameter sensitivity of stability margin S with joint robustness map."""

    apply_style()
    params = anchor_params()
    ranges = plausible_ranges()
    n_agents = 5
    margin_value = params.beta_alpha + params.beta_kappa - params.beta_D

    old_params = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig = plt.figure(figsize=(8.25, 10.2))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.1, 1.1, 1.1, 1.6], hspace=0.38, wspace=0.2)
    ax_kappa = fig.add_subplot(gs[0, 0])
    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_Omega = fig.add_subplot(gs[1, 1])
    ax_ell = fig.add_subplot(gs[2, 0])
    ax_N = fig.add_subplot(gs[2, 1])
    ax_joint = fig.add_subplot(gs[3, :])

    beta_kappa_vals = np.logspace(
        np.log10(ranges.beta_kappa[0]), np.log10(ranges.beta_kappa[1]), 120
    )
    beta_kappa_c1 = [replace_param(params, beta_kappa=v).beta_alpha + v - params.beta_D for v in beta_kappa_vals]
    beta_kappa_c2 = [params.beta_Omega - params.beta_ell / n_agents for _ in beta_kappa_vals]
    beta_kappa_margins = [min(c1, c2) for c1, c2 in zip(beta_kappa_c1, beta_kappa_c2)]
    _sensitivity_panel(
        ax_kappa,
        beta_kappa_vals,
        beta_kappa_margins,
        beta_kappa_c1,
        beta_kappa_c2,
        params.beta_kappa,
        r"$\beta_\kappa$",
        "(a) Deterrence Strength",
        plausible_range=ranges.beta_kappa,
    )
    ax_kappa.legend(loc="lower right", fontsize=8)
    kappa_min, kappa_max = _log_pad(ranges.beta_kappa[0], ranges.beta_kappa[1], factor=1.3)
    ax_kappa.set_xlim([kappa_min, kappa_max])

    beta_alpha_vals = np.logspace(
        np.log10(ranges.beta_alpha[0]), np.log10(ranges.beta_alpha[1]), 120
    )
    beta_alpha_c1 = [v + params.beta_kappa - params.beta_D for v in beta_alpha_vals]
    beta_alpha_c2 = [params.beta_Omega - params.beta_ell / n_agents for _ in beta_alpha_vals]
    beta_alpha_margins = [min(c1, c2) for c1, c2 in zip(beta_alpha_c1, beta_alpha_c2)]
    _sensitivity_panel(
        ax_alpha,
        beta_alpha_vals,
        beta_alpha_margins,
        beta_alpha_c1,
        beta_alpha_c2,
        params.beta_alpha,
        r"$\beta_\alpha$",
        "(b) Coordination Value",
        plausible_range=ranges.beta_alpha,
    )
    alpha_min, alpha_max = _log_pad(ranges.beta_alpha[0], ranges.beta_alpha[1], factor=1.3)
    ax_alpha.set_xlim([alpha_min, alpha_max])
    ax_alpha.legend(loc="lower right", fontsize=8)
    # Add patience-free threshold (β_α = 0.7)
    ax_alpha.axvline(0.7, color="#10b981", linestyle="--", linewidth=2, alpha=0.8, label="Patience-free")

    beta_D_vals = np.logspace(
        np.log10(ranges.beta_D[0]), np.log10(ranges.beta_D[1]), 120
    )
    beta_D_c1 = [params.beta_alpha + params.beta_kappa - v for v in beta_D_vals]
    beta_D_c2 = [params.beta_Omega - params.beta_ell / n_agents for _ in beta_D_vals]
    beta_D_margins = [min(c1, c2) for c1, c2 in zip(beta_D_c1, beta_D_c2)]
    _sensitivity_panel(
        ax_D,
        beta_D_vals,
        beta_D_margins,
        beta_D_c1,
        beta_D_c2,
        params.beta_D,
        r"$\beta_D$",
        "(c) Defection Temptation",
        log_scale=True,
        plausible_range=ranges.beta_D,
    )
    d_min, d_max = _log_pad(ranges.beta_D[0], ranges.beta_D[1], factor=1.3)
    ax_D.set_xlim([d_min, d_max])
    ax_D.legend(loc="lower right", fontsize=8)
    # Add patience-free threshold (β_D = 0.7)
    ax_D.axvline(0.7, color="#10b981", linestyle="--", linewidth=2, alpha=0.8, label="Patience-free")

    beta_Omega_vals = np.logspace(
        np.log10(ranges.beta_Omega[0]), np.log10(ranges.beta_Omega[1]), 120
    )
    beta_Omega_c1 = [params.beta_alpha + params.beta_kappa - params.beta_D for _ in beta_Omega_vals]
    beta_Omega_c2 = [v - params.beta_ell / n_agents for v in beta_Omega_vals]
    beta_Omega_margins = [min(c1, c2) for c1, c2 in zip(beta_Omega_c1, beta_Omega_c2)]
    _sensitivity_panel(
        ax_Omega,
        beta_Omega_vals,
        beta_Omega_margins,
        beta_Omega_c1,
        beta_Omega_c2,
        params.beta_Omega,
        r"$\beta_\Omega$",
        "(d) Oversight Value",
        plausible_range=ranges.beta_Omega,
    )
    omega_min, omega_max = _log_pad(ranges.beta_Omega[0], ranges.beta_Omega[1], factor=1.3)
    ax_Omega.set_xlim([omega_min, omega_max])
    ax_Omega.legend(loc="lower right", fontsize=8)

    beta_ell_vals = np.logspace(
        np.log10(ranges.beta_ell[0]), np.log10(ranges.beta_ell[1]), 120
    )
    beta_ell_c1 = [params.beta_alpha + params.beta_kappa - params.beta_D for _ in beta_ell_vals]
    beta_ell_c2 = [params.beta_Omega - v / n_agents for v in beta_ell_vals]
    beta_ell_margins = [min(c1, c2) for c1, c2 in zip(beta_ell_c1, beta_ell_c2)]
    _sensitivity_panel(
        ax_ell,
        beta_ell_vals,
        beta_ell_margins,
        beta_ell_c1,
        beta_ell_c2,
        params.beta_ell,
        r"$\beta_\ell$",
        "(e) Removal Gain",
        plausible_range=ranges.beta_ell,
    )
    ell_min, ell_max = _log_pad(ranges.beta_ell[0], ranges.beta_ell[1], factor=1.3)
    ax_ell.set_xlim([ell_min, ell_max])
    ax_ell.legend(loc="lower right", fontsize=8)

    n_vals = np.arange(ranges.n_agents[0], ranges.n_agents[1] + 1)
    n_c1 = [params.beta_alpha + params.beta_kappa - params.beta_D for _ in n_vals]
    n_c2 = [params.beta_Omega - params.beta_ell / int(n) for n in n_vals]
    n_margins = [min(c1, c2) for c1, c2 in zip(n_c1, n_c2)]
    ax_N.plot(n_vals, n_margins, color="#1f5a99", linewidth=2, label="S = min(C1*, C2*)")
    ax_N.plot(n_vals, n_c1, color="#64748b", linestyle="--", linewidth=1.2, label="C1* margin")
    ax_N.plot(n_vals, n_c2, color="#94a3b8", linestyle=":", linewidth=1.4, label="C2* margin")
    ax_N.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_N.fill_between(n_vals, 0.0, n_margins, where=np.array(n_margins) >= 0.0, color="#b7e3b3", alpha=0.6)

    # Band-robust threshold at N=6
    ax_N.axvline(6, color="#10b981", linestyle="--", linewidth=2.5, label="Band-robust (N≥6)")
    ax_N.axvspan(6, ranges.n_agents[1] + 0.5, color="#10b981", alpha=0.12)

    n_ir = int(np.ceil(ranges.beta_ell[1] / ranges.beta_Omega[0]))
    n_ir = max(ranges.n_agents[0], min(ranges.n_agents[1], n_ir))
    ax_N.axvline(n_ir, color="#111827", linestyle=":", linewidth=1.2)
    ax_N.text(
        0.02,
        0.92,
        f"C2* guaranteed\nfor N≥{n_ir} (bands)",
        transform=ax_N.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax_N.axvspan(ranges.n_agents[0], ranges.n_agents[1], color="#fef3c7", alpha=0.18)
    ax_N.set_xlabel("N")
    ax_N.set_title("(f) Group Size")
    ax_N.grid(True, alpha=0.25)
    ax_N.legend(loc="lower right", fontsize=8)
    _set_margin_ylim(ax_N, n_margins, n_c1, n_c2, cap=1.5)
    ax_N.set_xlim([ranges.n_agents[0] - 0.5, ranges.n_agents[1] + 0.5])

    for ax in [ax_kappa, ax_alpha, ax_D, ax_Omega, ax_ell, ax_N]:
        ax.set_ylabel("Stability Margin S")

    # Joint robustness map (C1* relative margin).
    kappa_2d_min, kappa_2d_max = _log_pad(ranges.beta_kappa[0], ranges.beta_kappa[1], factor=1.5)
    d_2d_min, d_2d_max = _log_pad(ranges.beta_D[0], ranges.beta_D[1], factor=1.5)
    kappa_2d = np.logspace(np.log10(kappa_2d_min), np.log10(kappa_2d_max), 90)
    d_2d = np.linspace(d_2d_min, d_2d_max, 90)
    K2D, D2D = np.meshgrid(kappa_2d, d_2d)
    c1_margin = params.beta_alpha + K2D - D2D
    relative_margin = np.clip(c1_margin / (params.beta_alpha + K2D), 0.0, 1.0)

    im = ax_joint.contourf(K2D, D2D, relative_margin, levels=20, cmap="RdYlGn")
    contours = ax_joint.contour(K2D, D2D, relative_margin, levels=[0.1, 0.5, 0.9], colors="black", linewidths=1, alpha=0.6)
    ax_joint.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    boundary_line = params.beta_alpha + kappa_2d
    ax_joint.plot(kappa_2d, boundary_line, color="black", linestyle="--", linewidth=1.2, label="C1* boundary")

    # Patience-free boundary: β_D = 0.7
    ax_joint.axhline(0.7, color="#10b981", linestyle="--", linewidth=2.5, label="Patience-free (δ*=0)")
    ax_joint.axhspan(d_2d_min, 0.7, color="#10b981", alpha=0.1)

    ax_joint.add_patch(
        Rectangle(
            (ranges.beta_kappa[0], ranges.beta_D[0]),
            ranges.beta_kappa[1] - ranges.beta_kappa[0],
            ranges.beta_D[1] - ranges.beta_D[0],
            facecolor="#fef3c7",
            edgecolor="#f59e0b",
            linewidth=1.0,
            alpha=0.2,
            label="Plausible range",
        )
    )
    ax_joint.set_xscale("log")
    ax_joint.set_xlim([kappa_2d.min(), kappa_2d.max()])
    ax_joint.set_ylim([d_2d.min(), d_2d.max()])
    ax_joint.margins(x=0.0)
    ax_joint.set_xlabel(r"$\beta_\kappa$ (deterrence strength)", fontsize=11)
    ax_joint.set_ylabel(r"$\beta_D$ (defection temptation)", fontsize=11)
    ax_joint.set_title("(g) Joint Robustness Map (Relative C1* Margin)", fontsize=10)
    ax_joint.tick_params(labelsize=9)
    ax_joint.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax_joint, label="Relative margin (0=fragile, 1=robust)", pad=0.01, fraction=0.035)
    ax_joint.text(
        0.02,
        0.08,
        "Higher values = larger buffer before C1* fails\nGreen region: Patience-free (δ*=0)",
        transform=ax_joint.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.suptitle("Sensitivity of Stability Margin", fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.07)
    _save(fig, path)
    plt.rcParams.update(old_params)


def figure_ce_convergence(path: str) -> None:
    """Welfare comparison across coordination mechanisms (paper-style)."""

    params = anchor_params()
    ranges = plausible_ranges()
    n_agents = 5
    delta_assumed = 0.9
    n_samples = 6000

    with plt.style.context("seaborn-v0_8-whitegrid"):
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel (a): Welfare comparison
        ax = axes[0]
        mechanisms = [
            "Nash\n(No Coord.)",
            "Random\nMediator",
            "Correlated\nEquilibrium",
            "Public\nMonitoring",
            "Perfect\nCoord.",
        ]

        G0 = params.G0
        p_rand = 0.5
        p_all = p_rand ** (n_agents - 1)
        expected_defectors = (n_agents - 1) * (1.0 - p_rand)

        def simulate_distribution(dist: str, seed: int) -> dict[str, np.ndarray]:
            rng = np.random.default_rng(seed)
            beta_alpha = _sample_param_range(
                rng,
                ranges.beta_alpha[0],
                ranges.beta_alpha[1],
                params.beta_alpha,
                dist,
                n_samples,
            )
            beta_kappa = _sample_param_range(
                rng,
                ranges.beta_kappa[0],
                ranges.beta_kappa[1],
                params.beta_kappa,
                dist,
                n_samples,
            )
            beta_D = _sample_param_range(
                rng,
                ranges.beta_D[0],
                ranges.beta_D[1],
                params.beta_D,
                dist,
                n_samples,
            )

            welfare_coop = 1.0 + beta_alpha
            welfare_defect = 1.0 + beta_D - beta_kappa * n_agents
            u_coop = 1.0 + beta_alpha * p_all
            u_defect = 1.0 + beta_D - beta_kappa * (1.0 + expected_defectors)
            welfare_random = p_rand * u_coop + (1.0 - p_rand) * u_defect

            c1_holds = beta_alpha + beta_kappa >= beta_D
            welfare_ce = np.where(c1_holds, welfare_coop, welfare_defect)

            numerator = beta_D - beta_alpha
            denominator = numerator + beta_kappa
            delta_star = np.zeros_like(numerator, dtype=float)
            mask = denominator > 0.0
            delta_star[mask] = numerator[mask] / denominator[mask]
            delta_star = np.clip(delta_star, 0.0, 1.0)
            delta_star[numerator <= 0.0] = 0.0
            c1pp_holds = delta_assumed >= delta_star
            welfare_ppe = np.where(c1pp_holds, welfare_coop, welfare_defect)

            welfare_perfect = welfare_coop

            return {
                "nash": welfare_defect,
                "random": welfare_random,
                "ce": welfare_ce,
                "ppe": welfare_ppe,
                "perfect": welfare_perfect,
                "c1_holds": c1_holds,
                "c1pp_holds": c1pp_holds,
            }

        uniform_metrics = simulate_distribution("uniform", seed=41)

        welfare_values = [
            float(np.mean(uniform_metrics["nash"])),
            float(np.mean(uniform_metrics["random"])),
            float(np.mean(uniform_metrics["ce"])),
            float(np.mean(uniform_metrics["ppe"])),
            float(np.mean(uniform_metrics["perfect"])),
        ]
        welfare_p5 = [
            float(np.percentile(uniform_metrics[key], 5.0))
            for key in ("nash", "random", "ce", "ppe", "perfect")
        ]
        welfare_p95 = [
            float(np.percentile(uniform_metrics[key], 95.0))
            for key in ("nash", "random", "ce", "ppe", "perfect")
        ]
        welfare_err = np.array(
            [[m - p5 for m, p5 in zip(welfare_values, welfare_p5)],
             [p95 - m for m, p95 in zip(welfare_values, welfare_p95)]]
        )
        colors = ["#d73027", "#fee08b", "#91cf60", "#1a9850", "#006837"]

        bars = ax.bar(
            range(len(mechanisms)),
            welfare_values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            yerr=welfare_err,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

        span = max(welfare_values) - min(welfare_values)
        pad = max(0.1 * span, 0.2)
        offset = max(0.05 * span, 0.08)
        for bar, val in zip(bars, welfare_values):
            height = bar.get_height()
            y = height + offset if height >= 0 else height - offset
            va = "bottom" if height >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontweight="bold",
            )

        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms)
        ax.set_ylabel("Per-agent welfare", fontsize=12)
        ax.set_ylim([min(welfare_values) - pad, max(welfare_values) + pad])
        ax.set_title("(a) Equilibrium Welfare Under Different Mechanisms", fontweight="bold")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        marker_defs = [
            ("Log-uniform", "^", "#1f5a99", 0.15),
            ("Normal", "s", "#15803d", 0.0),
            ("Log-normal", "D", "#7c3aed", -0.15),
        ]
        other_metrics = {
            dist: simulate_distribution(dist, seed=51 + idx)
            for idx, dist in enumerate(["log-uniform", "normal", "log-normal"])
        }
        for dist, marker, color, shift in marker_defs:
            series = other_metrics[dist.lower()]
            mean_vals = [
                float(np.mean(series["nash"])),
                float(np.mean(series["random"])),
                float(np.mean(series["ce"])),
                float(np.mean(series["ppe"])),
                float(np.mean(series["perfect"])),
            ]
            ax.scatter(
                np.arange(len(mechanisms)) + shift,
                mean_vals,
                color=color,
                marker=marker,
                s=26,
                zorder=5,
                edgecolor="black",
                linewidth=0.5,
            )
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="white",
                label=label,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=6,
            )
            for label, marker, color, _ in marker_defs
        ]
        ax.legend(
            handles=marker_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            fontsize=9,
            frameon=True,
        )

        # Panel (b): Equilibrium selection
        ax = axes[1]

        mechanisms_short = ["Nash", "CE", "PPE"]
        selection_probs = [
            0.5,
            float(np.mean(uniform_metrics["c1_holds"])),
            float(np.mean(uniform_metrics["c1pp_holds"])),
        ]
        selection_entropy = []
        for prob in selection_probs:
            if prob <= 0.0 or prob >= 1.0:
                selection_entropy.append(0.0)
            else:
                selection_entropy.append(
                    -(prob * np.log2(prob) + (1.0 - prob) * np.log2(1.0 - prob))
                )

        x_pos = np.arange(len(mechanisms_short))
        width = 0.34
        mechanism_colors = [colors[0], colors[2], colors[3]]
        bars_prob = ax.bar(
            x_pos - width / 2.0,
            selection_probs,
            width,
            color=mechanism_colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=1.8,
        )
        bars_entropy = ax.bar(
            x_pos + width / 2.0,
            selection_entropy,
            width,
            color=mechanism_colors,
            alpha=0.4,
            edgecolor="black",
            linewidth=1.8,
            hatch="///",
        )
        for idx, (bar, val) in enumerate(zip(bars_prob, selection_probs)):
            extra = 0.06 if idx == 1 else 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + extra,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for bar, val in zip(bars_entropy, selection_entropy):
            label_y = val + 0.03 if val >= 0.12 else 0.12
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        for dist, marker, color, shift in marker_defs:
            series = other_metrics[dist.lower()]
            prob_vals = [
                0.5,
                float(np.mean(series["c1_holds"])),
                float(np.mean(series["c1pp_holds"])),
            ]
            ax.scatter(
                x_pos - width / 2.0 + shift,
                prob_vals,
                color=color,
                marker=marker,
                s=24,
                zorder=5,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(mechanisms_short)
        ax.set_ylabel("Selection probability / uncertainty (bits)", fontsize=12)
        ax.set_title("(b) Cooperative Selection and Uncertainty", fontweight="bold")
        legend_handles = [
            Patch(facecolor="white", edgecolor="black", linewidth=1.8, label="P(cooperative)"),
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=1.8,
                hatch="///",
                label="Selection uncertainty",
            ),
        ]
        legend_main = ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.28),
            fontsize=10,
            frameon=True,
        )
        ax.add_artist(legend_main)
        ax.legend(
            handles=marker_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, -0.28),
            ncol=1,
            fontsize=9,
            frameon=True,
        )
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3, axis="y")

        c1_fail = 1.0 - float(np.mean(uniform_metrics["c1_holds"]))
        c1pp_fail = 1.0 - float(np.mean(uniform_metrics["c1pp_holds"]))

        ce_bar = bars_prob[1]
        ppe_bar = bars_prob[2]
        ax.annotate(
            f"C1*: fail {c1_fail:.1%}\nC1**: fail {c1pp_fail:.1%}",
            xy=(ce_bar.get_x() + ce_bar.get_width() / 2.0, selection_probs[1]),
            xytext=(x_pos[1] - 0.35, 1.18),
            textcoords="data",
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
        )

        plt.suptitle(
            "Mechanism Design Comparison (Equilibrium Analysis)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.subplots_adjust(top=0.88, bottom=0.28, left=0.06, right=0.98, wspace=0.25)
        plt.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)


def figure_ppe_sustainability(path: str) -> None:
    """PPE sustainability: delta_crit curve and feasible region."""

    apply_style()
    params = anchor_params()
    ranges = plausible_ranges()
    n_agents = 5
    delta_assumed = 0.9
    beta_alpha = params.beta_alpha
    beta_kappa = params.beta_kappa
    beta_D = params.beta_D
    n_samples = 2500
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(11, 7.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel (a): delta_crit vs beta_D at anchor beta_alpha, beta_kappa.
    ax1 = fig.add_subplot(gs[0, 0])
    beta_d_min, beta_d_max = _log_pad(ranges.beta_D[0], ranges.beta_D[1], factor=3.0)
    beta_d_range = np.linspace(beta_d_min, beta_d_max, 240)
    delta_vals = np.maximum(
        0.0,
        (beta_d_range - beta_alpha) / (beta_d_range - beta_alpha + beta_kappa),
    )
    delta_vals = np.clip(delta_vals, 0.0, 1.0)
    alpha_samples = rng.uniform(ranges.beta_alpha[0], ranges.beta_alpha[1], n_samples)
    kappa_samples = rng.uniform(ranges.beta_kappa[0], ranges.beta_kappa[1], n_samples)
    beta_d_grid = beta_d_range[None, :]
    num = beta_d_grid - alpha_samples[:, None]
    den = num + kappa_samples[:, None]
    delta_samples = np.where(den <= 0.0, 0.0, num / den)
    delta_samples = np.clip(delta_samples, 0.0, 1.0)
    delta_p5 = np.percentile(delta_samples, 5.0, axis=0)
    delta_p95 = np.percentile(delta_samples, 95.0, axis=0)
    ax1.fill_between(
        beta_d_range,
        delta_vals,
        1.0,
        color="#b7e3b3",
        alpha=0.18,
        label="PPE sustainable (δ ≥ δ_crit)",
    )
    ax1.fill_between(
        beta_d_range,
        0.0,
        delta_vals,
        color="#f4b6b6",
        alpha=0.18,
        label="PPE fails (δ < δ_crit)",
    )
    ax1.plot(beta_d_range, delta_vals, color="#1f5a99", linewidth=2, label=r"$\delta_{crit}$")
    ax1.fill_between(
        beta_d_range,
        delta_p5,
        delta_p95,
        color="#1f5a99",
        alpha=0.15,
        label="Plausible range band",
    )
    ax1.axvspan(ranges.beta_D[0], ranges.beta_D[1], color="#fef3c7", alpha=0.18, label="Plausible $\\beta_D$")
    # Patience-free boundary: where β_D = β_α (enhanced)
    ax1.axvline(
        beta_alpha,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        alpha=0.9,
        label=r"Patience-free boundary ($\beta_D=\beta_\alpha$, $\delta^*=0$)",
    )
    ax1.axhline(delta_assumed, color="#0f766e", linestyle=":", linewidth=1.5, label=r"$\delta=0.9$")
    ax1.set_xlabel(r"Defection temptation ($\beta_D$)")
    ax1.set_ylabel(r"Critical discount factor ($\delta_{crit}$)")
    ax1.set_title("(a) C1** Threshold vs $\\,\\beta_D$")
    ax1.set_xlim([beta_d_min, beta_d_max])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Panel (b): sustainability region in (delta, beta_D).
    ax2 = fig.add_subplot(gs[0, 1])
    delta_range = np.linspace(0.0, 1.0, 200)
    beta_d_grid = np.linspace(beta_d_min, beta_d_max, 240)
    grid_delta, grid_d = np.meshgrid(delta_range, beta_d_grid)
    delta_crit_grid = np.maximum(
        0.0,
        (grid_d - beta_alpha) / (grid_d - beta_alpha + beta_kappa),
    )
    delta_crit_grid = np.clip(delta_crit_grid, 0.0, 1.0)
    sustainable_grid = grid_delta >= delta_crit_grid

    ax2.contourf(
        grid_delta,
        grid_d,
        sustainable_grid,
        levels=[0.0, 0.5, 1.0],
        colors=["#f4b6b6", "#b7e3b3"],
        alpha=0.85,
    )
    ax2.contour(
        grid_delta,
        grid_d,
        delta_crit_grid,
        levels=[delta_assumed],
        colors="#334155",
        linestyles="--",
        linewidths=1.2,
    )
    ax2.axhspan(ranges.beta_D[0], ranges.beta_D[1], color="#fef3c7", alpha=0.18, label="_nolegend_")
    # Patience-free boundary: β_D = β_α (enhanced)
    ax2.axhline(
        beta_alpha,
        color="#10b981",
        linestyle="--",
        linewidth=2.5,
        alpha=0.9,
        label=r"Patience-free ($\beta_D=\beta_\alpha$)",
    )
    ax2.plot([], [], linestyle="--", color="#334155", label=r"$\delta=0.9$ boundary")
    ax2.text(
        0.05,
        0.95,
        "PPE fails",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#b91c1c",
    )
    ax2.text(
        0.72,
        0.18,
        "PPE holds",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#15803d",
    )

    ax2.set_xlabel(r"Discount factor ($\delta$)")
    ax2.set_ylabel(r"Defection temptation ($\beta_D$)")
    ax2.set_title("(b) PPE Sustainability Region")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)

    # Panel (c): trigger-strategy incentive gap vs delta.
    ax3 = fig.add_subplot(gs[1, 0])
    rounds = np.arange(1, 51)
    discount_profile = delta_assumed ** (rounds - 1)
    gap_anchor = (1.0 + beta_alpha) / (1.0 - delta_assumed) - (
        (1.0 + beta_D - beta_kappa) + delta_assumed * (1.0 - beta_kappa * n_agents) / (1.0 - delta_assumed)
    )
    gap_curve = gap_anchor * discount_profile
    alpha_s = rng.uniform(ranges.beta_alpha[0], ranges.beta_alpha[1], n_samples)
    kappa_s = rng.uniform(ranges.beta_kappa[0], ranges.beta_kappa[1], n_samples)
    d_s = rng.uniform(ranges.beta_D[0], ranges.beta_D[1], n_samples)
    gap_base = (1.0 + alpha_s) / (1.0 - delta_assumed) - (
        (1.0 + d_s - kappa_s) + delta_assumed * (1.0 - kappa_s * n_agents) / (1.0 - delta_assumed)
    )
    gap_samples = gap_base[:, None] * discount_profile[None, :]
    gap_p5 = np.percentile(gap_samples, 5.0, axis=0)
    gap_p95 = np.percentile(gap_samples, 95.0, axis=0)
    ax3.fill_between(rounds, gap_p5, gap_p95, color="#1f5a99", alpha=0.15, label="Plausible range band")
    ax3.plot(rounds, gap_curve, color="#1f5a99", linewidth=2, label="Mid-range gap")
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Discounted incentive gap")
    ax3.set_title("(c) Trigger-Strategy Gap Over Rounds")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # Panel (d): repeated-game advantage (present value).
    ax4 = fig.add_subplot(gs[1, 1])
    delta_levels = [0.0, 0.5, 0.9]
    coop_vals = []
    defect_vals = []
    coop_err = []
    defect_err = []
    for delta_val in delta_levels:
        coop_anchor = (1.0 + beta_alpha) / (1.0 - delta_val) if delta_val < 1.0 else np.nan
        defect_anchor = (
            (1.0 + beta_D - beta_kappa)
            + delta_val * (1.0 - beta_kappa * n_agents) / (1.0 - delta_val)
            if delta_val < 1.0
            else np.nan
        )
        coop_vals.append(coop_anchor)
        defect_vals.append(defect_anchor)

        alpha_s = rng.uniform(ranges.beta_alpha[0], ranges.beta_alpha[1], n_samples)
        kappa_s = rng.uniform(ranges.beta_kappa[0], ranges.beta_kappa[1], n_samples)
        d_s = rng.uniform(ranges.beta_D[0], ranges.beta_D[1], n_samples)
        coop_s = (1.0 + alpha_s) / (1.0 - delta_val)
        defect_s = (1.0 + d_s - kappa_s) + delta_val * (1.0 - kappa_s * n_agents) / (1.0 - delta_val)
        coop_err.append(
            [coop_anchor - np.percentile(coop_s, 5.0), np.percentile(coop_s, 95.0) - coop_anchor]
        )
        defect_err.append(
            [defect_anchor - np.percentile(defect_s, 5.0), np.percentile(defect_s, 95.0) - defect_anchor]
        )
    x_pos = np.arange(len(delta_levels))
    width = 0.35
    ax4.bar(
        x_pos - width / 2.0,
        coop_vals,
        width,
        color="#1f5a99",
        label=r"$V_{coop}$",
        yerr=np.array(coop_err).T,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax4.bar(
        x_pos + width / 2.0,
        defect_vals,
        width,
        color="#d97706",
        label=r"$V_{defect}$",
        yerr=np.array(defect_err).T,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"$\\delta={d:.1f}$" for d in delta_levels])
    ax4.set_ylabel("Present value")
    ax4.set_title("(d) Repeated-Game Advantage")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.25, axis="y")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95, wspace=0.28, hspace=0.32)
    _save(fig, path)


def figure_shapley_participation(path: str) -> None:
    """Shapley participation vs removal benefit."""

    apply_style()
    params = anchor_params()
    ranges = plausible_ranges()
    beta_synergy = 0.0
    n_samples = 2500
    rng = np.random.default_rng(101)
    n_vals = np.arange(ranges.n_agents[0], ranges.n_agents[1] + 1)
    shapley_vals = np.array(
        [
            symmetric_shapley_per_agent(int(n), params.beta_Omega, beta_synergy)
            for n in n_vals
        ]
    )
    removal_vals = params.beta_ell / n_vals.astype(float)
    margin_vals = shapley_vals - removal_vals
    beta_omega_samples = rng.uniform(ranges.beta_Omega[0], ranges.beta_Omega[1], n_samples)
    beta_ell_samples = rng.uniform(ranges.beta_ell[0], ranges.beta_ell[1], n_samples)
    shapley_p5 = np.full_like(n_vals, np.percentile(beta_omega_samples, 5.0), dtype=float)
    shapley_p95 = np.full_like(n_vals, np.percentile(beta_omega_samples, 95.0), dtype=float)
    removal_p5 = np.array([np.percentile(beta_ell_samples / n, 5.0) for n in n_vals])
    removal_p95 = np.array([np.percentile(beta_ell_samples / n, 95.0) for n in n_vals])

    fig = plt.figure(figsize=(11, 7.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel (a): Shapley vs removal benefit.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_vals, shapley_vals, marker="o", color="#1f5a99", label=r"Shapley $\phi_i$")
    ax1.plot(n_vals, removal_vals, marker="s", color="#d97706", label=r"Removal $\beta_\ell/N$")
    ax1.fill_between(
        n_vals,
        shapley_p5,
        shapley_p95,
        color="#1f5a99",
        alpha=0.12,
        label="Plausible range band (Shapley)",
    )
    ax1.fill_between(
        n_vals,
        removal_p5,
        removal_p95,
        color="#d97706",
        alpha=0.12,
        label="Plausible range band (Removal)",
    )
    ax1.fill_between(
        n_vals,
        removal_vals,
        shapley_vals,
        where=shapley_vals >= removal_vals,
        color="#b7e3b3",
        alpha=0.3,
        label="C2* satisfied",
    )
    ax1.set_xlabel("Number of Agents (N)")
    ax1.set_ylabel("Per-Agent Value")
    ax1.set_title("(a) Shapley Share vs Removal Benefit")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Marginal contribution vs coalition size.
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = np.arange(1, 13)
    coalition_vals = np.array(
        [coalition_value_by_size(int(s), params.beta_Omega, beta_synergy) for s in sizes]
    )
    marginal = np.diff(np.concatenate(([0.0], coalition_vals)))
    ax2.plot(sizes, marginal, marker="o", color="#15803d", label="Marginal contribution")
    ax2.axhline(params.beta_Omega, color="#1f5a99", linestyle="--", linewidth=1.2, label=r"Baseline $\beta_\Omega$")
    ax2.axhspan(ranges.beta_Omega[0], ranges.beta_Omega[1], color="#fef3c7", alpha=0.18, label="Plausible $\\beta_\\Omega$")
    ax2.set_xlabel("Coalition Size |S|")
    ax2.set_ylabel("Marginal Value")
    ax2.set_title("(b) Agent Marginal Contribution")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Coalition value vs size.
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sizes, coalition_vals, marker="s", color="#1f5a99", label=r"Coalition value $v(S)$")
    ax3.set_xlabel("Coalition Size |S|")
    ax3.set_ylabel("Coalition Value")
    ax3.set_title("(c) Coalition Welfare Benefit")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel (d): Required beta_Omega threshold vs N.
    ax4 = fig.add_subplot(gs[1, 1])
    required_beta = params.beta_ell / n_vals.astype(float)
    ax4.plot(n_vals, required_beta, color="#d97706", linewidth=2, label=r"Threshold $\beta_\Omega=\beta_\ell/N$")
    ax4.fill_between(
        n_vals,
        removal_p5,
        removal_p95,
        color="#d97706",
        alpha=0.12,
        label="Plausible range band (Removal)",
    )
    ax4.axhspan(ranges.beta_Omega[0], ranges.beta_Omega[1], color="#fef3c7", alpha=0.18, label="Plausible $\\beta_\\Omega$")

    # Band-robust threshold at N=6
    ax4.axvline(6, color="#10b981", linestyle="--", linewidth=2.5, label="Band-robust (N≥6)")
    ax4.axvspan(6, ranges.n_agents[1] + 0.5, color="#10b981", alpha=0.12)
    ax4.set_xlabel("Number of Agents (N)")
    ax4.set_ylabel(r"Required $\beta_\Omega$")
    ax4.set_title("(d) Oversight Value Needed for C2*")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95, wspace=0.28, hspace=0.32)
    _save(fig, path)


def figure_scaling_cost_model(path: str) -> None:
    """Communication and coordination scaling analysis (multi-panel)."""

    apply_style()
    cost_params = default_cost_params()
    benefit_params = default_benefit_params()

    n_vals = np.arange(2, 21)

    def net_series(cost, benefit, key: str = "saturating") -> np.ndarray:
        return np.array([net_values(int(n), cost, benefit)[key] for n in n_vals])

    base_saturating = net_series(cost_params, benefit_params, "saturating")

    benefit_high = replace(benefit_params, sat_scale=benefit_params.sat_scale * 1.5)
    benefit_low = replace(benefit_params, sat_scale=benefit_params.sat_scale * 0.5)
    cost_eff = replace(cost_params, consensus_exp=1.0)
    cost_ineff = replace(cost_params, consensus_exp=2.0)
    cost_low_hedge = replace(cost_params, hedge_linear=cost_params.hedge_linear * 0.5)

    variants = [
        base_saturating,
        net_series(cost_params, benefit_high, "saturating"),
        net_series(cost_params, benefit_low, "saturating"),
        net_series(cost_eff, benefit_params, "saturating"),
        net_series(cost_ineff, benefit_params, "saturating"),
        net_series(cost_low_hedge, benefit_params, "saturating"),
    ]
    min_band = np.min(variants, axis=0)
    max_band = np.max(variants, axis=0)

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel (a): Net value vs N with uncertainty band.
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(n_vals, base_saturating, color="#1f5a99", linewidth=2, label="Base (saturating)")
    ax1.fill_between(n_vals, min_band, max_band, color="#1f5a99", alpha=0.15, label="Parameter band")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    idx = int(np.argmax(base_saturating))
    ax1.plot(n_vals[idx], base_saturating[idx], "r*", markersize=10, label=f"Optimal N={n_vals[idx]}")
    ax1.set_xlabel("Number of Agents (N)")
    ax1.set_ylabel("Net Value J(N)")
    ax1.set_title("(a) Net Coordination Value vs Group Size")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Cost breakdown.
    ax2 = fig.add_subplot(gs[0, 2])
    n_sample = np.arange(2, 16)
    costs = {
        "Communication": [cost_comm(n, cost_params) for n in n_sample],
        "Consensus": [cost_consensus(n, cost_params) for n in n_sample],
        "Hedging": [cost_hedging(n, cost_params) for n in n_sample],
        "Reliability": [cost_reliability(n, cost_params) for n in n_sample],
        "Latency": [cost_latency(cost_params) for _ in n_sample],
    }
    bottom = np.zeros(len(n_sample))
    palette = ["#ef4444", "#0ea5e9", "#f59e0b", "#22c55e", "#a855f7"]
    for color, (label, values) in zip(palette, costs.items()):
        ax2.fill_between(n_sample, bottom, bottom + values, color=color, alpha=0.65, label=label)
        bottom += np.array(values)
    ax2.set_xlabel("Number of Agents (N)")
    ax2.set_ylabel("Cost Components")
    ax2.set_title("(b) Cost Breakdown")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Benefit forms.
    ax3 = fig.add_subplot(gs[1, 0])
    sat_b = [benefit_saturating(int(n), benefit_params) for n in n_vals]
    lin_b = [benefit_linear(int(n), benefit_params) for n in n_vals]
    net_b = [benefit_network(int(n), benefit_params) for n in n_vals]
    ax3.plot(n_vals, sat_b, color="#1f5a99", linewidth=2, label="Saturating")
    ax3.plot(n_vals, lin_b, color="#15803d", linewidth=2, linestyle="--", label="Linear")
    ax3.plot(n_vals, net_b, color="#d97706", linewidth=2, linestyle=":", label="Network")
    ax3.set_xlabel("Number of Agents (N)")
    ax3.set_ylabel("Benefit B(N)")
    ax3.set_title("(c) Benefit Growth Forms")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel (d): Protocol efficiency impact.
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_vals, base_saturating, color="#1f5a99", linewidth=2, label="Baseline (p=1.5)")
    ax4.plot(n_vals, net_series(cost_eff, benefit_params, "saturating"), color="#15803d", linestyle="--", linewidth=2, label="Efficient (p=1)")
    ax4.plot(n_vals, net_series(cost_ineff, benefit_params, "saturating"), color="#d97706", linestyle=":", linewidth=2, label="Inefficient (p=2)")
    ax4.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax4.set_xlabel("Number of Agents (N)")
    ax4.set_ylabel("Net Value J(N)")
    ax4.set_title("(d) Protocol Efficiency Impact")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel (e): Optimal N across scenarios.
    ax5 = fig.add_subplot(gs[1, 2])
    scenarios = [
        ("Base", base_saturating),
        ("High Ben", net_series(cost_params, benefit_high, "saturating")),
        ("Low Ben", net_series(cost_params, benefit_low, "saturating")),
        ("Eff Cons", net_series(cost_eff, benefit_params, "saturating")),
        ("Ineff Cons", net_series(cost_ineff, benefit_params, "saturating")),
        ("Linear", net_series(cost_params, benefit_params, "linear")),
        ("Network", net_series(cost_params, benefit_params, "network")),
    ]
    optimal_ns = [n_vals[int(np.argmax(vals))] for _, vals in scenarios]
    labels = [label for label, _ in scenarios]
    bars = ax5.bar(labels, optimal_ns, color="#94a3b8", alpha=0.8)
    ax5.axhline(6, color="gray", linestyle="--", linewidth=1, label="N=6")
    ax5.set_ylabel("Optimal N*")
    ax5.set_title("(e) Optimal Group Size Across Scenarios")
    ax5.set_ylim([0, max(optimal_ns) + 4])
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.tick_params(axis="x", labelsize=7, rotation=20)
    for bar, val in zip(bars, optimal_ns):
        ax5.text(bar.get_x() + bar.get_width() / 2.0, val + 0.2, f"{val}", ha="center", fontsize=8)

    fig.suptitle("Communication and Coordination Scaling Analysis", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.92, wspace=0.3, hspace=0.35)
    _save(fig, path)


def figure_v_dynamic_by_n(path: str, results_path: str) -> None:
    """Stability volume (V_dynamic) progression by fixed N from robustness analysis."""
    import json

    apply_style()

    # Load results from revision experiments
    with open(results_path) as f:
        data = json.load(f)

    # Extract V_dynamic by N
    uniform_run = data['runs'].get('uniform')
    if uniform_run is None:
        raise ValueError("No 'uniform' run found in results")

    volume_by_n = uniform_run.get('volume_by_N')
    if volume_by_n is None:
        raise ValueError("No volume_by_N found in uniform run")

    n_values = []
    v_dynamic_values = []
    v_c2_values = []

    for n_str, metrics in volume_by_n.items():
        n_values.append(int(n_str))
        v_dynamic_values.append(metrics['V_dynamic'])
        v_c2_values.append(metrics['V_C2'])

    # Sort by N
    sorted_idx = np.argsort(n_values)
    n_values = np.array(n_values)[sorted_idx]
    v_dynamic_values = np.array(v_dynamic_values)[sorted_idx]
    v_c2_values = np.array(v_c2_values)[sorted_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): V_dynamic progression
    ax1.plot(n_values, v_dynamic_values, marker='o', color='#1f5a99', linewidth=2, markersize=8, label=r'$V_{\mathrm{dynamic}}$ (all conditions)')
    ax1.plot(n_values, v_c2_values, marker='s', color='#15803d', linewidth=2, markersize=6, linestyle='--', label=r'$V_{C2^*}$ (participation only)')
    ax1.axhline(0.95, color='#dc2626', linestyle=':', linewidth=1.5, label='95% threshold')
    ax1.axhline(0.979, color='#059669', linestyle=':', linewidth=1.5, label='Saturation (~98%)')
    ax1.axvline(4, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axvline(6, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(4, 0.77, 'N=4\n(95%)', ha='center', fontsize=9, color='#4b5563')
    ax1.text(6, 0.77, 'N≥6\n(saturated)', ha='center', fontsize=9, color='#4b5563')
    ax1.set_xlabel('Number of Agents (N)', fontsize=11)
    ax1.set_ylabel('Stability Volume (prior-conditional mass)', fontsize=11)
    ax1.set_title('(a) Dynamic Stability Volume vs Group Size', fontsize=12)
    ax1.set_ylim([0.7, 1.02])
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Marginal gains
    marginal_gains = np.diff(v_dynamic_values)
    n_mid = n_values[:-1] + 0.5
    ax2.bar(n_mid, marginal_gains, width=0.8, color='#60a5fa', alpha=0.7, edgecolor='#1f5a99', linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Transition (N→N+1)', fontsize=11)
    ax2.set_ylabel('Marginal Stability Gain', fontsize=11)
    ax2.set_title('(b) Diminishing Marginal Returns', fontsize=12)
    ax2.set_xticks(n_mid)
    ax2.set_xticklabels([f'{int(n)}→{int(n)+1}' for n in n_values[:-1]], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Stability Volume by Group Size (Robustness Analysis)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, path)


def figure_margin_distributions(path: str, results_path: str) -> None:
    """Margin distribution quantiles showing robustness under declared prior."""
    import json

    apply_style()

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Extract margin quantiles from uniform prior run
    uniform_run = data['runs'].get('uniform')
    if uniform_run is None:
        raise ValueError("No 'uniform' run found in results")

    margins = uniform_run['metrics']['quantiles']

    margin_names = ['c1_margin', 'c2_margin', 's_static', 'delta_margin', 's_dynamic']
    labels = [r'$C1^*$ margin', r'$C2^*$ margin', r'$S_{\mathrm{static}}$', r'$\delta$ margin', r'$S_{\mathrm{dynamic}}$']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (margin_name, label, color) in enumerate(zip(margin_names, labels, colors)):
        ax = axes[idx]
        dist_data = margins[margin_name]

        q05 = dist_data['q05']
        q50 = dist_data['q50']
        q95 = dist_data['q95']
        mean = dist_data['mean']
        std = dist_data['std']

        # Box plot style visualization
        positions = [1]
        ax.boxplot([[q05, q50, q95]], positions=positions, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color=color, linewidth=1.5),
                   capprops=dict(color=color, linewidth=1.5))

        # Add mean marker
        ax.plot(1, mean, 'D', color='black', markersize=8, label=f'Mean: {mean:.3f}', zorder=10)

        # Add zero line
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stability threshold')

        # Annotations
        ax.text(0.5, q05, f'5%: {q05:.3f}', ha='right', va='center', fontsize=9)
        ax.text(0.5, q50, f'50%: {q50:.3f}', ha='right', va='center', fontsize=9, fontweight='bold')
        ax.text(0.5, q95, f'95%: {q95:.3f}', ha='right', va='center', fontsize=9)

        ax.set_ylabel('Margin value', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove extra subplot
    fig.delaxes(axes[5])

    fig.suptitle('Stability Margin Distributions (Uniform Prior, N=2-10)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, path)


def figure_sobol_by_n(path: str, results_path: str) -> None:
    """Sobol sensitivity showing regime shift from β_Ω dominance to δ dominance."""
    import json

    apply_style()

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Extract Sobol by N
    uniform_run = data['runs'].get('uniform')
    if uniform_run is None:
        raise ValueError("No 'uniform' run found in results")

    sobol_by_n = uniform_run.get('sobol_per_N')
    if sobol_by_n is None:
        raise ValueError("No sobol_per_N found in uniform run")

    # Parse data
    n_values = sorted([int(n) for n in sobol_by_n.keys()])
    params = ['beta_alpha', 'beta_kappa', 'beta_D', 'beta_Omega', 'beta_ell', 'N', 'delta']
    param_labels = [r'$\beta_\alpha$', r'$\beta_\kappa$', r'$\beta_D$', r'$\beta_\Omega$', r'$\beta_\ell$', r'$N$', r'$\delta$']

    # Focus on key parameters
    key_params = ['beta_Omega', 'delta', 'beta_ell']
    key_labels = [r'$\beta_\Omega$', r'$\delta$', r'$\beta_\ell$']
    key_colors = ['#10b981', '#ef4444', '#f59e0b']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Total-order indices by N
    for param, label, color in zip(key_params, key_labels, key_colors):
        st_values = []
        for n in n_values:
            sobol_data = sobol_by_n[str(n)]
            param_idx = sobol_data['parameter_names'].index(param)
            output_idx = sobol_data['output_names'].index('s_dynamic')
            st_val = sobol_data['S_total'][param_idx][output_idx]
            st_values.append(st_val)

        ax1.plot(n_values, st_values, marker='o', linewidth=2.5, markersize=8, label=label, color=color)

    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='ST = 0.5')
    ax1.axvline(4, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(4, 0.05, 'N=4', ha='center', fontsize=9, color='#4b5563')
    ax1.set_xlabel('Group Size (N)', fontsize=11)
    ax1.set_ylabel('Total-Order Sobol Index (ST)', fontsize=11)
    ax1.set_title(r'(a) Sensitivity Regime Shift: $\beta_\Omega$ → $\delta$', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.legend(loc='right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Stacked bars showing full parameter breakdown
    width = 0.6
    x_pos = np.arange(len(n_values))

    # Get all ST values for all parameters
    all_st = {param: [] for param in params}
    for n in n_values:
        sobol_data = sobol_by_n[str(n)]
        for param in params:
            param_idx = sobol_data['parameter_names'].index(param)
            output_idx = sobol_data['output_names'].index('s_dynamic')
            st_val = sobol_data['S_total'][param_idx][output_idx]
            all_st[param].append(st_val)

    # Normalize to show relative importance
    total_st = np.array([sum(all_st[p][i] for p in params) for i in range(len(n_values))])
    normalized_st = {param: np.array(all_st[param]) / total_st for param in params}

    bottom = np.zeros(len(n_values))
    param_colors_all = ['#93c5fd', '#a5b4fc', '#fca5a5', '#6ee7b7', '#fde047', '#d4d4d4', '#dc2626']

    for param, label, color in zip(params, param_labels, param_colors_all):
        values = normalized_st[param]
        ax2.bar(x_pos, values, width, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values

    ax2.set_xlabel('Group Size (N)', fontsize=11)
    ax2.set_ylabel('Relative Contribution (normalized ST)', fontsize=11)
    ax2.set_title('(b) Parameter Importance Breakdown by N', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(n_values)
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(r'Global Sensitivity Analysis: Dominance Shift with Group Size', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, path)


def figure_summary(path: str, results_path: str) -> None:
    """Comprehensive summary figure guiding reader through the full analysis."""
    import json
    from matplotlib import gridspec

    apply_style()

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Extract data
    uniform_run = data['runs'].get('uniform')
    if uniform_run is None:
        raise ValueError("No 'uniform' run found in results")

    volume = uniform_run['metrics']['volume']
    by_n = uniform_run['volume_by_N']
    margins = uniform_run['metrics']['quantiles']
    sobol_per_n = uniform_run.get('sobol_per_N', {})

    params = anchor_params()
    ranges = plausible_ranges()

    # Create 2x3 grid
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Parameter space with regime boundaries
    ax1 = fig.add_subplot(gs[0, 0])
    beta_kappa = np.linspace(0.3, 3.5, 100)
    beta_D_boundary = params.beta_alpha + beta_kappa
    ax1.plot(beta_kappa, beta_D_boundary, 'k-', linewidth=2, label=r'$C1^*$ boundary')
    ax1.fill_between(beta_kappa, 0, beta_D_boundary, color='#b7e3b3', alpha=0.3, label='Stable')
    ax1.fill_between(beta_kappa, beta_D_boundary, 3, color='#f4b6b6', alpha=0.3, label='Unstable')

    # Patience-free boundary: β_D = 0.7 (where β_α ≥ β_D → δ*=0)
    ax1.axhline(0.7, color='#10b981', linestyle='--', linewidth=2.5, label=r'Patience-free ($\delta^*=0$)', zorder=9)
    ax1.fill_between([0.3, 3.5], 0, 0.7, color='#10b981', alpha=0.15, zorder=0)

    ax1.add_patch(plt.Rectangle((ranges.beta_kappa[0], ranges.beta_D[0]),
                                 ranges.beta_kappa[1] - ranges.beta_kappa[0],
                                 ranges.beta_D[1] - ranges.beta_D[0],
                                 facecolor='yellow', alpha=0.2, edgecolor='orange', linewidth=2, label='Plausible range'))
    ax1.set_xlabel(r'$\beta_\kappa$ (deterrence)', fontsize=10)
    ax1.set_ylabel(r'$\beta_D$ (defection)', fontsize=10)
    ax1.set_title('(1) Parameter Space', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.3, 3.5])
    ax1.set_ylim([0, 1.5])

    # Panel 2: Overall stability volumes
    ax2 = fig.add_subplot(gs[0, 1])
    conditions = ['C1*', 'C1**', 'C2*', 'Static', 'Dynamic']
    volumes = [volume['V_C1'], volume['V_C1_dynamic'], volume['V_C2'], volume['V_static'], volume['V_dynamic']]
    colors_bar = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    bars = ax2.barh(conditions, volumes, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.axvline(0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    for bar, vol in zip(bars, volumes):
        ax2.text(vol + 0.01, bar.get_y() + bar.get_height()/2, f'{vol:.3f}',
                va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Stability Volume (prior mass)', fontsize=10)
    ax2.set_title('(2) Overall Robustness', fontsize=11, fontweight='bold')
    ax2.set_xlim([0.7, 1.02])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: V_dynamic by N with regime boundaries
    ax3 = fig.add_subplot(gs[0, 2])
    n_vals = sorted([int(n) for n in by_n.keys()])
    v_dyn = [by_n[str(n)]['V_dynamic'] for n in n_vals]
    ax3.plot(n_vals, v_dyn, marker='o', color='#ef4444', linewidth=2.5, markersize=8)
    ax3.axhline(0.95, color='red', linestyle=':', linewidth=1.5, label='95% threshold')

    # N=4 threshold: 95% crossing
    ax3.axvline(4, color='#f59e0b', linestyle='--', linewidth=2.5, alpha=0.8, label='N=4 (95% crossing)')
    ax3.axvspan(1, 4, color='#fef3c7', alpha=0.2)

    # N≥6 band-robust regime
    ax3.axvline(6, color='#10b981', linestyle='--', linewidth=2.5, alpha=0.8, label='N≥6 (band-robust)')
    ax3.axvspan(6, max(n_vals), color='#10b981', alpha=0.15)
    ax3.set_xlabel('Group Size (N)', fontsize=10)
    ax3.set_ylabel(r'$V_{\mathrm{dynamic}}$', fontsize=10)
    ax3.set_title('(3) Thresholds: N=4 (95%), N≥6 (saturated)', fontsize=11, fontweight='bold')
    ax3.set_ylim([0.7, 1.02])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Margin distributions (compact)
    ax4 = fig.add_subplot(gs[1, 0])
    margin_types = ['s_dynamic', 's_static', 'c1_margin', 'c2_margin']
    margin_labels_short = [r'$S_{dyn}$', r'$S_{stat}$', r'$M_{C1}$', r'$M_{C2}$']
    for i, (mtype, mlabel) in enumerate(zip(margin_types, margin_labels_short)):
        q50 = margins[mtype]['q50']
        q05 = margins[mtype]['q05']
        q95 = margins[mtype]['q95']
        ax4.errorbar(i, q50, yerr=[[q50-q05], [q95-q50]], fmt='o', markersize=8, capsize=5, capthick=2, label=mlabel)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xticks(range(len(margin_types)))
    ax4.set_xticklabels(margin_labels_short, fontsize=10)
    ax4.set_ylabel('Margin (5%/50%/95%)', fontsize=10)
    ax4.set_title('(4) Margin Distributions', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Sobol regime shift
    ax5 = fig.add_subplot(gs[1, 1])
    if sobol_per_n:
        n_sobol = sorted([int(n) for n in sobol_per_n.keys()])
        beta_omega_st = []
        delta_st = []
        for n in n_sobol:
            sdata = sobol_per_n[str(n)]
            bo_idx = sdata['parameter_names'].index('beta_Omega')
            d_idx = sdata['parameter_names'].index('delta')
            out_idx = sdata['output_names'].index('s_dynamic')
            beta_omega_st.append(sdata['S_total'][bo_idx][out_idx])
            delta_st.append(sdata['S_total'][d_idx][out_idx])

        ax5.plot(n_sobol, beta_omega_st, marker='o', linewidth=2, markersize=7, label=r'$\beta_\Omega$', color='#10b981')
        ax5.plot(n_sobol, delta_st, marker='s', linewidth=2, markersize=7, label=r'$\delta$', color='#ef4444')
        ax5.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
        ax5.set_xlabel('Group Size (N)', fontsize=10)
        ax5.set_ylabel('Total Sobol Index (ST)', fontsize=10)
        ax5.set_title(r'(5) Sensitivity Shift: $\beta_\Omega$ → $\delta$', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

    # Panel 6: Key takeaways (text)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    takeaway_text = [
        'KEY FINDINGS:',
        '',
        '• N=1-3: Risky (<95% stable)',
        '• N=4: Crosses 95% threshold',
        '• N≥6: Saturated (~98%)',
        '',
        '• Small N: β_Ω dominates',
        '• Large N: δ dominates',
        '',
        '• δ* = 0 achievable',
        '  (when β_α ≥ β_D)',
        '',
        '• V_dynamic = 94.2%',
        '  (under uniform prior)',
    ]
    ax6.text(0.1, 0.9, '\n'.join(takeaway_text), transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f9ff', edgecolor='#3b82f6', linewidth=2))

    fig.suptitle('Summary: Coordination Stability Analysis', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, path)
