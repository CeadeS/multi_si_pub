"""Matplotlib styling helpers for reproduction figures."""

import matplotlib.pyplot as plt


def apply_style() -> None:
    """Apply a lightweight style for consistent plots."""

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
