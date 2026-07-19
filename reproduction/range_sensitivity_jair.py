#!/usr/bin/env python3
"""Range-sensitivity analysis (±20% range variation) for V_dynamic.

For each of the eight model inputs (beta_D, beta_alpha, beta_kappa, beta_Omega,
beta_ell, delta, N, q_detect) the declared [lo, hi] range is shrunk ("narrow")
or widened ("wide") by 20% symmetrically about its midpoint, V_dynamic is
recomputed with 200,000 Monte Carlo samples (seed 0), and a signed sensitivity

    S = sign * max(|V_narrow - V_base|, |V_wide - V_base|) / (0.2 * V_base)

is reported, where sign is negative if narrowing reduces V_dynamic.

Caps: beta_alpha hi <= 1.0, delta hi <= 1.0, q lo >= 0.0, N bounds integer in
[2, 12]. All parameter lower bounds are floored at 0 where a negative bound
would be meaningless (none binds except q).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from repro.model import ParamRanges, plausible_ranges
from repro.robustness_metrics import (
    PriorSpec,
    _sample_prior,
    margins_from_samples,
    volume_metrics,
)

DELTA_RANGE_BASE = (0.1, 0.95)
N_SAMPLES = 200_000
SEED = 0

PARAMETERS = [
    "beta_D",
    "beta_alpha",
    "beta_kappa",
    "beta_Omega",
    "beta_ell",
    "delta",
    "N",
    "q_detect",
]


def _scaled_interval(lo: float, hi: float, factor: float) -> tuple[float, float]:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def _varied_prior(param: str, factor: float, seed: int = SEED) -> PriorSpec:
    """PriorSpec with one parameter's range scaled by `factor` about its midpoint."""

    ranges = plausible_ranges()
    delta_range = DELTA_RANGE_BASE

    if param == "delta":
        lo, hi = _scaled_interval(*DELTA_RANGE_BASE, factor)
        delta_range = (max(0.0, lo), min(1.0, hi))
    elif param == "N":
        n_lo, n_hi = ranges.n_agents
        lo, hi = _scaled_interval(float(n_lo), float(n_hi), factor)
        n_lo_new = int(np.clip(round(lo), 2, 12))
        n_hi_new = int(np.clip(round(hi), 2, 12))
        if n_hi_new <= n_lo_new:
            raise ValueError(f"Degenerate N range: ({n_lo_new}, {n_hi_new})")
        ranges = replace(ranges, n_agents=(n_lo_new, n_hi_new))
    else:
        key = param
        lo, hi = _scaled_interval(*getattr(ranges, key), factor)
        if param == "q_detect":
            lo = max(0.0, lo)
        if param == "beta_alpha":
            hi = min(1.0, hi)
        lo = max(0.0, lo)
        ranges = replace(ranges, **{key: (lo, hi)})

    return PriorSpec(
        name=f"uniform_{param}_x{factor:.1f}",
        ranges=ranges,
        delta_range=delta_range,
        seed=seed,
    )


def _v_dynamic(prior: PriorSpec, n_samples: int = N_SAMPLES) -> float:
    rng = np.random.default_rng(prior.seed)
    samples = _sample_prior(rng=rng, prior=prior, n_samples=n_samples)
    return volume_metrics(margins_from_samples(samples))["V_dynamic"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="documentation/revision_experiments_jair",
        help="Output directory for RANGE_SENSITIVITY.{md,json}",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_prior = PriorSpec(
        name="uniform_base",
        ranges=plausible_ranges(),
        delta_range=DELTA_RANGE_BASE,
        seed=SEED,
    )
    v_base = _v_dynamic(base_prior)

    rows: dict[str, dict] = {}
    for param in PARAMETERS:
        prior_narrow = _varied_prior(param, 0.8)
        prior_wide = _varied_prior(param, 1.2)
        v_narrow = _v_dynamic(prior_narrow)
        v_wide = _v_dynamic(prior_wide)
        d_narrow = v_narrow - v_base
        d_wide = v_wide - v_base
        magnitude = max(abs(d_narrow), abs(d_wide)) / (0.2 * v_base)
        sign = -1.0 if d_narrow < 0 else 1.0
        sensitivity = sign * magnitude

        def _fmt_range(prior: PriorSpec, key: str):
            if key == "delta":
                return list(prior.delta_range)
            if key == "N":
                return list(prior.ranges.n_agents)
            return list(getattr(prior.ranges, key))

        rows[param] = {
            "range_narrow": _fmt_range(prior_narrow, param),
            "range_wide": _fmt_range(prior_wide, param),
            "V_narrow": v_narrow,
            "V_wide": v_wide,
            "delta_narrow": d_narrow,
            "delta_wide": d_wide,
            "S": sensitivity,
        }

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "delta_range_base": list(DELTA_RANGE_BASE),
            "definition": (
                "S = sign * max(|V_narrow-V_base|, |V_wide-V_base|) / (0.2*V_base); "
                "sign < 0 iff narrowing reduces V_dynamic"
            ),
        },
        "V_base": v_base,
        "parameters": rows,
    }
    (out_dir / "RANGE_SENSITIVITY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Range Sensitivity of V_dynamic (±20% range variation)",
        "",
        f"- Generated: {payload['meta']['generated_at']}",
        f"- Samples per estimate: `{N_SAMPLES}` (seed `{SEED}`)",
        f"- Baseline V_dynamic: `{v_base:.4f}`",
        "- S = sign × max(|V_narrow−V_base|, |V_wide−V_base|) / (0.2·V_base);"
        " sign negative iff narrowing reduces V_dynamic.",
        "",
        "| Parameter | Narrow range | Wide range | V_narrow | V_wide | S |",
        "|---|---|---|---:|---:|---:|",
    ]
    for param in PARAMETERS:
        row = rows[param]
        nr = ", ".join(f"{v:g}" for v in row["range_narrow"])
        wr = ", ".join(f"{v:g}" for v in row["range_wide"])
        lines.append(
            f"| {param} | [{nr}] | [{wr}] | {row['V_narrow']:.4f} | "
            f"{row['V_wide']:.4f} | {row['S']:+.4f} |"
        )
    lines.append("")
    (out_dir / "RANGE_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"V_base = {v_base:.4f}")
    for param in PARAMETERS:
        row = rows[param]
        print(
            f"{param:>10s}: V_narrow={row['V_narrow']:.4f} "
            f"V_wide={row['V_wide']:.4f} S={row['S']:+.4f}"
        )


if __name__ == "__main__":
    main()
