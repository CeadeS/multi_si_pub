#!/usr/bin/env python3
"""Corrected-model baseline with perfect monitoring (q pinned to 0).

Re-runs the robustness metrics of run_revision_experiments.py with the
detection-failure probability fixed at q=0 (q_detect=(0.0, 0.0)), so the
difference to the main run isolates the effect of sampling q ~ U[0, 0.3].
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from repro.model import ParamRanges, plausible_ranges
from repro.robustness_metrics import PriorSpec, run_metrics, volume_metrics_by_fixed_N


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="documentation/revision_experiments_jair/RESULTS_q0.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=200_000)
    args = parser.parse_args()

    base = plausible_ranges()
    ranges_q0 = ParamRanges(
        beta_kappa=base.beta_kappa,
        beta_alpha=base.beta_alpha,
        beta_D=base.beta_D,
        beta_Omega=base.beta_Omega,
        beta_ell=base.beta_ell,
        n_agents=base.n_agents,
        q_detect=(0.0, 0.0),
    )
    prior = PriorSpec(
        name="uniform_q0",
        ranges=ranges_q0,
        delta_range=(0.1, 0.95),
        seed=args.seed,
    )

    metrics = run_metrics(prior=prior, n_samples=args.n_samples)
    volume_by_n = volume_metrics_by_fixed_N(
        prior=prior,
        n_samples=max(50_000, args.n_samples // 4),
        seed=args.seed + 3,
    )

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "note": "Corrected model, perfect-monitoring baseline (q_detect pinned to 0).",
        },
        "metrics": metrics,
        "volume_by_N": volume_by_n,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(metrics["volume"], indent=2))


if __name__ == "__main__":
    main()
