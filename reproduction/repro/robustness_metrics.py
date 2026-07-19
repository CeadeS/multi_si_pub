"""Robustness metrics and Sobol sensitivity for mediator stability conditions.

This module intentionally stays close to the paper's closed-form inequalities
(all stated under imperfect public monitoring with detection-failure
probability q; q=0 recovers perfect monitoring):

- C1*: beta_alpha + (1-q)*beta_kappa >= beta_D
- C1**: delta >= delta_crit(beta_alpha, beta_kappa, beta_D, q)
        = g / (g + (1-q)*(beta_alpha + beta_kappa)), g = beta_D - beta_alpha
- C2*: beta_Omega / (1 - delta) >= beta_ell / N   (present value of membership
        vs one-time removal gain; delta=0 recovers the conservative flow bound)

It provides:
- "volume" (uniform prior mass) estimates for stability regions
- margin distributions (quantiles of stability margins)
- worst-case (robust) design requirements for beta_alpha(delta) and beta_Omega(N)
- Sobol first/total-order indices under an explicit prior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .model import ParamRanges


class PriorFamily(str, Enum):
    """Supported independent prior families (truncated to declared ranges)."""

    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"


@dataclass(frozen=True)
class PriorSpec:
    """Independent priors over model inputs.

    By default we use uniform priors over declared ranges (maximum entropy given bounds).
    Families can be overridden per parameter via `families`.
    """

    name: str
    ranges: ParamRanges
    delta_range: tuple[float, float] = (0.1, 0.95)
    families: dict[str, PriorFamily] = field(default_factory=dict)
    seed: int = 0


def _validate_delta_range(delta_range: tuple[float, float]) -> tuple[float, float]:
    delta_min, delta_max = delta_range
    if not (0.0 <= delta_min <= 1.0 and 0.0 <= delta_max <= 1.0):
        raise ValueError("delta_range must be within [0, 1]")
    if delta_min >= delta_max:
        raise ValueError("delta_range must satisfy delta_min < delta_max")
    return delta_min, delta_max


def _normal_sigma(mean: float, min_val: float, max_val: float) -> float:
    width = min(mean - min_val, max_val - mean)
    return max(width / 2.0, mean * 0.05)


def _lognormal_sigma(mean: float, min_val: float, max_val: float) -> float:
    lo = np.log(mean / min_val)
    hi = np.log(max_val / mean)
    width = min(lo, hi)
    return max(width / 2.0, 0.08)


def _sample_param_vector(
    rng: np.random.Generator,
    min_val: float,
    max_val: float,
    mean: float,
    *,
    family: PriorFamily,
    n_samples: int,
) -> np.ndarray:
    if family == PriorFamily.UNIFORM:
        return rng.uniform(min_val, max_val, size=n_samples)
    if family == PriorFamily.LOG_UNIFORM:
        return np.exp(rng.uniform(np.log(min_val), np.log(max_val), size=n_samples))
    if family == PriorFamily.NORMAL:
        sigma = _normal_sigma(mean, min_val, max_val)
        draws = rng.normal(mean, sigma, size=n_samples)
        return np.clip(draws, min_val, max_val)
    if family == PriorFamily.LOG_NORMAL:
        sigma = _lognormal_sigma(mean, min_val, max_val)
        mu = np.log(mean)
        draws = rng.lognormal(mean=mu, sigma=sigma, size=n_samples)
        return np.clip(draws, min_val, max_val)
    raise ValueError(f"Unsupported prior family: {family}")


def _family_for(prior: PriorSpec, key: str) -> PriorFamily:
    return prior.families.get(key, PriorFamily.UNIFORM)


def _sample_prior(
    rng: np.random.Generator,
    prior: PriorSpec,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Sample independent inputs under the declared independent prior."""

    ranges = prior.ranges
    delta_min, delta_max = _validate_delta_range(prior.delta_range)

    beta_alpha = _sample_param_vector(
        rng,
        ranges.beta_alpha[0],
        ranges.beta_alpha[1],
        mean=0.5 * (ranges.beta_alpha[0] + ranges.beta_alpha[1]),
        family=_family_for(prior, "beta_alpha"),
        n_samples=n_samples,
    )
    beta_kappa = _sample_param_vector(
        rng,
        ranges.beta_kappa[0],
        ranges.beta_kappa[1],
        mean=0.5 * (ranges.beta_kappa[0] + ranges.beta_kappa[1]),
        family=_family_for(prior, "beta_kappa"),
        n_samples=n_samples,
    )
    beta_D = _sample_param_vector(
        rng,
        ranges.beta_D[0],
        ranges.beta_D[1],
        mean=0.5 * (ranges.beta_D[0] + ranges.beta_D[1]),
        family=_family_for(prior, "beta_D"),
        n_samples=n_samples,
    )
    beta_Omega = _sample_param_vector(
        rng,
        ranges.beta_Omega[0],
        ranges.beta_Omega[1],
        mean=0.5 * (ranges.beta_Omega[0] + ranges.beta_Omega[1]),
        family=_family_for(prior, "beta_Omega"),
        n_samples=n_samples,
    )
    beta_ell = _sample_param_vector(
        rng,
        ranges.beta_ell[0],
        ranges.beta_ell[1],
        mean=0.5 * (ranges.beta_ell[0] + ranges.beta_ell[1]),
        family=_family_for(prior, "beta_ell"),
        n_samples=n_samples,
    )

    # Treat N as a continuous uncertainty and round inside the model.
    # Sampling on [Nmin-0.5, Nmax+0.5] makes each integer receive equal mass.
    n_min, n_max = ranges.n_agents
    n_cont = rng.uniform(float(n_min) - 0.5, float(n_max) + 0.5, size=n_samples)
    n_agents = np.rint(n_cont).astype(int)
    n_agents = np.clip(n_agents, n_min, n_max)

    delta = _sample_param_vector(
        rng,
        delta_min,
        delta_max,
        mean=0.5 * (delta_min + delta_max),
        family=_family_for(prior, "delta"),
        n_samples=n_samples,
    )

    q_lo, q_hi = ranges.q_detect
    if q_hi > q_lo:
        q_detect = _sample_param_vector(
            rng,
            q_lo,
            q_hi,
            mean=0.5 * (q_lo + q_hi),
            family=_family_for(prior, "q_detect"),
            n_samples=n_samples,
        )
    else:
        q_detect = np.full(n_samples, q_lo, dtype=float)

    return {
        "beta_alpha": beta_alpha,
        "beta_kappa": beta_kappa,
        "beta_D": beta_D,
        "beta_Omega": beta_Omega,
        "beta_ell": beta_ell,
        "n_agents": n_agents,
        "delta": delta,
        "q_detect": q_detect,
    }


def _delta_crit_vector(
    beta_alpha: np.ndarray,
    beta_kappa: np.ndarray,
    beta_D: np.ndarray,
    q_detect: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Vectorized C1** threshold under imperfect public monitoring.

    delta*(q) = g / (g + (1-q) * (beta_alpha + beta_kappa)), g = beta_D - beta_alpha.
    """

    numerator = beta_D - beta_alpha
    denominator = numerator + (1.0 - np.asarray(q_detect, dtype=float)) * (beta_alpha + beta_kappa)
    value = np.ones_like(numerator, dtype=float)
    mask = denominator > 0
    value[mask] = numerator[mask] / denominator[mask]
    value = np.clip(value, 0.0, 1.0)
    value[numerator <= 0] = 0.0
    return value


def margins_from_samples(samples: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute core margins and aggregate stability margins from sampled inputs."""

    beta_alpha = np.asarray(samples["beta_alpha"], dtype=float)
    beta_kappa = np.asarray(samples["beta_kappa"], dtype=float)
    beta_D = np.asarray(samples["beta_D"], dtype=float)
    beta_Omega = np.asarray(samples["beta_Omega"], dtype=float)
    beta_ell = np.asarray(samples["beta_ell"], dtype=float)
    n_agents = np.asarray(samples["n_agents"], dtype=float)
    delta = np.asarray(samples["delta"], dtype=float)
    if "q_detect" in samples:
        q_detect = np.asarray(samples["q_detect"], dtype=float)
    else:
        q_detect = np.zeros_like(delta)

    c1_margin = beta_alpha + (1.0 - q_detect) * beta_kappa - beta_D
    # Present-value participation constraint in flow-equivalent form:
    # beta_Omega/(1-delta) >= beta_ell/N  <=>  beta_Omega - (1-delta)*beta_ell/N >= 0.
    c2_margin = beta_Omega - (1.0 - delta) * beta_ell / n_agents
    # Immediate-enforcement object: C1*(rho=1) AND C2*.
    s_static = np.minimum(c1_margin, c2_margin)

    delta_crit = _delta_crit_vector(
        beta_alpha=beta_alpha, beta_kappa=beta_kappa, beta_D=beta_D, q_detect=q_detect
    )
    delta_margin = delta - delta_crit
    # Continuation-enforcement equilibrium object of the main theorem:
    # C1**(q) AND C2*. (The former cross-regime min(s_static, delta_margin)
    # mixed rho=1 and rho=0 conditions; it is retained only as V_cross.)
    s_dynamic = np.minimum(c2_margin, delta_margin)
    s_cross = np.minimum(s_static, delta_margin)

    # Full immediate-enforcement equilibrium: delta >= delta*(q, s) with
    # s = rho*beta_kappa at rho = 1, residual gain g_s = max(0, beta_D -
    # beta_alpha - (1-q)*beta_kappa). P[C1* & C2*] (s_static) is the myopic
    # subset of this regime; the paper reports both.
    g_s = np.maximum(0.0, beta_D - beta_alpha - (1.0 - q_detect) * beta_kappa)
    kappa_eff = beta_alpha + beta_kappa
    delta_crit_s = np.where(g_s <= 0.0, 0.0, g_s / (g_s + (1.0 - q_detect) * kappa_eff))
    s_immediate_eq = np.minimum(delta - delta_crit_s, c2_margin)

    return {
        "c1_margin": c1_margin,
        "c2_margin": c2_margin,
        "s_static": s_static,
        "delta_crit": delta_crit,
        "delta_margin": delta_margin,
        "s_dynamic": s_dynamic,
        "s_cross": s_cross,
        "delta_crit_s": delta_crit_s,
        "s_immediate_eq": s_immediate_eq,
    }


def quantiles(values: np.ndarray, probs: Sequence[float] = (0.05, 0.5, 0.95)) -> Dict[str, float]:
    """Compute selected quantiles."""

    qs = np.quantile(values, probs)
    return {f"q{int(p*100):02d}": float(q) for p, q in zip(probs, qs)}


def distribution_summary(values: np.ndarray, probs: Sequence[float] = (0.05, 0.5, 0.95)) -> Dict[str, float]:
    """Quantiles + mean/std summary (JSON-friendly)."""

    summary = quantiles(values, probs=probs)
    summary["mean"] = float(np.mean(values))
    summary["std"] = float(np.std(values, ddof=1))
    return summary


def volume_metrics(margins: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Uniform-prior stability mass for individual conditions and combined."""

    c1_ok = margins["c1_margin"] >= 0.0
    c2_ok = margins["c2_margin"] >= 0.0
    c1_dynamic_ok = margins["delta_margin"] >= 0.0
    static_ok = margins["s_static"] >= 0.0
    dynamic_ok = margins["s_dynamic"] >= 0.0

    out = {
        "V_C1": float(np.mean(c1_ok)),
        "V_C1_dynamic": float(np.mean(c1_dynamic_ok)),
        "V_C2": float(np.mean(c2_ok)),
        "V_myopic": float(np.mean(static_ok)),
        "V_static": float(np.mean(static_ok)),  # legacy alias for V_myopic
        "V_dynamic": float(np.mean(dynamic_ok)),
    }
    if "s_cross" in margins:
        out["V_cross"] = float(np.mean(margins["s_cross"] >= 0.0))
    if "s_immediate_eq" in margins:
        # Full immediate-regime equilibrium volume P[delta >= delta*(q,s) & C2*];
        # V_myopic above is its myopic (discount-free) subset.
        out["V_immediate"] = float(np.mean(margins["s_immediate_eq"] >= 0.0))
    return out


def volume_metrics_by_fixed_N(
    prior: PriorSpec,
    n_samples: int,
    *,
    n_values: Iterable[int] | None = None,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Estimate volume metrics conditional on fixed group size N.

    This answers: for a given N, how often do C2* and the combined stability
    constraints hold under the declared uncertainty model for the remaining inputs?
    """

    ranges = prior.ranges
    n_min, n_max = ranges.n_agents
    if n_values is None:
        n_values = range(n_min, n_max + 1)

    rng = np.random.default_rng(prior.seed if seed is None else seed)
    base = _sample_prior(rng=rng, prior=prior, n_samples=n_samples)

    results: Dict[str, Dict[str, float]] = {}
    for n_agents in n_values:
        n_int = int(n_agents)
        if not (n_min <= n_int <= n_max):
            raise ValueError(f"N={n_int} outside declared range [{n_min}, {n_max}]")
        samples = dict(base)
        samples["n_agents"] = np.full(n_samples, n_int, dtype=int)
        margins = margins_from_samples(samples)
        results[str(n_int)] = volume_metrics(margins)
    return results


def worst_case_beta_alpha_required(
    ranges: ParamRanges,
    deltas: Iterable[float],
    *,
    include_c1: bool = True,
    include_c1_dynamic: bool = True,
) -> Dict[float, float]:
    """Worst-case beta_alpha needed to satisfy C1* and/or C1** over declared ranges.

    Worst case over the declared ranges: beta_D at its maximum, beta_kappa at
    its minimum, and detection failure q at its maximum.

    C1*(q):  beta_alpha >= beta_D - (1-q)*beta_kappa.
    C1**(q): delta >= g/(g + (1-q)(beta_alpha+beta_kappa)) inverts to
             beta_alpha >= [(1-delta)*beta_D - delta*(1-q)*beta_kappa] / (1 - delta*q).
    """

    beta_D_max = ranges.beta_D[1]
    beta_kappa_min = ranges.beta_kappa[0]
    q_max = ranges.q_detect[1]

    requirements: Dict[float, float] = {}
    for delta in deltas:
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1) for the worst-case formula")

        bounds = []
        if include_c1:
            bounds.append(beta_D_max - (1.0 - q_max) * beta_kappa_min)
        if include_c1_dynamic:
            bounds.append(
                ((1.0 - delta) * beta_D_max - delta * (1.0 - q_max) * beta_kappa_min)
                / (1.0 - delta * q_max)
            )
        requirements[float(delta)] = float(max(bounds))
    return requirements


def worst_case_beta_omega_required(
    ranges: ParamRanges, n_agents_values: Iterable[int], delta_min: float = 0.1
) -> Dict[int, float]:
    """Worst-case beta_Omega needed to satisfy the PV form of C2*.

    beta_Omega >= (1 - delta) * beta_ell / N is hardest at beta_ell max and
    delta min (delta_min defaults to the lower end of the declared delta range).
    """

    beta_ell_max = ranges.beta_ell[1]
    requirements: Dict[int, float] = {}
    for n_agents in n_agents_values:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        requirements[int(n_agents)] = float((1.0 - delta_min) * beta_ell_max / float(n_agents))
    return requirements


def _evaluate_outputs_from_unit_inputs(
    unit_inputs: np.ndarray,
    prior: PriorSpec,
) -> Dict[str, np.ndarray]:
    """Map unit hypercube inputs to model parameters and evaluate margins/outputs."""

    ranges = prior.ranges
    delta_min, delta_max = _validate_delta_range(prior.delta_range)

    if unit_inputs.ndim != 2:
        raise ValueError("unit_inputs must be a 2D array")
    if unit_inputs.shape[1] != 8:
        raise ValueError("Expected 8 inputs: alpha,kappa,D,Omega,ell,N,delta,q")

    u_alpha, u_kappa, u_D, u_Omega, u_ell, u_N, u_delta, u_q = unit_inputs.T

    def map_unit(u: np.ndarray, lo: float, hi: float, family: PriorFamily) -> np.ndarray:
        if family == PriorFamily.UNIFORM:
            return lo + u * (hi - lo)
        if family == PriorFamily.LOG_UNIFORM:
            return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
        raise ValueError(
            f"Sobol mapping supports only uniform/log_uniform; got {family.value}"
        )

    beta_alpha = map_unit(
        u_alpha, ranges.beta_alpha[0], ranges.beta_alpha[1], _family_for(prior, "beta_alpha")
    )
    beta_kappa = map_unit(
        u_kappa, ranges.beta_kappa[0], ranges.beta_kappa[1], _family_for(prior, "beta_kappa")
    )
    beta_D = map_unit(
        u_D, ranges.beta_D[0], ranges.beta_D[1], _family_for(prior, "beta_D")
    )
    beta_Omega = map_unit(
        u_Omega, ranges.beta_Omega[0], ranges.beta_Omega[1], _family_for(prior, "beta_Omega")
    )
    beta_ell = map_unit(
        u_ell, ranges.beta_ell[0], ranges.beta_ell[1], _family_for(prior, "beta_ell")
    )

    n_min, n_max = ranges.n_agents
    n_cont = (float(n_min) - 0.5) + u_N * ((float(n_max) + 0.5) - (float(n_min) - 0.5))
    n_agents = np.rint(n_cont).astype(int)
    n_agents = np.clip(n_agents, n_min, n_max)

    delta_family = _family_for(prior, "delta")
    if delta_family != PriorFamily.UNIFORM:
        raise ValueError("Sobol mapping currently assumes δ ~ uniform on [delta_min, delta_max]")
    delta = delta_min + u_delta * (delta_max - delta_min)

    q_lo, q_hi = ranges.q_detect
    q_detect = q_lo + u_q * (q_hi - q_lo)

    c1_margin = beta_alpha + (1.0 - q_detect) * beta_kappa - beta_D
    # Present-value participation constraint, flow-equivalent form (see margins_from_samples).
    c2_margin = beta_Omega - (1.0 - delta) * beta_ell / n_agents.astype(float)
    s_static = np.minimum(c1_margin, c2_margin)
    delta_margin = delta - _delta_crit_vector(
        beta_alpha=beta_alpha, beta_kappa=beta_kappa, beta_D=beta_D, q_detect=q_detect
    )
    s_dynamic = np.minimum(c2_margin, delta_margin)  # continuation object (C1** AND C2*)

    return {
        "c1_margin": c1_margin,
        "c2_margin": c2_margin,
        "s_static": s_static,
        "delta_margin": delta_margin,
        "s_dynamic": s_dynamic,
    }


def sobol_indices(
    prior: PriorSpec,
    n_base: int,
    *,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Compute Sobol first/total-order indices using Saltelli-style estimators.

    Inputs (d=8): beta_alpha, beta_kappa, beta_D, beta_Omega, beta_ell, N, delta, q_detect.
    Outputs: c1_margin, c2_margin, s_static, delta_margin, s_dynamic.
    """

    if n_base <= 0:
        raise ValueError("n_base must be positive")
    rng = np.random.default_rng(prior.seed if seed is None else seed)
    n_dim = 8

    A = rng.random((n_base, n_dim))
    B = rng.random((n_base, n_dim))

    fA = _evaluate_outputs_from_unit_inputs(A, prior)
    fB = _evaluate_outputs_from_unit_inputs(B, prior)

    output_names = ["c1_margin", "c2_margin", "s_static", "delta_margin", "s_dynamic"]
    parameter_names = ["beta_alpha", "beta_kappa", "beta_D", "beta_Omega", "beta_ell", "N", "delta", "q_detect"]

    fA_stack = np.column_stack([fA[name] for name in output_names])
    fB_stack = np.column_stack([fB[name] for name in output_names])
    f_all = np.vstack([fA_stack, fB_stack])

    variance = np.var(f_all, axis=0, ddof=1)
    variance = np.where(variance <= 0.0, np.nan, variance)

    first = np.zeros((n_dim, len(output_names)), dtype=float)
    total = np.zeros((n_dim, len(output_names)), dtype=float)

    for i in range(n_dim):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = _evaluate_outputs_from_unit_inputs(ABi, prior)
        fABi_stack = np.column_stack([fABi[name] for name in output_names])

        # Saltelli 2002 first-order estimator; Jansen 1999 total-order estimator.
        first[i, :] = np.mean(fB_stack * (fABi_stack - fA_stack), axis=0) / variance
        total[i, :] = np.mean((fA_stack - fABi_stack) ** 2, axis=0) / (2.0 * variance)

    return {
        "n_base": int(n_base),
        "parameter_names": parameter_names,
        "output_names": output_names,
        "S_first": first.tolist(),
        "S_total": total.tolist(),
    }


def run_metrics(
    prior: PriorSpec,
    n_samples: int,
    *,
    quantile_probs: Sequence[float] = (0.05, 0.5, 0.95),
) -> Dict[str, Any]:
    """Run robustness evaluation and return a JSON-ready dict."""

    rng = np.random.default_rng(prior.seed)
    samples = _sample_prior(rng=rng, prior=prior, n_samples=n_samples)
    margins = margins_from_samples(samples)

    results: Dict[str, Any] = {
        "prior": {
            "name": prior.name,
            "ranges": {
                "beta_D": list(prior.ranges.beta_D),
                "beta_kappa": list(prior.ranges.beta_kappa),
                "beta_alpha": list(prior.ranges.beta_alpha),
                "beta_Omega": list(prior.ranges.beta_Omega),
                "beta_ell": list(prior.ranges.beta_ell),
                "N": list(prior.ranges.n_agents),
                "delta": list(prior.delta_range),
                "q_detect": list(prior.ranges.q_detect),
            },
            "families": {k: v.value for k, v in sorted(prior.families.items())},
            "seed": int(prior.seed),
            "n_samples": int(n_samples),
        },
        "volume": volume_metrics(margins),
        "quantiles": {
            "c1_margin": distribution_summary(margins["c1_margin"], probs=quantile_probs),
            "c2_margin": distribution_summary(margins["c2_margin"], probs=quantile_probs),
            "s_static": distribution_summary(margins["s_static"], probs=quantile_probs),
            "delta_margin": distribution_summary(margins["delta_margin"], probs=quantile_probs),
            "s_dynamic": distribution_summary(margins["s_dynamic"], probs=quantile_probs),
        },
    }
    return results
