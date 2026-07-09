"""Core model equations for figure reproduction."""

from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ModelParams:
    """Parameter bundle from paper/main.tex."""

    beta_kappa: float
    beta_alpha: float
    beta_D: float
    beta_Omega: float
    beta_ell: float
    G0: float = 1.0


@dataclass(frozen=True)
class ParamRanges:
    """Plausible parameter ranges derived from Appendix B."""

    beta_kappa: Tuple[float, float]
    beta_alpha: Tuple[float, float]
    beta_D: Tuple[float, float]
    beta_Omega: Tuple[float, float]
    beta_ell: Tuple[float, float]
    n_agents: Tuple[int, int]
    q_detect: Tuple[float, float] = (0.0, 0.3)


def anchor_params() -> ModelParams:
    """Anchors from the physical-scale discussion in paper/main.tex."""

    return ModelParams(
        beta_kappa=1.0,
        beta_alpha=0.7,
        beta_D=0.4,
        beta_Omega=1.0,
        beta_ell=1.5,
        G0=1.0,
    )


def plausible_ranges() -> ParamRanges:
    """Conservative plausible ranges for dimensionless parameters.

    Aligned with paper main text (nature_machine_intelligence_template.tex line 214
    and main_prototype.tex line 339). These are conservative estimates rather than
    order-of-magnitude bounds to ensure robustness claims are defensible.

    Note: beta_alpha upper bound is 0.9 (conservative) rather than 1.0
    (information-theoretic hard limit from Shannon capacity I(X;Y) <= H(Y)).
    """

    return ParamRanges(
        beta_kappa=(0.5, 3.0),
        beta_alpha=(0.3, 0.9),  # Conservative estimate (hard limit is 1.0)
        beta_D=(0.05, 1.0),
        beta_Omega=(0.05, 0.3),  # Conservative oversight value range
        beta_ell=(0.1, 0.5),     # Conservative removal benefit range
        n_agents=(2, 10),
        # Detection-failure probability for external-action monitoring.
        # q -> 0 is the sensor-bandwidth ideal (Appendix on observability);
        # the upper bound allows attribution ambiguity and adversarial
        # obfuscation in realistic deployments.
        q_detect=(0.0, 0.3),
    )


def replace_param(params: ModelParams, **updates: float) -> ModelParams:
    """Return a new params object with updated fields."""

    return replace(params, **updates)


def c1_margin(params: ModelParams, q: float = 0.0) -> float:
    """C1* margin: beta_alpha + (1-q)*beta_kappa - beta_D.

    Under imperfect public monitoring a deviation escapes retaliation with
    probability q, so expected deterrence is discounted by (1-q). q=0
    recovers the perfect-monitoring obedience constraint.
    """

    return params.beta_alpha + (1.0 - q) * params.beta_kappa - params.beta_D


def c2_margin(params: ModelParams, n_agents: int) -> float:
    """C2* margin: beta_Omega - beta_ell / N."""

    return params.beta_Omega - params.beta_ell / float(n_agents)


def stability_margin(params: ModelParams, n_agents: int) -> float:
    """Stability margin S = min(C1*, C2*)."""

    return min(c1_margin(params), c2_margin(params, n_agents))


def kappa_eff(params: ModelParams) -> float:
    """Effective per-period punishment swing: kappa_eff = beta_alpha + beta_kappa.

    A punished deviator both forgoes the coordination value beta_alpha and
    bears the conflict cost beta_kappa, so the per-period loss relative to
    the cooperative path is their sum.
    """

    return params.beta_alpha + params.beta_kappa


def delta_crit(params: ModelParams, q: float = 0.0) -> float:
    """C1** threshold for PPE sustainability under imperfect public monitoring.

    delta*(q) = g / (g + (1-q) * kappa_eff),  g = beta_D - beta_alpha,
    kappa_eff = beta_alpha + beta_kappa.

    Derived from grim-trigger present values with per-period detection
    failure probability q (deviations escape punishment with probability q
    each period; an undetected deviator optimally continues deviating).
    q=0 gives the perfect-monitoring threshold g/(beta_D + beta_kappa);
    q->1 drives the threshold to 1 (cooperation unsustainable).
    """

    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    numerator = params.beta_D - params.beta_alpha
    if numerator <= 0:
        return 0.0
    denominator = numerator + (1.0 - q) * kappa_eff(params)
    if denominator <= 0:
        return 1.0
    value = numerator / denominator
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def realized_payoffs(actions: Sequence[int], params: ModelParams) -> List[float]:
    """Realized payoffs for a single action profile (1=coop, 0=defect)."""

    m_defectors = len(actions) - sum(actions)
    payoffs: List[float] = []
    for action in actions:
        if action == 1:
            if m_defectors == 0:
                payoffs.append(params.G0 * (1.0 + params.beta_alpha))
            else:
                payoffs.append(params.G0)
        else:
            payoffs.append(
                params.G0
                * (1.0 + params.beta_D - params.beta_kappa * m_defectors)
            )
    return payoffs


def expected_welfare(actions: Sequence[int], params: ModelParams) -> float:
    """Total welfare for a single action profile."""

    return sum(realized_payoffs(actions, params))


def shapley_values(weights: Sequence[float], beta_omega: float, beta_synergy: float) -> List[float]:
    """Closed-form Shapley values for additive weights + symmetric synergy."""

    n_agents = len(weights)
    pair_term = beta_synergy * (n_agents - 1) / 2.0
    return [beta_omega * w + pair_term for w in weights]


def symmetric_shapley_per_agent(
    n_agents: int,
    beta_omega: float,
    beta_synergy: float,
    weight: float = 1.0,
) -> float:
    """Per-agent Shapley value for symmetric weights."""

    return beta_omega * weight + beta_synergy * (n_agents - 1) / 2.0


def coalition_value_by_size(
    size: int,
    beta_omega: float,
    beta_synergy: float,
    weight_avg: float = 1.0,
) -> float:
    """Super-additive coalition value for a given coalition size."""

    base = beta_omega * weight_avg * size
    synergy = beta_synergy * size * (size - 1) / 2.0
    return base + synergy


# ==============================================================================
# Functional Analysis (Section 2.X and Appendix A.5)
# ==============================================================================


def validate_params(params: ModelParams) -> bool:
    """Validate parameters respect physical and information-theoretic limits.

    Raises:
        ValueError: If any parameter violates fundamental limits
    """
    if params.beta_alpha > 1.0:
        raise ValueError(
            f"β_α = {params.beta_alpha} exceeds information-theoretic limit of 1.0"
        )
    if params.beta_alpha < 0:
        raise ValueError(f"β_α must be non-negative, got {params.beta_alpha}")
    if params.beta_kappa < 0:
        raise ValueError(f"β_κ must be non-negative, got {params.beta_kappa}")
    if params.beta_D < 0:
        raise ValueError(f"β_D must be non-negative, got {params.beta_D}")
    return True


def gap_from_patience_free(params: ModelParams) -> float:
    """Gap from patience-free boundary: g = β_D - β_α.

    When g > 0: patience-required regime
    When g = 0: phase transition (patience-free boundary)
    When g < 0: patience-free regime
    """
    return params.beta_D - params.beta_alpha


def patience_effectiveness(params: ModelParams, q: float = 0.0) -> float:
    """Effectiveness of deterrence: |∂δ*/∂β_κ| = g(1-q)/(g+(1-q)κ_eff)².

    Measures how much additional deterrence β_κ reduces the patience
    requirement. Diminishes hyperbolically in κ_eff (convex function) and
    is discounted by the detection probability (1-q).
    """
    g = gap_from_patience_free(params)
    if g <= 0:
        return 0.0  # Already patience-free
    denom = (g + (1.0 - q) * kappa_eff(params)) ** 2
    return g * (1.0 - q) / denom


def alpha_effectiveness(params: ModelParams, q: float = 0.0) -> float:
    """Effectiveness of coordination quality: |∂δ*/∂β_α|.

    β_α enters δ*(q) twice: it narrows the gap g = β_D - β_α and enlarges
    the effective punishment swing κ_eff = β_α + β_κ. Hence
    |∂δ*/∂β_α| = (1-q)(g + κ_eff)/(g + (1-q)κ_eff)², which exceeds
    |∂δ*/∂β_κ| by the factor (g + κ_eff)/g > 1 for all parameter values:
    signal quality is always the more effective lever.
    """
    g = gap_from_patience_free(params)
    if g <= 0:
        return 0.0
    k = kappa_eff(params)
    denom = (g + (1.0 - q) * k) ** 2
    return (1.0 - q) * (g + k) / denom


def lever_effectiveness_ratio(params: ModelParams) -> float:
    """|∂δ*/∂β_α| / |∂δ*/∂β_κ| = (g + κ_eff)/g > 1 (independent of q)."""
    g = gap_from_patience_free(params)
    if g <= 0:
        return float("inf")
    return (g + kappa_eff(params)) / g


def substitution_cross_derivative(params: ModelParams, q: float = 0.0) -> float:
    """Cross-derivative ∂²δ*/∂β_α∂β_κ = (1-q)(D - 2gq)/D³, D = g+(1-q)κ_eff.

    Positive for all q < 1/2 (and for q up to 1 whenever D > 2gq):
    coordination quality and deterrence are substitutes in relaxing the
    patience requirement.
    """
    g = gap_from_patience_free(params)
    if g <= 0:
        return 0.0
    D = g + (1.0 - q) * kappa_eff(params)
    return (1.0 - q) * (D - 2.0 * g * q) / D**3


def is_patience_free(params: ModelParams) -> bool:
    """Check if in patience-free regime: β_α ≥ β_D (robust to any q < 1)."""
    return params.beta_alpha >= params.beta_D


def q_max_tolerable(params: ModelParams, delta: float) -> float:
    """Maximum detection-failure probability compatible with δ ≥ δ*(q).

    Solving δ = δ*(q̄) yields q̄(δ) = 1 - (1-δ)g / (δ κ_eff). Returns 1.0 in
    the patience-free regime (any q tolerated) and 0.0 if even perfect
    monitoring cannot sustain cooperation at this δ.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    g = gap_from_patience_free(params)
    if g <= 0:
        return 1.0
    q_bar = 1.0 - (1.0 - delta) * g / (delta * kappa_eff(params))
    return float(min(1.0, max(0.0, q_bar)))


def phase_transition_jump(params: ModelParams, q: float = 0.0) -> float:
    """Derivative jump magnitude at phase transition: 1/((1-q)κ_eff).

    At β_α = β_D, ∂δ*/∂g has a discontinuous jump of magnitude
    1/((1-q)κ_eff). Imperfect monitoring (q>0) sharpens the transition.
    """
    return 1.0 / ((1.0 - q) * kappa_eff(params))


def critical_width_90_percent(params: ModelParams, q: float = 0.0) -> float:
    """Onset width of the phase transition: Δg = (1-q)κ_eff/9.

    Width of gap g over which δ* rises from 0 to 0.1 past the patience-free
    boundary (δ*(Δg) = 0.1 exactly). Reaching δ* = 0.9 takes g = 9(1-q)κ_eff,
    reflecting the hyperbolic flattening; the sharpness statement concerns
    the onset, not the full 0.1-to-0.9 traversal.
    """
    return (1.0 - q) * kappa_eff(params) / 9.0


def optimal_group_size(f: float, c: float) -> float:
    """Optimal group size from parabolic value function: N* = f/c.

    Derived from ∂V/∂N = 0 where V(N) = (N-1)f - cN²/2.

    Args:
        f: Mobilization fraction per agent
        c: Quadratic cost coefficient

    Returns:
        Optimal group size (real number, typically round to integer)
    """
    return f / c


def group_value_parabolic(N: int, f: float, c: float) -> float:
    """Parabolic group value function: V(N) = (N-1)f - cN²/2.

    Has unique maximum at N* = f/c.
    Becomes negative for N > 2f/c (too costly).

    Args:
        N: Group size
        f: Mobilization fraction per agent
        c: Quadratic cost coefficient

    Returns:
        Net value of coalition
    """
    return (N - 1) * f - c * N**2 / 2.0
