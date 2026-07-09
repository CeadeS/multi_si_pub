"""Cost model used for scaling figure."""

from dataclasses import dataclass
import math
from typing import Dict


@dataclass(frozen=True)
class CostParams:
    """Parameter bundle for the scaling cost model."""

    m_bits: float
    rounds: float
    cost_per_bit: float
    consensus_scale: float
    consensus_exp: float
    distance_m: float
    speed_m_s: float
    processing_delay: float
    logical_rounds: float
    latency_budget: float
    latency_penalty: float
    failure_prob: float
    reliability_penalty: float
    hedge_linear: float


@dataclass(frozen=True)
class BenefitParams:
    """Benefit parameter bundle."""

    sat_scale: float
    sat_rate: float
    lin_scale: float
    net_scale: float
    net_rate: float


def default_cost_params() -> CostParams:
    """Default values for Appendix cost model sweeps."""

    return CostParams(
        m_bits=2e5,
        rounds=3.0,
        cost_per_bit=2e-6,
        consensus_scale=0.15,
        consensus_exp=1.5,
        distance_m=1.0e6,
        speed_m_s=3.0e8,
        processing_delay=0.003,
        logical_rounds=3.0,
        latency_budget=0.02,
        latency_penalty=6.0,
        failure_prob=0.01,
        reliability_penalty=8.0,
        hedge_linear=0.35,
    )


def default_benefit_params() -> BenefitParams:
    """Default benefit parameters matching Appendix forms."""

    return BenefitParams(
        sat_scale=14.0,
        sat_rate=0.35,
        lin_scale=3.5,
        net_scale=4.5,
        net_rate=0.6,
    )


def benefit_saturating(n_agents: int, params: BenefitParams) -> float:
    """Saturating benefits: a * (1 - exp(-kN))."""

    return params.sat_scale * (1.0 - math.exp(-params.sat_rate * n_agents))


def benefit_linear(n_agents: int, params: BenefitParams) -> float:
    """Linear benefits: a * N."""

    return params.lin_scale * n_agents


def benefit_network(n_agents: int, params: BenefitParams) -> float:
    """Network-effect benefits: a * N * log(1 + bN)."""

    return params.net_scale * n_agents * math.log(1.0 + params.net_rate * n_agents)


def cost_comm(n_agents: int, params: CostParams) -> float:
    """Quadratic communication cost."""

    channels = n_agents * (n_agents - 1) / 2.0
    return params.cost_per_bit * params.m_bits * params.rounds * channels


def cost_consensus(n_agents: int, params: CostParams) -> float:
    """Consensus overhead cost."""

    return params.consensus_scale * (n_agents ** params.consensus_exp)


def cost_latency(params: CostParams) -> float:
    """Latency penalty cost (constant in N for fixed architecture)."""

    latency = params.distance_m / params.speed_m_s + params.processing_delay * params.logical_rounds
    excess = max(0.0, latency - params.latency_budget)
    return params.latency_penalty * excess


def cost_reliability(n_agents: int, params: CostParams) -> float:
    """Reliability risk cost."""

    p_failure = 1.0 - (1.0 - params.failure_prob) ** n_agents
    return params.reliability_penalty * p_failure


def cost_hedging(n_agents: int, params: CostParams) -> float:
    """Linear hedging cost."""

    return params.hedge_linear * n_agents


def total_cost(n_agents: int, params: CostParams) -> float:
    """Aggregate cost C(N)."""

    return (
        cost_comm(n_agents, params)
        + cost_consensus(n_agents, params)
        + cost_latency(params)
        + cost_reliability(n_agents, params)
        + cost_hedging(n_agents, params)
    )


def net_values(n_agents: int, cost: CostParams, benefit: BenefitParams) -> Dict[str, float]:
    """Return net values for each benefit form."""

    base_cost = total_cost(n_agents, cost)
    return {
        "saturating": benefit_saturating(n_agents, benefit) - base_cost,
        "linear": benefit_linear(n_agents, benefit) - base_cost,
        "network": benefit_network(n_agents, benefit) - base_cost,
    }
