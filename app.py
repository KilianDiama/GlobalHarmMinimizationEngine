from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Final, Dict, Any
from numba import njit, prange

# ============================================================
# 0. CONFIG — Stable, scale-aware, evolution + CEM hybrid
# ============================================================

@dataclass(frozen=True)
class WorldConfig:
    n_regions: int = 64
    resource_dim: int = 4
    policy_samples: int = 1024
    seed: int = 42

    max_alloc_factor: float = 1.0

    # Cost weights (interpretable, scale-aware)
    lack_weight: float = 1.0
    inequality_weight: float = 0.7
    fairness_weight: float = 0.3
    coupling_weight: float = 0.4

    risk_nonlin_beta: float = 2.0

    eps: Final[float] = 1e-6

    # Optimizer / CEM
    n_iterations: int = 24
    top_fraction: float = 0.12
    elite_fraction: float = 0.06
    fresh_fraction: float = 0.20

    perturb_scale_init: float = 0.18
    perturb_scale_min: float = 0.04
    ema_prior_eta: float = 0.20
    top_k_final: int = 16

    # CEM / variance control
    min_cost_std: float = 1e-3
    anneal_floor: float = 0.25
    collapse_factor: float = 0.12


# ============================================================
# 1. STATE — Consistent float32 layout
# ============================================================

@dataclass
class RegionState:
    resources: np.ndarray  # (d,)
    needs: np.ndarray      # (d,)
    risk: float

@dataclass
class WorldState:
    regions: List[RegionState]

    def to_arrays(self, cfg: WorldConfig):
        n, d = cfg.n_regions, cfg.resource_dim
        R = np.zeros((n, d), dtype=np.float32)
        N = np.zeros((n, d), dtype=np.float32)
        K = np.zeros(n, dtype=np.float32)

        for i, r in enumerate(self.regions):
            R[i] = r.resources.astype(np.float32)
            N[i] = r.needs.astype(np.float32)
            K[i] = np.float32(r.risk)

        return R, N, K


# ============================================================
# 2. COST MODEL — Scale-aware, globally normalized, robust
# ============================================================

@njit(cache=True)
def _safe_mean(x, eps):
    return x.mean() if x.size > 0 else 0.0

@njit(cache=True)
def _safe_var(x, eps):
    if x.size == 0:
        return 0.0
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return v

@njit(cache=True)
def _robust_norm_global(term, eps):
    m = _safe_mean(term, eps)
    v = _safe_var(term, eps)
    return (term - m) / np.sqrt(v + eps)


@njit(cache=True)
def compute_satisfaction_batch(resources, needs, allocations, eps):
    """
    resources:   (n_regions, d)
    needs:       (n_regions, d)
    allocations: (n_samples, n_regions, d)
    returns: sat: (n_samples, n_regions, d)
    """
    needs_safe = np.maximum(needs, eps)
    total = resources[None, :, :] + allocations
    sat = total / needs_safe[None, :, :]
    sat = np.minimum(np.maximum(sat, 0.0), 1.0)
    return sat


@njit(parallel=True, cache=True)
def compute_cost_batch_vectorized(resources, needs, risks, allocations,
                                  lack_w, ineq_w, fair_w, coup_w,
                                  risk_beta, eps):
    """
    resources:   (n_regions, d)
    needs:       (n_regions, d)
    risks:       (n_regions,)
    allocations: (n_samples, n_regions, d)
    returns: costs: (n_samples,)
    """
    n_samples = allocations.shape[0]
    n_regions = resources.shape[0]
    d = resources.shape[1]

    costs_raw = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        alloc_i = allocations[i:i+1]  # (1, n_regions, d)
        sat_i = compute_satisfaction_batch(resources, needs, alloc_i, eps)[0]  # (n_regions, d)

        # Lack per region, per dim
        lack = np.maximum(1.0 - sat_i, 0.0)  # (n_regions, d)
        lack_nl = lack ** risk_beta

        # Risk factor per region
        risk_factor = (1.0 + risks)[:, None]  # (n_regions, 1)

        # Base harm: sum over regions and dims
        base = (lack_nl * risk_factor).sum()

        # Inequality: dispersion of mean satisfaction per region
        sat_region_mean = sat_i.mean(axis=1)  # (n_regions,)
        mean_sat = sat_region_mean.mean()
        dev = np.abs(sat_region_mean - mean_sat) * (1.0 + risks)
        ineq = dev.mean()

        # Fairness: variance of satisfaction across all region-dim pairs
        sat_flat = sat_i.reshape(n_regions * d)
        fair = _safe_var(sat_flat, eps)

        # Coupling: interaction term
        coupling = ineq * fair

        total = (
            lack_w * base
            + ineq_w * ineq
            + fair_w * fair
            + coup_w * coupling
        )
        costs_raw[i] = np.float32(total)

    # Global normalization for stability and comparability
    costs = _robust_norm_global(costs_raw, eps).astype(np.float32)
    return costs


@njit(cache=True)
def _top_k_indices(costs, k):
    idx = np.argsort(costs)
    return idx[:k]


# ============================================================
# 3. POLICY GENERATOR — Gamma-Dirichlet with risk-aware prior
# ============================================================

class PolicyGenerator:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.lambda_scale: float = 48.0
        self.alpha0: float = 1.2

    def _build_alpha(self, prior: np.ndarray | None,
                     risks: np.ndarray,
                     gaps: np.ndarray) -> np.ndarray:
        """
        prior: (n_regions, d) or None
        risks: (n_regions,)
        gaps:  (n_regions, d)
        returns alpha: (n_regions, d)
        """
        cfg = self.cfg
        n_reg, d = cfg.n_regions, cfg.resource_dim

        if prior is None:
            base = np.ones((n_reg, d), dtype=np.float32)
        else:
            base = np.maximum(prior.astype(np.float32), cfg.eps)

        # Risk-aware and gap-aware shaping:
        # - higher risk -> higher alpha (more attention)
        # - larger gap -> higher alpha (more need)
        risk_term = (1.0 + risks)[:, None].astype(np.float32)
        gap_term = np.maximum(gaps, cfg.eps)

        scaled = base * risk_term * np.log1p(gap_term)
        alpha = self.lambda_scale * scaled + self.alpha0
        return np.maximum(alpha, cfg.eps)

    def _sample_dirichlet_gamma(self, alpha: np.ndarray, n_samples: int) -> np.ndarray:
        """
        alpha: (n_regions, d)
        returns: weights: (n_samples, n_regions, d)
        """
        cfg = self.cfg
        n_reg, d = cfg.n_regions, cfg.resource_dim
        weights = np.empty((n_samples, n_reg, d), dtype=np.float32)

        # Correlated across dims via shared gamma draws per region
        for j in range(d):
            a = alpha[:, j].astype(np.float32)
            g = self.rng.gamma(a[None, :], 1.0, size=(n_samples, n_reg)).astype(np.float32)
            g_sum = g.sum(axis=1, keepdims=True)
            weights[:, :, j] = g / np.maximum(g_sum, cfg.eps)

        return weights

    def generate(self, resources, needs, risks, n_samples, prior=None):
        cfg = self.cfg

        gaps = np.maximum(needs - resources, 0.0).astype(np.float32)
        caps = resources.sum(axis=0).astype(np.float32) * cfg.max_alloc_factor

        alpha = self._build_alpha(prior, risks, gaps)
        weights = self._sample_dirichlet_gamma(alpha, n_samples)

        alloc = weights * caps[None, :, :]
        alloc = np.minimum(alloc, gaps[None, :, :])

        return weights, alloc, caps, gaps


# ============================================================
# 4. OPTIMIZER — CEM-style, anti-collapse, variance-aware
# ============================================================

@njit(cache=True)
def _apply_perturbation(best_mean, cov_scale, n_samples, n_reg, d, eps):
    """
    best_mean: (n_regions, d)
    cov_scale: scalar controlling noise magnitude
    returns: new_w: (n_samples, n_regions, d)
    """
    noise = np.random.normal(0.0, cov_scale, (n_samples, n_reg, d)).astype(np.float32)
    base = np.maximum(best_mean[None, :, :], eps)
    new_w = base * np.exp(noise).astype(np.float32)
    new_w = np.maximum(new_w, eps)

    new_w_sum = new_w.sum(axis=1, keepdims=True)
    new_w = new_w / np.maximum(new_w_sum, eps)
    return new_w


class PolicyOptimizer:
    def __init__(self, cfg: WorldConfig, cost_model, policy_gen: PolicyGenerator):
        self.cfg = cfg
        self.cost_model = cost_model
        self.policy_gen = policy_gen
        self.rng = np.random.default_rng(cfg.seed + 1)

        self._alloc_buffer = np.empty(
            (cfg.policy_samples, cfg.n_regions, cfg.resource_dim),
            dtype=np.float32,
        )
        self._weights_buffer = np.empty_like(self._alloc_buffer)

    def _compute_costs(self, R, N, K, allocations):
        cfg = self.cfg
        return self.cost_model(
            R, N, K, allocations,
            cfg.lack_weight,
            cfg.inequality_weight,
            cfg.fairness_weight,
            cfg.coupling_weight,
            cfg.risk_nonlin_beta,
            cfg.eps,
        )

    def optimize(self, R, N, K, n_samples):
        cfg = self.cfg

        assert R.shape == (cfg.n_regions, cfg.resource_dim)
        assert N.shape == (cfg.n_regions, cfg.resource_dim)
        assert K.shape == (cfg.n_regions,)

        # Initial population
        weights, alloc, caps, gaps = self.policy_gen.generate(R, N, K, n_samples)
        costs = self._compute_costs(R, N, K, alloc)

        prior = weights.mean(axis=0).astype(np.float32)
        initial_std = float(max(costs.std(), cfg.min_cost_std))

        perturb_scale = cfg.perturb_scale_init

        for _ in range(cfg.n_iterations):
            # Top-k selection
            k = max(1, int(cfg.top_fraction * n_samples))
            best_idx = _top_k_indices(costs, k)
            best_w = weights[best_idx]
            best_alloc = alloc[best_idx]
            best_costs = costs[best_idx]

            # Elite subset
            elite_k = max(1, int(cfg.elite_fraction * n_samples))
            elite_idx = best_idx[:elite_k]
            elite_w = weights[elite_idx]
            elite_alloc = alloc[elite_idx]
            elite_costs = costs[elite_idx]

            # Update prior via EMA
            best_mean = best_w.mean(axis=0)
            prior = (1.0 - cfg.ema_prior_eta) * prior + cfg.ema_prior_eta * best_mean

            # Cost std for annealing
            current_std = float(max(costs.std(), cfg.min_cost_std))
            temperature = current_std / initial_std
            temperature = max(cfg.anneal_floor, temperature)

            # Anti-collapse: if std too low, re-expand
            if current_std < cfg.collapse_factor * initial_std:
                temperature = max(temperature, cfg.anneal_floor * 2.0)

            # Perturbation scale
            cov_scale = max(cfg.perturb_scale_min, perturb_scale * temperature)

            # Local CEM-style perturbation around best_mean
            new_w = _apply_perturbation(
                best_mean.astype(np.float32),
                cov_scale,
                n_samples,
                cfg.n_regions,
                cfg.resource_dim,
                cfg.eps,
            )

            new_alloc = new_w * caps[None, :, :]
            new_alloc = np.minimum(new_alloc, gaps[None, :, :])

            # Fresh samples with updated prior (exploration)
            fresh_n = max(1, int(cfg.fresh_fraction * n_samples))
            fresh_w, fresh_alloc, _, _ = self.policy_gen.generate(
                R, N, K, fresh_n, prior
            )

            # Combine elite + local perturbation + fresh
            combined_w = np.concatenate([elite_w, new_w, fresh_w], axis=0)
            combined_alloc = np.concatenate([elite_alloc, new_alloc, fresh_alloc], axis=0)

            combined_costs = self._compute_costs(R, N, K, combined_alloc)

            # Keep best n_samples
            keep_idx = _top_k_indices(combined_costs, n_samples)
            weights = combined_w[keep_idx]
            alloc = combined_alloc[keep_idx]
            costs = combined_costs[keep_idx]

        final_k = min(cfg.top_k_final, len(costs))
        idx = _top_k_indices(costs, final_k)
        return alloc[idx], costs[idx]


# ============================================================
# 5. ENGINE — Global harm minimization core
# ============================================================

class GlobalHarmMinimizationEngine:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.policy_gen = PolicyGenerator(cfg)
        self.optimizer = PolicyOptimizer(cfg, compute_cost_batch_vectorized, self.policy_gen)

    def evaluate(self, world: WorldState,
                 n_samples: int | None = None,
                 top_k: int | None = None) -> Dict[str, Any]:
        cfg = self.cfg
        n_samples = n_samples or cfg.policy_samples
        top_k = top_k or cfg.top_k_final

        R, N, K = world.to_arrays(cfg)
        best_alloc, best_costs = self.optimizer.optimize(R, N, K, n_samples)

        return {
            "best_policies": best_alloc[:top_k],
            "best_costs": best_costs[:top_k],
        }


# ============================================================
# 6. SIMPLE AGENT WRAPPER — For multi-agent / IA integration
# ============================================================

@dataclass
class AllocationAgent:
    name: str
    engine: GlobalHarmMinimizationEngine

    def decide(self, world: WorldState) -> Dict[str, Any]:
        return self.engine.evaluate(world)


# ============================================================
# 7. UTILITY — Random world generator (for testing / demos)
# ============================================================

def generate_random_world(cfg: WorldConfig, seed: int | None = None) -> WorldState:
    rng = np.random.default_rng(seed or cfg.seed + 123)
    regions: List[RegionState] = []

    for _ in range(cfg.n_regions):
        resources = rng.uniform(10.0, 100.0, size=cfg.resource_dim).astype(np.float32)
        needs = resources + rng.uniform(5.0, 80.0, size=cfg.resource_dim).astype(np.float32)
        risk = float(rng.uniform(0.0, 1.5))
        regions.append(RegionState(resources=resources, needs=needs, risk=risk))

    return WorldState(regions=regions)


# ============================================================
# 8. HIGH-LEVEL API — One-shot global allocation
# ============================================================

def run_global_allocation(cfg: WorldConfig | None = None,
                          world: WorldState | None = None) -> Dict[str, Any]:
    cfg = cfg or WorldConfig()
    world = world or generate_random_world(cfg)
    engine = GlobalHarmMinimizationEngine(cfg)
    return engine.evaluate(world)
