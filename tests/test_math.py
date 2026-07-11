"""CPU unit tests for the belief-memory math. Run: python -m pytest tests/ -q"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefgauss.memory.associate import (greedy_match, innovation_stats,
                                          sinkhorn_log)
from beliefgauss.memory.dynamics import cv_transition_matrix, predict
from beliefgauss.memory.memory import BeliefMemory, MemoryConfig
from beliefgauss.memory.update import existence_update, kalman_update
from beliefgauss.sim.synthetic import SimConfig, generate_episode
from beliefgauss.utils.linalg import mahalanobis_sq


def rand_spd(n, dim, scale=1.0):
    A = torch.randn(n, dim, dim)
    return scale * (A @ A.transpose(-1, -2)) + 0.1 * torch.eye(dim)


def test_kalman_matches_closed_form():
    torch.manual_seed(0)
    s, p = 4, 2
    H = torch.zeros(p, s); H[:, :p] = torch.eye(p)
    mean = torch.randn(1, s)
    P = rand_spd(1, s)
    R = rand_spd(1, p, 0.1)
    z = torch.randn(1, p)
    new_mean, new_cov = kalman_update(mean, P, H, z, R)
    S = H @ P[0] @ H.T + R[0]
    K = P[0] @ H.T @ torch.linalg.inv(S)
    ref_mean = mean[0] + K @ (z[0] - H @ mean[0])
    ref_cov = (torch.eye(s) - K @ H) @ P[0]
    assert torch.allclose(new_mean[0], ref_mean, atol=1e-4)
    assert torch.allclose(new_cov[0], 0.5 * (ref_cov + ref_cov.T), atol=1e-3)


def test_kalman_reduces_and_predict_inflates_uncertainty():
    torch.manual_seed(1)
    s, p = 4, 2
    H = torch.zeros(p, s); H[:, :p] = torch.eye(p)
    P = rand_spd(3, s)
    mean = torch.randn(3, s)
    R = rand_spd(3, p, 0.05)
    z = torch.randn(3, p)
    _, post = kalman_update(mean, P, H, z, R)
    assert (torch.diagonal(post, dim1=-2, dim2=-1).sum(-1)
            <= torch.diagonal(P, dim1=-2, dim2=-1).sum(-1) + 1e-5).all()
    F = cv_transition_matrix(p, 0.5)
    Q = 0.01 * torch.eye(s).expand(3, s, s)
    _, prior = predict(mean, post, F, torch.zeros_like(mean), Q)
    evals = torch.linalg.eigvalsh(prior)
    assert (evals > 0).all()  # stays PSD


def test_mahalanobis_coverage_of_true_gaussian():
    torch.manual_seed(2)
    from scipy.stats import chi2
    p = 2
    cov = rand_spd(1, p)[0]
    L = torch.linalg.cholesky(cov)
    samples = (L @ torch.randn(p, 20000)).T
    d2 = mahalanobis_sq(samples, cov.expand(20000, p, p))
    for level in (0.68, 0.95):
        emp = (d2 <= chi2.ppf(level, df=p)).float().mean()
        assert abs(emp - level) < 0.02


def test_sinkhorn_marginals():
    torch.manual_seed(3)
    scores = torch.randn(5, 7)
    log_a = sinkhorn_log(scores, torch.tensor(0.0), iters=100)
    a = log_a.exp()
    assert torch.allclose(a[:5].sum(1), torch.ones(5), atol=1e-3)
    assert torch.allclose(a[:, :7].sum(0), torch.ones(7), atol=1e-3)


def test_greedy_match_respects_gate():
    d2 = torch.tensor([[1.0, 50.0], [50.0, 2.0]])
    ok = torch.tensor([True, True])
    m = greedy_match(d2, gate=10.0, slot_ok=ok, obs_ok=ok)
    assert sorted(m) == [(0, 0), (1, 1)]
    m = greedy_match(d2, gate=0.5, slot_ok=ok, obs_ok=ok)
    assert m == []


def test_existence_occlusion_rule():
    alpha = torch.tensor([0.9, 0.9, 0.9])
    matched = torch.tensor([True, False, False])
    vis = torch.tensor([1.0, 1.0, 0.0])  # slot2 occluded
    out = existence_update(alpha, matched, vis, boost=0.4, decay=0.35)
    assert out[0] > 0.9          # matched: rises
    assert out[1] < 0.9          # unmatched & visible: decays
    assert abs(out[2] - 0.9) < 1e-6  # unmatched & occluded: kept


def test_memory_step_runs_and_births():
    torch.manual_seed(4)
    cfg = MemoryConfig()
    model = BeliefMemory(cfg)
    state = model.init_state()
    z = torch.randn(3, 2) * 5
    quality = torch.rand(3, 3)
    feat = torch.zeros(3, cfg.feat_dim)
    ids = torch.tensor([0, 1, 2])
    vis = lambda pos: torch.ones(pos.shape[0])
    state, aux = model.step(state, z, quality, feat, ids, vis)
    assert state.active.sum() == 3
    for _ in range(4):
        state, aux = model.step(state, z, quality, feat, ids, vis)
    assert torch.isfinite(state.mean).all()
    assert torch.isfinite(state.cov).all()
    assert (torch.linalg.eigvalsh(state.cov[state.active]) > 0).all()


def test_simulator_produces_occlusion_events():
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    total_occ = 0
    for _ in range(10):
        ep = generate_episode(cfg, rng)
        assert ep.gt_pos.shape == (cfg.horizon, ep.cls.shape[0], 2)
        total_occ += int(ep.occluded.sum())
        # visibility oracle agrees with hard occlusion labels most of the time
        vis = ep.visibility_fn()(ep.gt_pos.reshape(-1, 2)).reshape(ep.occluded.shape)
        agree = ((vis < 0.5) == ep.occluded).float().mean()
        assert agree > 0.9
    assert total_occ > 20
