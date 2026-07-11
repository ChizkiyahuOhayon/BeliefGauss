"""Association between predicted slots and observation proposals (plan §2.3).

Cost = squared Mahalanobis distance in innovation space, gated at a chi^2
threshold. Differentiable soft assignment via log-domain Sinkhorn with a
dustbin row/column (SuperGlue-style); hard matches for the Kalman update via
greedy mutual-nearest within the gate.
"""
from typing import List, Tuple

import torch

from ..utils.linalg import add_jitter, symmetrize


def innovation_stats(mean: torch.Tensor, cov: torch.Tensor, H: torch.Tensor,
                     z: torch.Tensor, R: torch.Tensor,
                     eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise innovation statistics.

    mean: (K, s), cov: (K, s, s), H: (p, s), z: (M, p), R: (M, p, p)
    Returns:
      err:    (K, M, p)   z_j - H x_i
      S:      (K, M, p, p) H P_i H^T + R_j
      d2:     (K, M)      squared Mahalanobis
      logdet: (K, M)      log|S|
    """
    K, M = mean.shape[0], z.shape[0]
    Hx = mean @ H.T                                   # (K, p)
    err = z.unsqueeze(0) - Hx.unsqueeze(1)            # (K, M, p)
    HPH = H @ cov @ H.T                               # (K, p, p)
    S = HPH.unsqueeze(1) + R.unsqueeze(0)             # (K, M, p, p)
    chol = torch.linalg.cholesky(add_jitter(symmetrize(S), eps))
    sol = torch.cholesky_solve(err.unsqueeze(-1), chol)
    d2 = (err.unsqueeze(-1) * sol).sum(dim=(-1, -2))  # (K, M)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return err, S, d2, logdet


def sinkhorn_log(scores: torch.Tensor, dustbin: torch.Tensor,
                 iters: int = 30) -> torch.Tensor:
    """Log-domain Sinkhorn over an augmented (K+1, M+1) score matrix.

    scores: (K, M) similarity (higher = better match); dustbin: scalar
    learnable score for the unmatched bins. Returns log assignment
    (K+1, M+1) whose exp has rows 0..K-1 and cols 0..M-1 summing to <= 1
    with mass escaping to the dustbin.
    """
    K, M = scores.shape
    b = dustbin.expand(1)
    aug = torch.cat(
        [torch.cat([scores, b.expand(K, 1)], dim=1),
         torch.cat([b.expand(1, M), b.expand(1, 1)], dim=1)], dim=0)
    # marginals: each real row/col carries mass 1; dustbin absorbs the rest
    log_mu = torch.cat([torch.zeros(K), torch.tensor([float(M)]).log().clamp(min=0)])
    log_nu = torch.cat([torch.zeros(M), torch.tensor([float(K)]).log().clamp(min=0)])
    log_mu, log_nu = log_mu.to(scores), log_nu.to(scores)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(aug + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(aug + u.unsqueeze(1), dim=0)
    return aug + u.unsqueeze(1) + v.unsqueeze(0)


def greedy_match(d2: torch.Tensor, gate: float,
                 slot_ok: torch.Tensor, obs_ok: torch.Tensor) -> List[Tuple[int, int]]:
    """Greedy mutual-nearest matching within the chi^2 gate (inference path).

    d2: (K, M); slot_ok: (K,) bool; obs_ok: (M,) bool.
    Returns a list of (slot_idx, obs_idx).
    """
    K, M = d2.shape
    cost = d2.detach().clone()
    cost[~slot_ok] = float("inf")
    cost[:, ~obs_ok] = float("inf")
    cost[cost > gate] = float("inf")
    matches: List[Tuple[int, int]] = []
    while torch.isfinite(cost).any():
        idx = int(torch.argmin(cost))
        i, j = idx // M, idx % M
        matches.append((i, j))
        cost[i, :] = float("inf")
        cost[:, j] = float("inf")
    return matches
