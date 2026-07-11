"""Training objectives (plan §2.5, with the v3.1 revision-2 split):

* innovation NLL  — predicted slots vs observation proposals (trains R_theta,
  online filtering behaviour);
* forecast NLL    — predicted slots vs GT states (trains F/d_theta/Q_theta);
* chi^2 coverage calibration regulariser — applied to GT-referenced residuals
  only (calibrating against your own proposals would be self-referential);
* existence BCE.

Deliberately NOT here (v2's pathologies): KL between unmatched Gaussian sets,
trace(Sigma) penalties.
"""
import math
from typing import Sequence, Tuple

import torch
from scipy.stats import chi2 as scipy_chi2

from ..utils.linalg import gaussian_nll, mahalanobis_sq


def innovation_nll(d2: torch.Tensor, logdet: torch.Tensor,
                   pairs: Sequence[Tuple[int, int]], pos_dim: int) -> torch.Tensor:
    """Mean NLL over matched (slot, obs) pairs; zero if no pairs."""
    if not pairs:
        return torch.tensor(0.0, device=d2.device)
    si = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=d2.device)
    oj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=d2.device)
    vals = 0.5 * (d2[si, oj] + logdet[si, oj] + pos_dim * math.log(2 * math.pi))
    return vals.mean()


def forecast_nll(prior_mean: torch.Tensor, prior_cov: torch.Tensor,
                 gt_pos: torch.Tensor, pos_dim: int) -> torch.Tensor:
    """NLL of GT positions under the predicted position marginal.

    prior_mean: (B, s), prior_cov: (B, s, s), gt_pos: (B, p).
    """
    if prior_mean.shape[0] == 0:
        return torch.tensor(0.0, device=prior_mean.device)
    err = gt_pos - prior_mean[:, :pos_dim]
    cov = prior_cov[:, :pos_dim, :pos_dim]
    return gaussian_nll(err, cov).mean()


def gt_mahalanobis(prior_mean: torch.Tensor, prior_cov: torch.Tensor,
                   gt_pos: torch.Tensor, pos_dim: int) -> torch.Tensor:
    err = gt_pos - prior_mean[:, :pos_dim]
    return mahalanobis_sq(err, prior_cov[:, :pos_dim, :pos_dim])


def chi2_coverage_penalty(d2_gt: torch.Tensor, pos_dim: int,
                          levels: Sequence[float] = (0.68, 0.95),
                          temperature: float = 0.5) -> torch.Tensor:
    """Differentiable coverage-calibration regulariser.

    If the belief is calibrated, GT-referenced squared Mahalanobis distances
    follow chi^2_p, so the fraction below chi2.ppf(level) should equal
    `level`. A soft (sigmoid) indicator makes the empirical coverage
    differentiable; the penalty is the squared gap at each level.
    """
    if d2_gt.numel() < 8:
        return torch.tensor(0.0, device=d2_gt.device)
    penalty = torch.tensor(0.0, device=d2_gt.device)
    for level in levels:
        thresh = float(scipy_chi2.ppf(level, df=pos_dim))
        soft_cov = torch.sigmoid((thresh - d2_gt) / temperature).mean()
        penalty = penalty + (soft_cov - level) ** 2
    return penalty


def existence_bce(alpha: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """BCE on existence probabilities over `mask`-selected slots."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=alpha.device)
    a = alpha[mask].clamp(1e-5, 1 - 1e-5)
    t = target[mask]
    return -(t * a.log() + (1 - t) * (1 - a).log()).mean()
