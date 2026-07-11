"""Kalman update (Joseph form) and existence/birth/death rules (plan §2.4)."""
from typing import Tuple

import torch

from ..utils.linalg import add_jitter, symmetrize


def kalman_update(mean: torch.Tensor, cov: torch.Tensor, H: torch.Tensor,
                  z: torch.Tensor, R: torch.Tensor,
                  eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Kalman update in Joseph form (numerically PSD-stable).

    mean: (B, s), cov: (B, s, s), H: (p, s), z: (B, p), R: (B, p, p)
    """
    err = z - mean @ H.T                                     # (B, p)
    S = H @ cov @ H.T + R                                    # (B, p, p)
    chol = torch.linalg.cholesky(add_jitter(symmetrize(S), eps))
    PHt = cov @ H.T                                          # (B, s, p)
    # K = P H^T S^{-1}  via solve on the transposed system
    K = torch.cholesky_solve(PHt.transpose(-1, -2), chol).transpose(-1, -2)
    new_mean = mean + (K @ err.unsqueeze(-1)).squeeze(-1)
    I = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    ImKH = I - K @ H
    new_cov = ImKH @ cov @ ImKH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    return new_mean, symmetrize(new_cov)


def existence_update(alpha: torch.Tensor, matched: torch.Tensor,
                     visibility: torch.Tensor, boost: float, decay: float) -> torch.Tensor:
    """Visibility-gated existence update (the occlusion-persistence rule).

    matched slots      -> alpha rises toward 1.
    unmatched, visible -> "should have seen it": alpha decays.
    unmatched, occluded-> alpha kept (belief covariance keeps inflating instead).
    """
    up = alpha + (1.0 - alpha) * boost
    down = alpha * (1.0 - decay * visibility)
    return torch.where(matched, up, down)


def birth_cov(R_pos: torch.Tensor, pos_dim: int, pos_scale: float,
              vel_std: float) -> torch.Tensor:
    """Initial belief covariance for a newborn slot: inflated observation
    noise on position, weakly-informative prior on velocity."""
    s = 2 * pos_dim
    cov = torch.zeros(s, s, dtype=R_pos.dtype, device=R_pos.device)
    cov[:pos_dim, :pos_dim] = pos_scale * R_pos
    cov[pos_dim:, pos_dim:] = (vel_std ** 2) * torch.eye(pos_dim, dtype=R_pos.dtype, device=R_pos.device)
    return cov
