"""Predict step: constant-velocity transition + learned residual dynamics
and learned heteroscedastic process noise Q_theta (plan §2.2).

State layout: x = [pos (p), vel (p)], s = 2p.
"""
from typing import Tuple

import torch
import torch.nn as nn


def cv_transition_matrix(pos_dim: int, dt: float,
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """F = [[I, dt*I], [0, I]], shape (s, s)."""
    s = 2 * pos_dim
    F = torch.eye(s, dtype=dtype, device=device)
    F[:pos_dim, pos_dim:] = dt * torch.eye(pos_dim, dtype=dtype, device=device)
    return F


class ResidualDynamics(nn.Module):
    """d_theta: small MLP on [state, feat] -> residual state delta.

    Zero-initialized output layer so training starts at the pure CV prior.
    In the full system this becomes a slot-interaction transformer; the MLP
    keeps the smoke test honest about the Kalman structure itself.
    """

    def __init__(self, state_dim: int, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mean: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([mean, feat], dim=-1))


class ProcessNoiseHead(nn.Module):
    """Q_theta: MLP on [state, feat] -> diagonal SPD process noise (K, s, s).

    softplus + floor guarantees positive-definiteness; the floor prevents the
    filter from claiming certainty it cannot have.
    """

    def __init__(self, state_dim: int, feat_dim: int, hidden: int = 64,
                 floor: float = 1e-4, init_std: float = 0.1):
        super().__init__()
        self.floor = floor
        self.net = nn.Sequential(
            nn.Linear(state_dim + feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        # bias so that softplus(bias) ~= init_std^2 at start
        init_var = torch.tensor(init_std ** 2)
        nn.init.constant_(self.net[-1].bias, float(torch.log(torch.expm1(init_var))))

    def forward(self, mean: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        diag = nn.functional.softplus(self.net(torch.cat([mean, feat], dim=-1))) + self.floor
        return torch.diag_embed(diag)


class ObservationNoiseHead(nn.Module):
    """R_theta: MLP on the observation quality descriptor -> diagonal R (M, p, p).

    In the full system the quality descriptor comes from image evidence
    (distance, truncation, lighting tokens); in the smoke test the simulator
    provides it.
    """

    def __init__(self, quality_dim: int, pos_dim: int, hidden: int = 32,
                 floor: float = 1e-4, init_std: float = 0.3):
        super().__init__()
        self.floor = floor
        self.net = nn.Sequential(
            nn.Linear(quality_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, pos_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        init_var = torch.tensor(init_std ** 2)
        nn.init.constant_(self.net[-1].bias, float(torch.log(torch.expm1(init_var))))

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        diag = nn.functional.softplus(self.net(quality)) + self.floor
        return torch.diag_embed(diag)


def predict(mean: torch.Tensor, cov: torch.Tensor, F: torch.Tensor,
            residual: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x' = F x + d_theta,  P' = F P F^T + Q."""
    new_mean = mean @ F.T + residual
    new_cov = F @ cov @ F.T + Q
    return new_mean, new_cov
