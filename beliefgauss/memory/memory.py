"""BeliefMemory: one predict-associate-update step over instance slots.

This is the minimal, math-first implementation used to validate the Kalman
structure, the learned noise heads, and the NLL/calibration objectives on
simulated observation streams (plan §9, item 5). The full system swaps the
simulated observations for GaussianFormer-2 instance proposals and the
visibility oracle for rendered expected depth.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .associate import greedy_match, innovation_stats
from .dynamics import (ObservationNoiseHead, ProcessNoiseHead,
                       ResidualDynamics, cv_transition_matrix, predict)
from .state import SlotState
from .update import birth_cov, existence_update, kalman_update


@dataclass
class MemoryConfig:
    pos_dim: int = 2
    feat_dim: int = 16
    quality_dim: int = 3
    capacity: int = 32
    dt: float = 0.5
    gate_chi2: float = 13.8          # ~chi2_2 at 0.999
    alpha_boost: float = 0.4
    alpha_decay: float = 0.35
    alpha_birth: float = 0.5
    alpha_kill: float = 0.05
    birth_pos_scale: float = 4.0
    birth_vel_std: float = 2.0
    hidden: int = 64
    learned_dynamics: bool = True    # ablation switch: False = pure CV
    learned_noise: bool = True       # ablation switch: False = fixed Q/R below
    fixed_q_std: float = 0.1
    fixed_r_std: float = 0.3


class BeliefMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        s = 2 * cfg.pos_dim
        self.state_dim = s
        H = torch.zeros(cfg.pos_dim, s)
        H[:, : cfg.pos_dim] = torch.eye(cfg.pos_dim)
        self.register_buffer("H", H)
        self.register_buffer("F", cv_transition_matrix(cfg.pos_dim, cfg.dt))
        self.dyn = ResidualDynamics(s, cfg.feat_dim, cfg.hidden)
        self.q_head = ProcessNoiseHead(s, cfg.feat_dim, cfg.hidden, init_std=cfg.fixed_q_std)
        self.r_head = ObservationNoiseHead(cfg.quality_dim, cfg.pos_dim, init_std=cfg.fixed_r_std)
        self.feat_fuse = nn.GRUCell(cfg.feat_dim, cfg.feat_dim)

    def init_state(self, device: torch.device = torch.device("cpu")) -> SlotState:
        return SlotState.empty(self.cfg.capacity, self.cfg.pos_dim, self.cfg.feat_dim, device=device)

    # ------------------------------------------------------------------
    def observation_noise(self, quality: torch.Tensor) -> torch.Tensor:
        if self.cfg.learned_noise:
            return self.r_head(quality)
        eye = torch.eye(self.cfg.pos_dim, device=quality.device)
        return (self.cfg.fixed_r_std ** 2) * eye.expand(quality.shape[0], -1, -1)

    def process_noise(self, mean: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.cfg.learned_noise:
            return self.q_head(mean, feat)
        eye = torch.eye(self.state_dim, device=mean.device)
        return (self.cfg.fixed_q_std ** 2) * eye.expand(mean.shape[0], -1, -1)

    # ------------------------------------------------------------------
    def step(
        self,
        state: SlotState,
        z: torch.Tensor,                 # (M, p) observation proposals
        quality: torch.Tensor,           # (M, q) observation quality descriptor
        obs_feat: torch.Tensor,          # (M, d)
        obs_obj_id: torch.Tensor,        # (M,) long; simulator object id (eval / teacher forcing)
        visibility_fn: Callable[[torch.Tensor], torch.Tensor],
        teacher_forcing: bool = False,
    ) -> Tuple[SlotState, Dict]:
        cfg, H = self.cfg, self.H
        K = state.capacity
        M = z.shape[0]

        # ---- 1. Predict --------------------------------------------------
        residual = self.dyn(state.mean, state.feat) if cfg.learned_dynamics \
            else torch.zeros_like(state.mean)
        Q = self.process_noise(state.mean, state.feat)
        prior_mean, prior_cov = predict(state.mean, state.cov, self.F, residual, Q)

        # ---- 2. Visibility of predicted slot positions -------------------
        vis = visibility_fn(prior_mean @ H.T)            # (K,) in [0, 1]

        # ---- 3. Associate -------------------------------------------------
        matched_slot = torch.zeros(K, dtype=torch.bool, device=z.device)
        matched_obs = torch.zeros(M, dtype=torch.bool, device=z.device)
        pairs: List[Tuple[int, int]] = []
        if M > 0 and state.active.any():
            R = self.observation_noise(quality)
            err, S, d2, logdet = innovation_stats(prior_mean, prior_cov, H, z, R)
            if teacher_forcing:
                for j in range(M):
                    cand = (state.track_id == obs_obj_id[j]) & state.active
                    if cand.any():
                        pairs.append((int(cand.nonzero()[0]), j))
            else:
                pairs = greedy_match(d2, cfg.gate_chi2, state.active,
                                     torch.ones(M, dtype=torch.bool, device=z.device))
        else:
            R = self.observation_noise(quality) if M > 0 \
                else torch.zeros(0, cfg.pos_dim, cfg.pos_dim, device=z.device)
            err = torch.zeros(K, M, cfg.pos_dim, device=z.device)
            d2 = torch.zeros(K, M, device=z.device)
            logdet = torch.zeros(K, M, device=z.device)

        mean, cov = prior_mean.clone(), prior_cov.clone()
        feat = state.feat.clone()
        track_id = state.track_id.clone()
        age = state.last_obs_age + 1.0

        # ---- 4. Update matched slots --------------------------------------
        if pairs:
            si = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=z.device)
            oj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=z.device)
            upd_mean, upd_cov = kalman_update(prior_mean[si], prior_cov[si], H, z[oj], R[oj])
            mean = torch.index_put(mean, (si,), upd_mean)
            cov = torch.index_put(cov, (si,), upd_cov)
            feat = torch.index_put(feat, (si,), self.feat_fuse(obs_feat[oj], state.feat[si]))
            matched_slot[si] = True
            matched_obs[oj] = True
            track_id[si] = obs_obj_id[oj]
            age[si] = 0.0

        # ---- 5. Existence update & death ----------------------------------
        alpha = existence_update(state.alpha, matched_slot, vis,
                                 cfg.alpha_boost, cfg.alpha_decay)
        active = state.active & (alpha > cfg.alpha_kill)

        # ---- 6. Birth from unmatched observations --------------------------
        born_slots: List[int] = []
        if M > 0:
            free = (~active).nonzero().flatten().tolist()
            for j in (~matched_obs).nonzero().flatten().tolist():
                if not free:
                    break
                i = free.pop(0)
                new_mean = torch.cat([z[j], torch.zeros(cfg.pos_dim, device=z.device)])
                mean = torch.index_put(mean, (torch.tensor([i]),), new_mean.unsqueeze(0))
                bc = birth_cov(R[j], cfg.pos_dim, cfg.birth_pos_scale, cfg.birth_vel_std)
                cov = torch.index_put(cov, (torch.tensor([i]),), bc.unsqueeze(0))
                feat = torch.index_put(feat, (torch.tensor([i]),), obs_feat[j].unsqueeze(0))
                alpha = torch.index_put(alpha, (torch.tensor([i]),),
                                        torch.tensor([cfg.alpha_birth], device=z.device))
                active = active.clone()
                active[i] = True
                track_id[i] = obs_obj_id[j]
                age[i] = 0.0
                born_slots.append(i)

        new_state = SlotState(mean=mean, cov=cov, alpha=alpha, feat=feat,
                              active=active, last_obs_age=age, track_id=track_id)
        aux = {
            "prior_mean": prior_mean, "prior_cov": prior_cov,
            "pairs": pairs, "d2": d2, "logdet": logdet, "err": err,
            "visibility": vis, "matched_slot": matched_slot,
            "matched_obs": matched_obs, "born_slots": born_slots,
        }
        return new_state, aux
