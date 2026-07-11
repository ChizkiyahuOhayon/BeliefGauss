"""Synthetic multi-object observation streams for math-correctness smoke tests
(plan §9 item 5). Mimics the statistical structure of the real problem:

* heteroscedastic process noise  — "pedestrian-like" objects manoeuvre a lot,
  "car-like" objects move fast but smoothly; the class is visible to the
  model only through the observation feature, so Q_theta must learn it;
* heteroscedastic observation noise — grows with distance from the ego at
  the origin; the distance is exposed in the quality descriptor, so R_theta
  must learn it;
* an occlusion sector — an angular shadow (a parked bus, say) behind which
  objects produce no observations; the simulator also provides the soft
  visibility oracle that the full system will obtain by rendering expected
  depth from the Gaussian field;
* sparse clutter — false observations that must be born and then killed.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch


@dataclass
class SimConfig:
    pos_dim: int = 2
    dt: float = 0.5
    horizon: int = 32
    n_objects_range: tuple = (4, 8)
    arena: float = 20.0              # world is [-arena, arena]^2
    # class 0 = pedestrian-like, class 1 = car-like
    speed_range: tuple = ((0.5, 1.5), (3.0, 6.0))
    accel_std: tuple = (0.5, 0.1)    # true per-class process noise (on velocity)
    obs_sigma0: float = 0.2          # observation noise at the ego
    obs_dist_scale: float = 20.0     # sigma = sigma0 * (1 + dist / scale)
    occluder_halfwidth: tuple = (0.25, 0.5)   # radians
    occluder_min_range: tuple = (3.0, 8.0)
    clutter_rate: float = 0.15       # expected false observations per step
    feat_dim: int = 16
    quality_dim: int = 3             # [dist_norm, class_onehot(2)]


@dataclass
class Episode:
    cfg: SimConfig
    gt_pos: torch.Tensor        # (T, N, 2)
    gt_vel: torch.Tensor        # (T, N, 2)
    cls: torch.Tensor           # (N,) long
    occluded: torch.Tensor      # (T, N) bool — GT visibility of each object
    obs: List[Dict]             # per step: z (M,2), quality (M,3), feat (M,d), obj_id (M,)
    occ_center: float           # occluder sector centre angle
    occ_halfwidth: float
    occ_min_range: float

    def visibility_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Soft visibility oracle o(pos) in [0,1]; 0 = deep inside the shadow.

        Differentiable in pos, mirroring the rendered-visibility mechanism of
        the full system.
        """
        c, w, r0 = self.occ_center, self.occ_halfwidth, self.occ_min_range

        def fn(pos: torch.Tensor) -> torch.Tensor:
            # atan2/norm have NaN gradients at the origin (where empty slots
            # sit); route those entries through a safe dummy input instead.
            degenerate = pos.norm(dim=-1) < 1e-6
            safe_pos = torch.where(degenerate.unsqueeze(-1), torch.ones_like(pos), pos)
            ang = torch.atan2(safe_pos[..., 1], safe_pos[..., 0])
            dang = torch.atan2(torch.sin(ang - c), torch.cos(ang - c)).abs()
            rng = safe_pos.norm(dim=-1)
            in_sector = torch.sigmoid((w - dang) / 0.05)
            behind = torch.sigmoid((rng - r0) / 0.5)
            vis = 1.0 - in_sector * behind
            return torch.where(degenerate, torch.ones_like(vis), vis)

        return fn


def _hard_occluded(pos: np.ndarray, c: float, w: float, r0: float) -> np.ndarray:
    ang = np.arctan2(pos[..., 1], pos[..., 0])
    dang = np.abs(np.arctan2(np.sin(ang - c), np.cos(ang - c)))
    rng = np.linalg.norm(pos, axis=-1)
    return (dang < w) & (rng > r0)


def generate_episode(cfg: SimConfig, rng: np.random.Generator) -> Episode:
    n = int(rng.integers(cfg.n_objects_range[0], cfg.n_objects_range[1] + 1))
    cls = rng.integers(0, 2, size=n)
    T, dt, A = cfg.horizon, cfg.dt, cfg.arena

    pos = rng.uniform(-A, A, size=(n, 2))
    heading = rng.uniform(-math.pi, math.pi, size=n)
    speed = np.array([rng.uniform(*cfg.speed_range[c]) for c in cls])
    vel = np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=-1)

    occ_center = float(rng.uniform(-math.pi, math.pi))
    occ_halfwidth = float(rng.uniform(*cfg.occluder_halfwidth))
    occ_min_range = float(rng.uniform(*cfg.occluder_min_range))

    gt_pos = np.zeros((T, n, 2))
    gt_vel = np.zeros((T, n, 2))
    occluded = np.zeros((T, n), dtype=bool)
    obs: List[Dict] = []
    clutter_id = -1000

    for t in range(T):
        gt_pos[t], gt_vel[t] = pos, vel
        occ = _hard_occluded(pos, occ_center, occ_halfwidth, occ_min_range)
        occluded[t] = occ

        z_list, q_list, f_list, id_list = [], [], [], []
        for i in range(n):
            if occ[i]:
                continue
            dist = float(np.linalg.norm(pos[i]))
            sigma = cfg.obs_sigma0 * (1.0 + dist / cfg.obs_dist_scale)
            z = pos[i] + rng.normal(0.0, sigma, size=2)
            onehot = np.eye(2)[cls[i]]
            z_list.append(z)
            q_list.append(np.concatenate([[dist / (2 * A)], onehot]))
            feat = np.zeros(cfg.feat_dim)
            feat[:2] = onehot
            f_list.append(feat)
            id_list.append(i)
        # clutter
        for _ in range(rng.poisson(cfg.clutter_rate)):
            z = rng.uniform(-A, A, size=2)
            dist = float(np.linalg.norm(z))
            z_list.append(z)
            q_list.append(np.concatenate([[dist / (2 * A)], [0.5, 0.5]]))
            f_list.append(np.zeros(cfg.feat_dim))
            id_list.append(clutter_id)
            clutter_id -= 1

        m = len(z_list)
        obs.append({
            "z": torch.tensor(np.asarray(z_list).reshape(m, 2), dtype=torch.float32),
            "quality": torch.tensor(np.asarray(q_list).reshape(m, cfg.quality_dim), dtype=torch.float32),
            "feat": torch.tensor(np.asarray(f_list).reshape(m, cfg.feat_dim), dtype=torch.float32),
            "obj_id": torch.tensor(np.asarray(id_list, dtype=np.int64).reshape(m)),
        })

        # true dynamics: CV + per-class acceleration noise + wall reflection
        for i in range(n):
            vel[i] += rng.normal(0.0, cfg.accel_std[cls[i]] * dt, size=2)
        pos = pos + vel * dt
        for d in range(2):
            over = np.abs(pos[:, d]) > A
            vel[over, d] *= -1.0
            pos[over, d] = np.clip(pos[over, d], -A, A)

    return Episode(
        cfg=cfg,
        gt_pos=torch.tensor(gt_pos, dtype=torch.float32),
        gt_vel=torch.tensor(gt_vel, dtype=torch.float32),
        cls=torch.tensor(cls, dtype=torch.long),
        occluded=torch.tensor(occluded),
        obs=obs,
        occ_center=occ_center,
        occ_halfwidth=occ_halfwidth,
        occ_min_range=occ_min_range,
    )
