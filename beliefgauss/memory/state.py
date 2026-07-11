"""Belief state container: a fixed-capacity array of instance slots.

Each slot carries the epistemic belief over one dynamic instance:
mean (position+velocity), belief covariance P, existence probability alpha,
a feature vector, and bookkeeping (age since last observation, a persistent
slot id used only for evaluation).

Design notes
------------
* Fixed capacity + `active` mask instead of dynamic lists: keeps every step
  a pure tensor op and makes truncated BPTT trivial (`detach()`).
* This is the "instance-slot layer" of the two-tier design (plan §0.5,
  revision 1). The primitive/shape layer lives in the encoder and is not
  represented here.
"""
from dataclasses import dataclass

import torch


@dataclass
class SlotState:
    mean: torch.Tensor          # (K, s)  s = 2 * pos_dim, [pos, vel]
    cov: torch.Tensor           # (K, s, s) belief covariance P
    alpha: torch.Tensor         # (K,) existence probability in [0, 1]
    feat: torch.Tensor          # (K, d) slot feature
    active: torch.Tensor        # (K,) bool
    last_obs_age: torch.Tensor  # (K,) steps since last matched observation
    track_id: torch.Tensor      # (K,) long; simulator object id for eval, -1 = clutter/unknown

    @property
    def capacity(self) -> int:
        return self.mean.shape[0]

    def detach(self) -> "SlotState":
        return SlotState(
            mean=self.mean.detach(),
            cov=self.cov.detach(),
            alpha=self.alpha.detach(),
            feat=self.feat.detach(),
            active=self.active.clone(),
            last_obs_age=self.last_obs_age.clone(),
            track_id=self.track_id.clone(),
        )

    @staticmethod
    def empty(capacity: int, pos_dim: int, feat_dim: int,
              dtype: torch.dtype = torch.float32,
              device: torch.device = torch.device("cpu")) -> "SlotState":
        s = 2 * pos_dim
        return SlotState(
            mean=torch.zeros(capacity, s, dtype=dtype, device=device),
            cov=torch.eye(s, dtype=dtype, device=device).expand(capacity, s, s).clone(),
            alpha=torch.zeros(capacity, dtype=dtype, device=device),
            feat=torch.zeros(capacity, feat_dim, dtype=dtype, device=device),
            active=torch.zeros(capacity, dtype=torch.bool, device=device),
            last_obs_age=torch.zeros(capacity, dtype=dtype, device=device),
            track_id=torch.full((capacity,), -1, dtype=torch.long, device=device),
        )
