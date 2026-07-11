"""Evaluation metrics: calibration (coverage), accuracy, occlusion behaviour."""
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import chi2 as scipy_chi2


class CalibrationMeter:
    """Accumulates GT-referenced squared Mahalanobis distances and errors."""

    def __init__(self, pos_dim: int):
        self.pos_dim = pos_dim
        self.d2: List[float] = []
        self.sq_err: List[float] = []
        self.nll: List[float] = []

    def add(self, d2: torch.Tensor, sq_err: torch.Tensor, nll: torch.Tensor):
        self.d2 += d2.detach().flatten().tolist()
        self.sq_err += sq_err.detach().flatten().tolist()
        self.nll += nll.detach().flatten().tolist()

    def summary(self) -> Dict[str, float]:
        if not self.d2:
            return {}
        d2 = np.asarray(self.d2)
        out = {
            "n": int(d2.size),
            "rmse": float(np.sqrt(np.mean(self.sq_err))),
            "nll": float(np.mean(self.nll)),
        }
        for level in (0.68, 0.95):
            thresh = scipy_chi2.ppf(level, df=self.pos_dim)
            out[f"coverage@{int(level * 100)}"] = float((d2 <= thresh).mean())
        return out


class OcclusionMeter:
    """Tracks belief-covariance behaviour through occlusion windows.

    For each occlusion event we record trace(P_pos) at entry, at the deepest
    point, and right after re-observation, plus the localisation error at
    reappearance. The signature claim: P grows monotonically while occluded
    and collapses on re-observation.
    """

    def __init__(self):
        self.trace_in: List[float] = []
        self.trace_peak: List[float] = []
        self.trace_out: List[float] = []
        self.reappear_err: List[float] = []
        self.monotonic: List[bool] = []

    def add_event(self, traces: List[float], trace_after: float, err_after: float):
        if len(traces) < 2:
            return
        self.trace_in.append(traces[0])
        self.trace_peak.append(max(traces))
        self.trace_out.append(trace_after)
        self.reappear_err.append(err_after)
        diffs = np.diff(np.asarray(traces))
        self.monotonic.append(bool((diffs >= -1e-6).all()))

    def summary(self) -> Dict[str, float]:
        if not self.trace_in:
            return {}
        return {
            "n_events": len(self.trace_in),
            "trace_growth_ratio": float(np.mean(np.asarray(self.trace_peak) /
                                                np.maximum(np.asarray(self.trace_in), 1e-8))),
            "trace_shrink_ratio": float(np.mean(np.asarray(self.trace_out) /
                                                np.maximum(np.asarray(self.trace_peak), 1e-8))),
            "frac_monotonic_growth": float(np.mean(self.monotonic)),
            "reappear_err_mean": float(np.mean(self.reappear_err)),
        }


def association_accuracy(pred_pairs, slot_track_id, obs_obj_id) -> Dict[str, float]:
    """Fraction of predicted matches whose slot/obs identities agree."""
    if not pred_pairs:
        return {"n": 0}
    correct = sum(int(slot_track_id[i] == obs_obj_id[j]) for i, j in pred_pairs)
    return {"n": len(pred_pairs), "acc": correct / len(pred_pairs)}
