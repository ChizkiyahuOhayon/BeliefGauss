"""Gate-0 smoke test: validate the belief-memory math on simulated streams.

Trains BeliefMemory (learned dynamics + heteroscedastic noise heads) on
synthetic multi-object episodes with occlusion, then evaluates against a
classic fixed-noise Kalman filter baseline (plan ablation 3, previewed).

Pass criteria (printed and written to report.json):
  1. NLL (ours) < NLL (fixed-noise KF baseline)
  2. coverage@68 in [0.58, 0.78] and coverage@95 in [0.88, 0.995]
  3. belief covariance grows monotonically through >=80% of occlusion
     windows and shrinks on re-observation (shrink ratio < 0.7)
  4. association accuracy >= 0.95

Usage:
  python scripts/run_smoke_synthetic.py --config configs/smoke_synthetic.yaml \
      --out outputs/smoke_v0
"""
import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beliefgauss.memory.losses import (chi2_coverage_penalty, existence_bce,
                                       forecast_nll, innovation_nll)
from beliefgauss.memory.memory import BeliefMemory, MemoryConfig
from beliefgauss.memory.metrics import (CalibrationMeter, OcclusionMeter,
                                        association_accuracy)
from beliefgauss.sim.synthetic import Episode, SimConfig, generate_episode
from beliefgauss.utils.linalg import gaussian_nll, mahalanobis_sq


def rollout_losses(model: BeliefMemory, ep: Episode, teacher_forcing: bool,
                   bptt_window: int, weights: dict) -> torch.Tensor:
    """Run one episode and return the scalar training loss."""
    p = model.cfg.pos_dim
    state = model.init_state()
    vis_fn = ep.visibility_fn()
    innov_terms, fore_terms, exist_terms, d2_gt_all = [], [], [], []

    for t in range(ep.gt_pos.shape[0]):
        pre_ids, pre_active = state.track_id.clone(), state.active.clone()
        o = ep.obs[t]
        state, aux = model.step(state, o["z"], o["quality"], o["feat"],
                                o["obj_id"], vis_fn, teacher_forcing=teacher_forcing)
        innov_terms.append(innovation_nll(aux["d2"], aux["logdet"], aux["pairs"], p))

        valid = pre_active & (pre_ids >= 0)
        if valid.any():
            gt = ep.gt_pos[t][pre_ids[valid]]
            pm, pc = aux["prior_mean"][valid], aux["prior_cov"][valid]
            fore_terms.append(forecast_nll(pm, pc, gt, p))
            err = gt - pm[:, :p]
            d2_gt_all.append(mahalanobis_sq(err, pc[:, :p, :p]))

        target = (state.track_id >= 0).float()
        exist_terms.append(existence_bce(state.alpha, target, state.active))

        if bptt_window > 0 and (t + 1) % bptt_window == 0:
            state = state.detach()

    loss = weights["innov"] * torch.stack(innov_terms).mean()
    if fore_terms:
        loss = loss + weights["forecast"] * torch.stack(fore_terms).mean()
    if d2_gt_all:
        loss = loss + weights["calib"] * chi2_coverage_penalty(torch.cat(d2_gt_all), p)
    loss = loss + weights["exist"] * torch.stack(exist_terms).mean()
    return loss


@torch.no_grad()
def evaluate(model: BeliefMemory, episodes) -> dict:
    p = model.cfg.pos_dim
    calib = CalibrationMeter(p)
    occm = OcclusionMeter()
    assoc_n, assoc_correct, events_total, events_lost = 0, 0, 0, 0

    for ep in episodes:
        state = model.init_state()
        vis_fn = ep.visibility_fn()
        T = ep.gt_pos.shape[0]
        rec = []  # per step: (pre_ids, pre_active, prior_cov, post_state, matched)
        for t in range(T):
            pre_ids, pre_active = state.track_id.clone(), state.active.clone()
            o = ep.obs[t]
            state, aux = model.step(state, o["z"], o["quality"], o["feat"],
                                    o["obj_id"], vis_fn, teacher_forcing=False)
            valid = pre_active & (pre_ids >= 0)
            if valid.any():
                gt = ep.gt_pos[t][pre_ids[valid]]
                pm, pc = aux["prior_mean"][valid], aux["prior_cov"][valid]
                err = gt - pm[:, :p]
                calib.add(mahalanobis_sq(err, pc[:, :p, :p]),
                          (err ** 2).sum(-1),
                          gaussian_nll(err, pc[:, :p, :p]))
            for i, j in aux["pairs"]:
                if pre_ids[i] >= 0:
                    assoc_n += 1
                    assoc_correct += int(pre_ids[i] == o["obj_id"][j])
            rec.append((pre_ids, pre_active, aux["prior_cov"].clone(),
                        state.mean.clone(), state.cov.clone(),
                        aux["matched_slot"].clone(), state.track_id.clone()))

        # occlusion events: contiguous occluded runs of length >= 3
        occ = ep.occluded.numpy()
        for n in range(occ.shape[1]):
            t = 0
            while t < T:
                if occ[t, n]:
                    t0 = t
                    while t < T and occ[t, n]:
                        t += 1
                    t1 = t  # first visible step after the run
                    if t1 - t0 >= 3 and t1 < T:
                        events_total += 1
                        traces, ok = [], True
                        for tt in range(t0, t1):
                            pre_ids, pre_active, prior_cov = rec[tt][0], rec[tt][1], rec[tt][2]
                            k = ((pre_ids == n) & pre_active).nonzero()
                            if len(k) == 0:
                                ok = False
                                break
                            k = int(k[0])
                            traces.append(float(torch.diagonal(prior_cov[k, :p, :p]).sum()))
                        if not ok:
                            events_lost += 1
                        else:
                            post_ids, matched = rec[t1][6], rec[t1][5]
                            k = ((post_ids == n) & matched).nonzero()
                            if len(k) == 0:
                                events_lost += 1
                            else:
                                k = int(k[0])
                                tr_after = float(torch.diagonal(rec[t1][4][k, :p, :p]).sum())
                                err_after = float((rec[t1][3][k, :p] - ep.gt_pos[t1, n]).norm())
                                occm.add_event(traces, tr_after, err_after)
                else:
                    t += 1

    out = {"calibration": calib.summary(), "occlusion": occm.summary()}
    out["association"] = {"n": assoc_n,
                          "acc": assoc_correct / max(assoc_n, 1)}
    out["occlusion"]["events_total"] = events_total
    out["occlusion"]["persistence_rate"] = 1.0 - events_lost / max(events_total, 1)
    return out


def gate0_verdict(ours: dict, baseline: dict) -> dict:
    c, o = ours["calibration"], ours["occlusion"]
    checks = {
        "nll_beats_fixed_kf": c.get("nll", 1e9) < baseline["calibration"].get("nll", 1e9),
        "coverage68_in_range": 0.58 <= c.get("coverage@68", 0) <= 0.78,
        "coverage95_in_range": 0.88 <= c.get("coverage@95", 0) <= 0.995,
        "occlusion_P_grows": o.get("frac_monotonic_growth", 0) >= 0.8,
        "reobs_P_shrinks": o.get("trace_shrink_ratio", 1e9) < 0.7,
        "association_acc": ours["association"]["acc"] >= 0.95,
    }
    checks["PASS"] = all(checks.values())
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_synthetic.yaml")
    ap.add_argument("--out", default="outputs/smoke_v0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    sim_cfg = SimConfig(**cfg["sim"])
    mem_cfg = MemoryConfig(**cfg["memory"])
    model = BeliefMemory(mem_cfg)
    base_cfg = dataclasses.replace(mem_cfg, learned_dynamics=False, learned_noise=False)
    baseline = BeliefMemory(base_cfg)

    tr = cfg["train"]
    train_eps = [generate_episode(sim_cfg, rng) for _ in range(tr["train_episodes"])]
    eval_eps = [generate_episode(sim_cfg, rng) for _ in range(tr["eval_episodes"])]

    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    weights = tr["loss_weights"]
    losses = []
    t_start = time.time()
    for it in range(tr["iterations"]):
        tf = it < tr["teacher_forcing_iters"]
        batch = [train_eps[i] for i in
                 rng.integers(0, len(train_eps), size=tr["batch_episodes"])]
        loss = torch.stack([
            rollout_losses(model, ep, tf, tr["bptt_window"], weights) for ep in batch
        ]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
        if it % 20 == 0 or it == tr["iterations"] - 1:
            print(f"iter {it:4d}  loss {float(loss):8.4f}  "
                  f"tf={tf}  elapsed {time.time() - t_start:6.1f}s", flush=True)

    print("\n== eval: ours ==", flush=True)
    model.eval()
    ours = evaluate(model, eval_eps)
    print(json.dumps(ours, indent=2))
    print("\n== eval: fixed-noise KF baseline ==", flush=True)
    base = evaluate(baseline, eval_eps)
    print(json.dumps(base, indent=2))

    verdict = gate0_verdict(ours, base)
    print("\n== Gate-0 verdict ==")
    print(json.dumps(verdict, indent=2))

    # ---- report bundle -------------------------------------------------
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1]).decode().strip()
    except Exception:
        commit = "n/a"
    report = {
        "config": cfg, "commit": commit, "seed": seed,
        "train_loss_first": losses[0], "train_loss_last": losses[-1],
        "ours": ours, "fixed_kf_baseline": base, "gate0": verdict,
        "wallclock_sec": time.time() - t_start,
        "torch": torch.__version__,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(losses)
        axes[0].set_title("training loss")
        axes[0].set_xlabel("iteration")
        labels = ["coverage@68", "coverage@95"]
        x = np.arange(2)
        axes[1].bar(x - 0.2, [ours["calibration"].get(l, 0) for l in labels], 0.4, label="ours")
        axes[1].bar(x + 0.2, [base["calibration"].get(l, 0) for l in labels], 0.4, label="fixed KF")
        axes[1].axhline(0.68, ls="--", c="gray")
        axes[1].axhline(0.95, ls="--", c="gray")
        axes[1].set_xticks(x, labels)
        axes[1].legend()
        axes[1].set_title("calibration coverage (dashed = nominal)")
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=120)
    except Exception as e:  # plots are best-effort on headless boxes
        print(f"plotting skipped: {e}")

    print(f"\nreport written to {out_dir}/report.json")
    sys.exit(0 if verdict["PASS"] else 1)


if __name__ == "__main__":
    main()
