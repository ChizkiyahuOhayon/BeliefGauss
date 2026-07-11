# BeliefGauss

Calibrated probabilistic scene memory for uncertainty-aware end-to-end driving.
Target: CVPR 2027. Research plan: [docs/RESEARCH_PLAN_v3.1.md](docs/RESEARCH_PLAN_v3.1.md).

**Core claim**: a driving world model should maintain a *calibrated, persistent
belief* over dynamic scene instances — belief covariance grows under occlusion,
shrinks on re-observation, and propagates in closed form
(`Σ_eff = S_shape + P_belief`) into planning risk.

## Layout

```
beliefgauss/
  memory/     predict-associate-update belief memory (instance slots)
  sim/        synthetic observation streams (occlusion, heteroscedastic noise)
  utils/      numerically careful linear algebra
configs/      one YAML per experiment
scripts/      one entry script per experiment; writes outputs/<run>/report.json
tests/        CPU unit tests (run before every GPU round)
server/       GPU-server setup & per-round command lists
docs/         research plan
```

## Quick start (CPU, no GPU needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # unit tests, ~2 s
.venv/bin/python scripts/run_smoke_synthetic.py \
    --config configs/smoke_synthetic.yaml --out outputs/smoke_v0
```

The smoke test trains the belief memory on simulated multi-object streams with
occlusion and checks Gate-0: NLL beats a fixed-noise Kalman baseline,
coverage@68/95 is calibrated, belief covariance grows through occlusion and
collapses on re-observation. Exit code 0 = pass; see `outputs/*/report.json`.

## Workflow

Development happens on a CPU-only machine; training runs on a remote 8×3090
node. Every GPU round: pull a tagged commit, run the single command in
`server/ROUND_XX.md`, send back the auto-generated `outputs/<run>` bundle.
Rules of engagement: `server/SETUP.md`.
