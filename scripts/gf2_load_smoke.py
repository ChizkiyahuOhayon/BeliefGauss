"""Data-free GaussianFormer-2 sanity check: build the model from config and
load the released checkpoint strictly. Requires NO dataset.

Run inside the `gf2` conda env:
  python scripts/gf2_load_smoke.py \
      --gf-root third_party/GaussianFormer \
      --config config/prob/nuscenes_gs6400.py \
      --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
      --out outputs/gf2_load_smoke

Prerequisite: the lifter init weights must exist at
  <gf-root>/out/prob/init/init.pth
Download (author-provided, issue #46):
  https://cloud.tsinghua.edu.cn/f/159a3370b4e843ddaec5/?dl=1
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--config", default="config/prob/nuscenes_gs6400.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/gf2_load_smoke")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt).resolve()

    gf_root = Path(args.gf_root).resolve()
    os.chdir(gf_root)          # configs use paths relative to the repo root
    sys.path.insert(0, str(gf_root))

    import torch
    from mmengine import Config
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(args.config)

    init_path = cfg.model["lifter"].get("pretrained_path")
    if init_path and not Path(init_path).exists():
        print(f"FATAL: lifter init weights missing: {gf_root / init_path}\n"
              "Download https://cloud.tsinghua.edu.cn/f/159a3370b4e843ddaec5/?dl=1 "
              f"and save it as {gf_root / init_path}")
        sys.exit(2)

    import model  # noqa: F401  registers custom modules
    net = build_segmentor(cfg.model)
    n_params = sum(p.numel() for p in net.parameters())

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = net.load_state_dict(sd, strict=False)

    per_module = {}
    for name, p in net.named_parameters():
        top = name.split(".")[0]
        per_module[top] = per_module.get(top, 0) + p.numel()

    report = {
        "config": args.config,
        "ckpt": str(ckpt_path),
        "n_params_total": n_params,
        "n_params_per_module": per_module,
        "n_ckpt_keys": len(sd),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "strict_load_ok": len(missing) == 0 and len(unexpected) == 0,
        "torch": torch.__version__,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("missing_keys", "unexpected_keys")}, indent=2))
    if not report["strict_load_ok"]:
        print(f"missing ({len(missing)}): {missing[:10]} ...")
        print(f"unexpected ({len(unexpected)}): {unexpected[:10]} ...")
        sys.exit(1)
    print("OK: checkpoint matches architecture exactly.")


if __name__ == "__main__":
    main()
