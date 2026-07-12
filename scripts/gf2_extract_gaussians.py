"""Extract per-frame Gaussian representations from GaussianFormer-2.

This is simultaneously (a) the forward-pass validation on real data and
(b) the feature-cache generator that decouples all downstream BeliefGauss
training from image IO (plan §3.3/§3.5: caching is mandatory on 4xA40+NAS).

No occupancy GT is needed: the SurroundOcc loader is dropped from the
pipeline; we only run images -> Gaussians (rep_only=True).

Run inside the `gf2` conda env from the BeliefGauss repo root:
  python scripts/gf2_extract_gaussians.py \
      --gf-root third_party/GaussianFormer \
      --config config/prob/nuscenes_gs6400.py \
      --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
      --pkl data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl \
      --pkl data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl \
      --mini-version v1.0-mini \
      --out $BG_DATA/cache/gauss_mini --limit 40

Output: one .npz per keyframe under <out>/<scene>/<idx>_<token>.npz with
means (N,3) / scales (N,3) / rotations (N,4) / opacities / semantics (fp16),
plus report.json with throughput and GPU memory stats.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def filter_pkls_to_version(pkl_paths, nusc_root, version, tmp_pkl):
    """Merge info pkls and keep only scenes present in the given nuScenes
    version (e.g. v1.0-mini).

    The pkls key scenes by scene *token* (32-char hash), while scene.json
    carries both token and human-readable name — match on either.
    Returns (filtered pkl path, kept keys, token->name map).
    """
    import json as _json

    import mmengine

    scene_file = Path(nusc_root) / version / "scene.json"
    assert scene_file.exists(), f"not found: {scene_file}"
    scene_meta = _json.loads(scene_file.read_text())
    names = {s["name"] for s in scene_meta}
    tokens = {s["token"] for s in scene_meta}
    tok2name = {s["token"]: s["name"] for s in scene_meta}
    keep = names | tokens

    infos, metadata, key_samples = {}, [], []
    for p in pkl_paths:
        data = mmengine.load(p)
        key_samples += list(data["infos"].keys())[:3]
        for scene, frames in data["infos"].items():
            if scene in keep and scene not in infos:
                infos[scene] = frames
        for scene, idx in data["metadata"]:
            if scene in keep:
                metadata.append((scene, idx))
    if not infos:
        raise SystemExit(
            f"no overlap between pkls and {version} scenes.\n"
            f"  pkl scene keys look like : {key_samples}\n"
            f"  {version} names e.g.     : {sorted(names)[:3]}\n"
            f"  {version} tokens e.g.    : {sorted(tokens)[:3]}\n"
            "Send this output back for diagnosis.")
    mmengine.dump({"infos": infos, "metadata": metadata}, tmp_pkl)
    return tmp_pkl, sorted(infos.keys()), tok2name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--config", default="config/prob/nuscenes_gs6400.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pkl", action="append", required=True,
                    help="info pkl(s); pass twice for train+val")
    ap.add_argument("--mini-version", default="v1.0-mini",
                    help="nuScenes version dir used to filter scenes; "
                         "'none' = keep all scenes in the pkls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index for multi-GPU extraction")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(os.path.expandvars(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt).resolve()
    pkl_paths = [str(Path(p).resolve()) for p in args.pkl]

    gf_root = Path(args.gf_root).resolve()
    os.chdir(gf_root)
    sys.path.insert(0, str(gf_root))

    import numpy as np
    import torch
    from mmengine import Config
    from mmseg.models import build_segmentor
    from torch.utils.data import DataLoader

    # pointops is an undocumented GaussianFormer-2 dependency (issue #47/#53).
    # The community build (point-transformer lib) names the function
    # `furthestsampling`; GaussianLifterV2 imports `farthest_point_sampling`.
    # Alias it BEFORE `import model` so the from-import inside the lifter
    # resolves, regardless of which pointops variant is installed.
    try:
        import pointops
        if not hasattr(pointops, "farthest_point_sampling"):
            from pointops.functions.pointops import furthestsampling
            pointops.farthest_point_sampling = furthestsampling
        print("pointops OK:", pointops.farthest_point_sampling)
    except ImportError as e:
        raise SystemExit(
            f"pointops missing ({e}). Install it first — see server/ROUND_03.md §1.5")

    import model  # noqa: F401
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal

    cfg = Config.fromfile(args.config)

    # ---- dataset: test pipeline minus occupancy loading ------------------
    ds_cfg = dict(cfg.val_dataset_config)
    pipeline = [t for t in ds_cfg["pipeline"]
                if "Occupancy" not in t["type"]]
    ds_cfg["pipeline"] = pipeline
    ds_cfg["return_keys"] = ["img", "projection_mat", "image_wh"]

    if args.mini_version != "none":
        tmp_pkl = str(out_dir / "_filtered_infos.pkl")
        data_root = ds_cfg.get("data_root", "data/nuscenes/")
        pkl_path, scenes, tok2name = filter_pkls_to_version(
            pkl_paths, data_root, args.mini_version, tmp_pkl)
        print(f"scenes kept ({len(scenes)}): "
              f"{[tok2name.get(s, s) for s in scenes]}")
        ds_cfg["imageset"] = pkl_path
    else:
        assert len(pkl_paths) == 1, "use one pkl when mini-version=none"
        ds_cfg["imageset"] = pkl_paths[0]
        tok2name = {}

    dataset = OPENOCC_DATASET.build(ds_cfg)
    if args.num_shards > 1:
        dataset.keyframes = dataset.keyframes[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}")
    print(f"frames: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=custom_collate_fn_temporal, pin_memory=True)

    # ---- model ------------------------------------------------------------
    net = build_segmentor(cfg.model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = net.load_state_dict(sd, strict=True)
    net = net.to(args.device).eval()

    # GaussianLifterV2.forward reads metas['occ_label']/['occ_cam_mask'] to
    # build 'pixel_gt' — a training-loss target that never influences the
    # Gaussian representation (returned as 'pixel_gt' only). We dropped the
    # occupancy loader, so feed dummies to keep the exact official inference
    # path (the benchmarking=True escape hatch would switch FPS to a chunked
    # approximation and add randomness — avoid).
    occ_res = list(net.lifter.occ_resolution)
    empty_label = int(net.lifter.empty_label)
    dummy_occ = torch.full([1] + occ_res, empty_label,
                           dtype=torch.long, device=args.device)
    dummy_mask = torch.zeros([1] + occ_res, dtype=torch.bool, device=args.device)

    keyframes = dataset.keyframes
    times, n_saved = [], 0
    torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.limit and n_saved >= args.limit:
                break
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)
            imgs = batch.pop("img")
            batch["occ_label"] = dummy_occ
            batch["occ_cam_mask"] = dummy_mask
            t0 = time.time()
            rep = net(imgs=imgs, metas=batch, rep_only=True)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.time() - t0)

            g = rep[-1]["gaussian"]  # last-layer GaussianPrediction
            scene, idx = keyframes[i]
            info = dataset.scene_infos[scene][idx]
            token = info.get("token", f"{idx:04d}") if isinstance(info, dict) else f"{idx:04d}"

            # ego pose + timestamp: required downstream for temporal
            # alignment of the belief memory (static-Gaussian ego warp)
            from pyquaternion import Quaternion
            pose = info["data"]["LIDAR_TOP"]["pose"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(pose["rotation"]).rotation_matrix
            ego2global[:3, 3] = np.asarray(pose["translation"])

            scene_dir = out_dir / tok2name.get(scene, scene)
            scene_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                scene_dir / f"{idx:04d}_{token}.npz",
                means=g.means[0].cpu().numpy().astype(np.float16),
                scales=g.scales[0].cpu().numpy().astype(np.float16),
                rotations=g.rotations[0].cpu().numpy().astype(np.float16),
                opacities=g.opacities[0].cpu().numpy().astype(np.float16),
                semantics=g.semantics[0].cpu().numpy().astype(np.float16),
                ego2global=ego2global.astype(np.float64),
                timestamp=np.float64(info["timestamp"] / 1e6),
                sample_token=np.array(token),
            )
            n_saved += 1
            if i % 20 == 0:
                print(f"[{i}] {scene}/{idx} {times[-1]*1000:.0f} ms", flush=True)

    files = list(out_dir.rglob("*.npz"))
    report = {
        "frames_saved": n_saved,
        "mean_ms_per_frame": 1000 * float(np.mean(times[2:])) if len(times) > 2 else None,
        "fps": float(1.0 / np.mean(times[2:])) if len(times) > 2 else None,
        "gpu_peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9
        if args.device.startswith("cuda") else None,
        "avg_npz_size_mb": float(np.mean([f.stat().st_size for f in files]) / 1e6)
        if files else None,
        "total_cache_size_mb": float(sum(f.stat().st_size for f in files) / 1e6),
        "torch": torch.__version__,
    }
    report_name = "report.json" if args.num_shards == 1 \
        else f"report_shard{args.shard}.json"
    (out_dir / report_name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
