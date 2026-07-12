# 第 4 轮：mini 数据包发回（解锁本地开发）+ trainval 全量提取

> tag：`round-4`　`cd ~/BeliefGauss && git fetch --tags && git checkout round-4`
> 本轮重点是【1. 打包发回】——它解锁我们这边的本地开发，请最优先做。

## 1. 打包三样东西发回（共约 300–500MB，网盘即可）

```bash
cd ~
mkdir -p bundle_r4
# (a) mini 的 Gaussian 缓存（~127MB）
cp -r ~/data_local/cache/gauss_mini bundle_r4/
# (b) mini 10 个场景的 Occ3D gts（含 mask_camera，遮挡实验的标签）
python - << 'EOF'
import json, shutil, os
from pathlib import Path
BG = os.environ['BG_DATA']
scenes = {s['name'] for s in json.loads((Path(BG)/'nuscenes/v1.0-mini/scene.json').read_text())}
src = Path(BG)/'occ3d/gts'; dst = Path.home()/'bundle_r4/occ3d_gts_mini'
n = 0
for d in src.iterdir():
    if d.name in scenes:
        shutil.copytree(d, dst/d.name, dirs_exist_ok=True); n += 1
print(f'copied {n} scenes (expect 10)')
EOF
# (c) nuScenes mini 的 metadata（json 目录，不含图像）
cp -r $BG_DATA/nuscenes/v1.0-mini bundle_r4/v1.0-mini

zip -r bundle_r4.zip bundle_r4   # 发回这个 zip
```

## 2. trainval 下载完成后：全量提取（4 卡分片，约 3.5 小时）

前提：`$BG_DATA/nuscenes/` 下 trainval 的 samples/sweeps/v1.0-trainval 就位。
输出到本地盘（全量缓存约 11GB）。4 张卡各开一个进程：

```bash
conda activate gf2
cd ~/BeliefGauss
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/gf2_extract_gaussians.py \
    --gf-root third_party/GaussianFormer \
    --config config/prob/nuscenes_gs6400.py \
    --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
    --pkl third_party/GaussianFormer/data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl \
    --mini-version none \
    --out ~/data_local/cache/gauss_train \
    --shard $i --num-shards 4 > extract_train_$i.log 2>&1 &
done
# 等 train 跑完后同样跑 val（换 val pkl，--out ~/data_local/cache/gauss_val）
```

注：GPU0 若被占用，把循环改成 `for i in 1 2 3` 并 `--num-shards 3`。

## 3. 上轮欠的两个数（一分钟）

```bash
dd if=/dev/zero of=$BG_DATA/iotest.bin bs=1M count=4096 oflag=direct 2>&1 | tail -1
dd if=$BG_DATA/iotest.bin of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
rm $BG_DATA/iotest.bin
```

## 4. 发回清单

1. `bundle_r4.zip`（最优先）；
2. trainval 提取的 4 个 `report_shard*.json` + 总 wallclock + `du -sh ~/data_local/cache/gauss_train`；
3. §3 两个 dd 数字；
4. 磁盘余量 `df -h / | tail -1`。
