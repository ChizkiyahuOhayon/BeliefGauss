# 第 4 轮：mini 数据包发回 + OccWorld 基线开训（GPU 饱和开始）+ trainval 提取

> tag：`round-4`　`cd ~/BeliefGauss && git fetch --tags && git checkout round-4`
> 顺序：【1. 打包发回】（10 分钟，解锁我们本地开发）→【1.6 OccWorld 开训】（今天就能开，
> 不需要等图像下载！让 GPU 从现在起一直有活干）→【2. trainval 提取】（图像到位后）。

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

## 1.6 OccWorld 基线复现——今天就开训（不需要图像！）

OccWorld（CVPR'24 占据世界模型）是我们表 1 的主要对比基线 + Plan C 的被试。
它的训练**只读 Occ3D gts**（已在 NAS）+ 2 个 pkl + nuScenes 的 metadata，不读相机图像。

```bash
# (a) 代码 + pkl（清华云盘，国内直连）
cd ~/BeliefGauss/third_party
git clone https://github.com/wzzheng/OccWorld.git
cd OccWorld && mkdir -p data out
# 从 https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/ 下载两个 pkl 放到 data/：
#   nuscenes_infos_train_temporal_v3_scene.pkl / nuscenes_infos_val_temporal_v3_scene.pkl
# 官方预训练模型（先用来验证评测链路）：https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/ 放到 out/

# (b) 数据布局：它要 data/nuscenes/gts/<scene>/<token>/labels.npz
#     注意：CIFS 网盘上不能建软链，且训练高频随机读 labels.npz——
#     把 gts 拷到本地盘（先看大小，<40G 就拷），软链全部建在本地：
du -sh $BG_DATA/occ3d/gts        # 把这个数字也发回
cp -r $BG_DATA/occ3d/gts ~/data_local/occ3d_gts
mkdir -p data/nuscenes
for d in maps samples sweeps v1.0-trainval lidarseg; do
  ln -s $BG_DATA/nuscenes/$d data/nuscenes/$d 2>/dev/null
done
ln -s ~/data_local/occ3d_gts data/nuscenes/gts

# (c) 环境：先直接用 gf2 env 试（同为 py3.8/cu118/mmdet3d 栈）；缺包就 pip 装并记录
conda activate gf2
pip install einops timm  # 若已装会跳过

# (d) 先验证评测链路（用官方权重，~30 分钟）：
CUDA_VISIBLE_DEVICES=1 python eval_metric_stp3.py --py-config config/occworld.py --work-dir out/occworld_eval \
  2>&1 | tee occworld_eval.log
# 成功标准：输出 forecasting mIoU/IoU 表，与论文表 1 数量级一致

# (e) 从头训练（两阶段，各占 1 张卡，共 2-4 天——这就是 GPU 饱和的主力）：
CUDA_VISIBLE_DEVICES=2 nohup python train.py --py-config config/train_vqvae.py \
  --work-dir out/vqvae > train_vqvae.log 2>&1 &
# VQVAE 训完后（看 log 收敛），改 config/train_occworld.py 里的 VQVAE ckpt 路径，再：
# CUDA_VISIBLE_DEVICES=2 nohup python train.py --py-config config/train_occworld.py \
#   --work-dir out/occworld > train_occworld.log 2>&1 &
```

任何一步报错：发完整 log，继续做其他步骤，不要卡在这里。

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
