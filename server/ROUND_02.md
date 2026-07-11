# 第 2 轮（新机器版）：4×A40 环境重建 + NAS 数据布局 + GaussianFormer-2 复现

> tag：`round-2`　适用机器：4×NVIDIA A40 46GB，driver 535.309.01（CUDA 12.2），根盘 846G（剩 70G），NAS 127T 空闲
> 有任何一步报错：停下，把完整输出发回，不要自行修改。

## 0. 磁盘布局（已定：数据集上 NAS，热数据/环境留本地）

```bash
# 数据集统一放 NAS（127T 空闲）：
export BG_DATA=/home/smbu/dy/nas/beliefgauss_data
mkdir -p $BG_DATA/{nuscenes,occ3d}
echo 'export BG_DATA=/home/smbu/dy/nas/beliefgauss_data' >> ~/.bashrc

# 本地只放：conda 环境、代码、mini + Occ3D gts 副本（~20G）、之后的特征缓存
mkdir -p ~/data_local
```

**先测 NAS 读写速度（决定后续训练策略，结果必须发回）**：

```bash
# 写速度
dd if=/dev/zero of=$BG_DATA/iotest.bin bs=1M count=4096 oflag=direct 2>&1 | tail -1
# 顺序读速度
dd if=$BG_DATA/iotest.bin of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
# 小文件随机读（模拟训练读图，关键指标）
python - << 'EOF'
import os, time, random
base = os.environ['BG_DATA'] + '/iotest_small'
os.makedirs(base, exist_ok=True)
for i in range(2000):
    open(f'{base}/{i}.bin','wb').write(os.urandom(150_000))  # ~150KB 模拟一张jpg
t0=time.time(); idx=list(range(2000)); random.shuffle(idx)
for i in idx: open(f'{base}/{i}.bin','rb').read()
dt=time.time()-t0
print(f'small-file random read: {2000/dt:.0f} files/s, {2000*0.15/dt:.0f} MB/s')
EOF
rm -rf $BG_DATA/iotest.bin $BG_DATA/iotest_small
```

## 1. GPU 使用规约（共享机）

GPU0 上有别人的进程（dyslam）。我们的任务默认用 1–3 号卡：

```bash
export CUDA_VISIBLE_DEVICES=1,2,3   # 大训练前和机主确认 GPU0 是否可征用
```

## 2. 环境重建（新机器从零开始，按 SETUP.md §2–3，摘要如下）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

conda create -n beliefgauss python=3.10 -y && conda activate beliefgauss
cd ~ && git clone https://github.com/ChizkiyahuOhayon/BeliefGauss.git && cd BeliefGauss
git fetch --tags && git checkout round-2
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 复跑冒烟（新机器指纹，必须发回）
python -m pytest tests/ -q
python scripts/run_smoke_synthetic.py --config configs/smoke_synthetic.yaml --out outputs/smoke_a40
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.device_count(), torch.cuda.get_device_name(0))" > outputs/smoke_a40/env.txt
nvidia-smi >> outputs/smoke_a40/env.txt; df -h / $BG_DATA >> outputs/smoke_a40/env.txt
```

## 3. 数据下载（目标路径都在 NAS）

```bash
# nuScenes：OpenDataLab CLI（SETUP.md §1a），target 指向 $BG_DATA/nuscenes
#   优先 mini + metadata；trainval 全量直接开始后台下（NAS 空间管够）
# Occ3D gts：SETUP.md §1b，解压到 $BG_DATA/occ3d/gts/

# mini + gts 各复制一份到本地（快 IO，日常开发用）：
cp -r $BG_DATA/nuscenes/v1.0-mini ~/data_local/nuscenes_mini_meta 2>/dev/null
# （samples/sweeps 的 mini 部分与 gts 同理，共 ~20G）

# Occ3D 验证：
python - << 'EOF'
import numpy as np, glob, os
fs = glob.glob(os.environ['BG_DATA'] + '/occ3d/gts/*/*/labels.npz')[:3]
assert fs, 'gts 目录结构不对'
d = np.load(fs[0])
print('OK keys:', list(d.keys()), 'semantics', d['semantics'].shape, 'mask_camera', d['mask_camera'].shape)
EOF
```

## 4. GaussianFormer-2 独立环境 + 编译 + mini 推理

A40 = Ampere sm_86，与 3090 同架构，官方 cu118 wheel 直接可用：

```bash
conda create -n gf2 python=3.8.16 -y && conda activate gf2
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim && mim install mmcv==2.0.1 mmdet==3.0.0 mmsegmentation==1.0.0 mmdet3d==1.1.1
pip install spconv-cu117 timm

cd ~/BeliefGauss && mkdir -p third_party
git clone https://github.com/huang-yh/GaussianFormer.git third_party/GaussianFormer
cd third_party/GaussianFormer
cd model/encoder/gaussian_encoder/ops && pip install -e . && cd -
cd model/head/localagg && pip install -e . && cd -
cd model/head/localagg_prob && pip install -e . && cd -
cd model/head/localagg_prob_fast && pip install -e . && cd -

# 权重：README Model Zoo 的 GaussianFormer-2 (prob6400)，HF 链接走 hf-mirror
# 数据软链 + mini 推理（命令以其 README 为准）：
ln -s $BG_DATA/nuscenes data/nuscenes; ln -s $BG_DATA/occ3d data/occ3d
# 推理时采集吞吐：
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 5 > gf2_infer_gpu.log &
```

## 5. 发回清单

1. §0 三项 IO 测速输出（尤其 small-file random read）；
2. `outputs/smoke_a40` 整个目录打包（新机器 Gate-0 指纹）；
3. Occ3D 验证输出；
4. GaussianFormer-2 四个算子编译成功与否（失败发完整 log）；
5. mini 推理 mIoU 表 + `gf2_infer_gpu.log` + wallclock；
6. trainval 下载进度一句话。
