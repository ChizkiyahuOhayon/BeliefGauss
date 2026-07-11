# 第 2 轮：磁盘处理 + mini 数据 + GaussianFormer-2 复现

> tag：`round-2`　预计人工操作 1–2 小时 + 下载/编译等待
> 有任何一步报错：停下，把完整输出发回，不要自行修改。

## 0.【阻塞项】磁盘触底（99% 满，剩 17GB）—— 必须先做

上轮 `df -h` 显示根分区 878G 只剩 17G。nuScenes 全量要 ~550G，加上特征缓存至少需要 **700G 连续可用**。请执行并把输出全部发回：

```bash
df -h                                   # 所有挂载点（看有没有第二块盘/NAS）
lsblk                                   # 物理盘列表（有没有未挂载的盘）
du -sh /home/* /data* /mnt/* 2>/dev/null | sort -rh | head -20   # 大头在哪
```

然后三选一（发回你们的选择）：
- **A**：有第二块盘/可加盘 → 挂载后把数据目录设为那块盘，后续所有 `~/data` 换成新路径；
- **B**：能清理出 ≥700G（旧实验产物、别人的数据集）→ 清理后发回新的 `df -h`；
- **C**：都不行 → 告诉我们，我们改用"流式解压 + 只保留 keyframe"方案（有损，尽量避免）。

**mini + Occ3D gts 约需 20G**：若暂时清不出大空间，至少腾出 30G，本轮照常进行。

## 1. 数据（本轮只要 mini + Occ3D gts）

```bash
# nuScenes mini（OpenDataLab，见 SETUP.md §1a；或官网 Asia 区直下，4GB）
# 解压到 ~/data/nuscenes/，应有 v1.0-mini/ samples/ sweeps/ maps/

# Occ3D gts（CVPR2023 challenge 版，见 SETUP.md §1b）
# 解压到 ~/data/occ3d/gts/
# 快速验证：
python - << 'EOF'
import numpy as np, glob
fs = glob.glob('/root/data/occ3d/gts/*/*/labels.npz')[:3] or \
     glob.glob(__import__('os').path.expanduser('~/data/occ3d/gts/*/*/labels.npz'))[:3]
assert fs, 'gts 目录结构不对'
d = np.load(fs[0])
print('OK keys:', list(d.keys()), 'semantics', d['semantics'].shape, 'mask_camera', d['mask_camera'].shape)
EOF
```

## 2. GaussianFormer-2 独立环境（与 beliefgauss env 隔离）

官方要求 python3.8 + torch 2.0.0 cu118（3090/sm_86 兼容）：

```bash
conda create -n gf2 python=3.8.16 -y && conda activate gf2
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim && mim install mmcv==2.0.1 mmdet==3.0.0 mmsegmentation==1.0.0 mmdet3d==1.1.1
pip install spconv-cu117 timm

cd ~/BeliefGauss && mkdir -p third_party
git clone https://github.com/huang-yh/GaussianFormer.git third_party/GaussianFormer
cd third_party/GaussianFormer
# 编译 4 个自定义 CUDA 算子（每个都要成功，报错即停）
cd model/encoder/gaussian_encoder/ops && pip install -e . && cd -
cd model/head/localagg && pip install -e . && cd -
cd model/head/localagg_prob && pip install -e . && cd -
cd model/head/localagg_prob_fast && pip install -e . && cd -
```

## 3. 权重 + mini 推理 + 吞吐

```bash
# 权重：GaussianFormer README 的 Model Zoo 下载 GaussianFormer-2 (prob, 6400 Gaussians)
# 链接若是 HuggingFace 走 hf-mirror；若是清华云盘直接 wget
# 放到 third_party/GaussianFormer/ckpts/

# 按其 README 跑 mini 评测（数据路径软链）：
ln -s ~/data/nuscenes data/nuscenes 2>/dev/null; ln -s ~/data/occ3d data/occ3d 2>/dev/null
# 具体 eval 命令以其 README 为准，典型形如：
# python eval.py --py-config config/prob/nuscenes_gs6400.py --work-dir out/eval_mini \
#     --resume-from ckpts/<权重>.pth

# 记录吞吐：单卡推理时另开终端采集
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5 > gf2_infer_gpu.log &
```

## 4. 发回清单

1. §0 的磁盘三条输出 + 你们的选择（A/B/C）；
2. Occ3D 验证脚本的输出；
3. GaussianFormer-2 编译是否全部成功（失败则发完整编译 log）；
4. mini 推理的指标输出（mIoU 表）+ `gf2_infer_gpu.log` + 大致 wallclock；
5. 一句话：trainval 下载进度（若已开始）。
