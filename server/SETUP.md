# GPU 服务器环境配置（8×RTX 3090）

> 执行人：请按顺序复制粘贴。每一步失败请把**完整报错**和 `env.txt` 发回，不要自行调整后继续。
> 预计耗时：环境 ~30 分钟；nuScenes 数据下载视网速（可后台进行，见 §4）。

## 1. 基础环境（conda + 国内镜像）

```bash
# 如果没有 conda：
# wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash && exec bash

# pip 用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# HuggingFace 镜像（以后下权重用）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

conda create -n beliefgauss python=3.10 -y
conda activate beliefgauss
```

## 2. 克隆仓库 + 本项目依赖

```bash
git clone https://github.com/ChizkiyahuOhayon/BeliefGauss.git
cd BeliefGauss
# 3090 = Ampere (sm_86)，装 CUDA 12.x 的 torch
pip install torch --index-url https://download.pytorch.org/whl/cu121 \
  || pip install torch   # 若 pytorch 官方源慢，清华源的默认 wheel 也带 CUDA
pip install -r requirements.txt
```

## 3. 验证（第 0 轮，必须发回结果）

```bash
python -m pytest tests/ -q                       # 应全部通过
python scripts/run_smoke_synthetic.py \
    --config configs/smoke_synthetic.yaml --out outputs/smoke_gpu_check
# 记录环境指纹
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.device_count(), torch.cuda.get_device_name(0))" > outputs/smoke_gpu_check/env.txt
nvidia-smi >> outputs/smoke_gpu_check/env.txt

# 打包发回
cd outputs && zip -r smoke_gpu_check.zip smoke_gpu_check && cd ..
```

**发回**：`outputs/smoke_gpu_check.zip`（内含 report.json / curves.png / env.txt）。

## 4. 数据集（可与 §3 并行，磁盘需求 ≈ 600 GB）

### nuScenes trainval（~550 GB，官网需注册账号）
- https://www.nuscenes.org/nuscenes#download 注册后下载 `v1.0-trainval`（10 个分卷 + metadata）与 `v1.0-mini`（先下这个，4 GB，够 W2-W4 用）
- 解压到 `data/nuscenes/`，目录结构应为 `data/nuscenes/{samples,sweeps,v1.0-trainval,v1.0-mini}`

### Occ3D-nuScenes（~50 GB，遮挡实验的关键标签）
- https://github.com/Tsinghua-MARS-Lab/Occ3D 的下载链接（HuggingFace 走 hf-mirror）
- 解压到 `data/occ3d/`，确认包含 `mask_camera`（相机可见性 mask）

**优先级**：先 mini + Occ3D 对应部分；trainval 后台慢慢下，W5 之前到位即可。

## 5. GaussianFormer-2 复现（第 1 轮预告，命令随 ROUND_01.md 提供）

```bash
git clone https://github.com/huang-yh/GaussianFormer.git third_party/GaussianFormer
# 其 mmcv/mmdet3d 依赖栈与本仓库隔离在单独 conda env，ROUND_01.md 给出锁定版本
```

## 通用规则

1. 只跑 tagged commit：`git fetch --tags && git checkout <tag>`（每轮的 tag 写在 ROUND_XX.md 里）。
2. 每次运行自动生成 `outputs/<run>/report.json` — 整个目录打包发回，不要挑文件。
3. 报错 = 立即停止 + 发回完整 log；不要现场改代码。
4. 长训练用 `nohup ... > train.log 2>&1 &` 或 tmux，防 ssh 断连。
