# GPU 服务器环境配置（8×RTX 3090）

> 执行人：请按顺序复制粘贴。每一步失败请把**完整报错**和 `env.txt` 发回，不要自行调整后继续。
> **第 1 步（数据下载）耗时最长，最先启动**，让它在后台跑；第 2-4 步（环境+验证）约 30 分钟，与下载并行完成。

## 1. 数据集下载（最先启动，后台进行，磁盘需求 ≈ 600 GB）

> 国内直连 nuScenes 官网（AWS 美区）很慢。按下面的优先顺序走国内渠道。

### 1a. nuScenes —— 首选 OpenDataLab（上海AI实验室，国内 CDN）

下载优先级：先 **`v1.0-mini`（4 GB，今天就要）**，`v1.0-trainval` 全量（~550 GB）后台慢慢下，W5（约 8 月中）前到位即可；CAN bus expansion 小，顺手下。

1. 注册 https://opendatalab.com （手机号即可），数据集页：
   https://opendatalab.com/OpenDataLab/nuScenes/download
2. 用官方 CLI 下载（断点续传，国内一般几十 MB/s）：

```bash
pip install openxlab
openxlab login          # 填 opendatalab 网页个人中心生成的 Access Key
mkdir -p ~/data && cd ~/data
# 文件名以网页文件列表为准；先下 mini + metadata，trainval 逐卷 nohup 挂后台
nohup openxlab dataset download --dataset-repo OpenDataLab/nuScenes \
    --source-path <网页上的 mini 包路径> --target-path ./nuscenes_dl > dl_mini.log 2>&1 &
```

**备选**（OpenDataLab 不行时）：nuScenes 官网 https://www.nuscenes.org/nuscenes#download 注册后，下载 region 选 **Asia**（新加坡节点，比默认美区快很多），拿签名链接 `nohup wget -c "<链接>" &`；或用社区脚本 https://github.com/li-xl/nuscenes-download 账号密码自动逐卷下载（设 `region='asia'`）。

解压后目录结构应为：`~/data/nuscenes/{samples,sweeps,maps,v1.0-trainval,v1.0-mini}`

### 1b. Occ3D-nuScenes 占据标签 —— 只需 gts 包（几个 GB，不是 50 GB 全量）

我们只用它的 `gts`（含 **mask_camera** 相机可见性 mask——遮挡实验的监督信号，缺了招牌实验做不了）；图像用 1a 的 nuScenes 原图，**不要下 Occ3D 的 imgs**。

- 入口：https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction
  （OccWorld / GaussianFormer 系用的就是这份 `gts.tar.gz`，README 提供多个下载源）
- Google Drive 国内不可达时：优先其 OpenDataLab / 百度网盘源；HuggingFace 副本走 hf-mirror（先做 §2 的 `HF_ENDPOINT` 配置）后用
  `huggingface-cli download <repo_id> --repo-type dataset --local-dir ~/data/occ3d`
- 解压到 `~/data/occ3d/gts/`，验证每帧 `labels.npz` 内含 `semantics / mask_lidar / mask_camera` 三个数组。

下载启动后即可进行第 2 步，不用等。

## 2. 基础环境（conda + 国内镜像）

```bash
# 如果没有 conda：
# wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash && exec bash

# pip 用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# HuggingFace 镜像（下权重/数据用）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

conda create -n beliefgauss python=3.10 -y
conda activate beliefgauss
```

## 3. 克隆仓库 + 本项目依赖

```bash
git clone https://github.com/ChizkiyahuOhayon/BeliefGauss.git
cd BeliefGauss
git checkout round-0
# 3090 = Ampere (sm_86)，装 CUDA 12.x 的 torch
pip install torch --index-url https://download.pytorch.org/whl/cu121 \
  || pip install torch   # 若 pytorch 官方源慢，清华源默认 wheel 也带 CUDA
pip install -r requirements.txt
```

## 4. 验证（第 0 轮，必须发回结果）

```bash
python -m pytest tests/ -q                       # 应全部通过
python scripts/run_smoke_synthetic.py \
    --config configs/smoke_synthetic.yaml --out outputs/smoke_gpu_check
# 记录环境指纹
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.device_count(), torch.cuda.get_device_name(0))" > outputs/smoke_gpu_check/env.txt
nvidia-smi >> outputs/smoke_gpu_check/env.txt
df -h ~/data >> outputs/smoke_gpu_check/env.txt   # 顺便报磁盘余量

# 打包发回
cd outputs && zip -r smoke_gpu_check.zip smoke_gpu_check && cd ..
```

**发回**：`outputs/smoke_gpu_check.zip`（内含 report.json / curves.png / env.txt）+ 一句话报数据下载进度。

## 5. GaussianFormer-2 复现（第 1 轮预告，命令随 ROUND_01.md 提供）

```bash
git clone https://github.com/huang-yh/GaussianFormer.git third_party/GaussianFormer
# 其 mmcv/mmdet3d 依赖栈与本仓库隔离在单独 conda env，ROUND_01.md 给出锁定版本
```

## 通用规则

1. 只跑 tagged commit：`git fetch --tags && git checkout <tag>`（每轮的 tag 写在 ROUND_XX.md 里）。
2. 每次运行自动生成 `outputs/<run>/report.json` — 整个目录打包发回，不要挑文件。
3. 报错 = 立即停止 + 发回完整 log；不要现场改代码。
4. 长任务（下载、训练）一律 `nohup ... &` 或 tmux，防 ssh 断连。
