# 第 3 轮：GaussianFormer-2 加载验证 + mini 上提取 Gaussians（不需要 SurroundOcc）

> tag：`round-3`　先 `cd ~/BeliefGauss && git fetch --tags && git checkout round-3`
> **决策已定：不下载 SurroundOcc、不跑官方 eval.py。** 我们只把 GaussianFormer-2 当"图像→Gaussians"编码器用，
> 提取脚本绕过了它对 occupancy GT 的依赖。baseline 的 mIoU 数字引用论文即可。

## 1. 补两个文件（都是清华云盘，国内直连）

```bash
conda activate gf2
cd ~/BeliefGauss/third_party/GaussianFormer

# (a) lifter 初始化权重（作者在 issue #46 补发的；模型构建时必须存在）
mkdir -p out/prob/init
wget -O out/prob/init/init.pth "https://cloud.tsinghua.edu.cn/f/159a3370b4e843ddaec5/?dl=1"

# (b) 数据 pkl（README "Download pkl files" 那个云盘目录）：
#     https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/
#     下载 nuscenes_infos_train_sweeps_occ.pkl 和 nuscenes_infos_val_sweeps_occ.pkl
mkdir -p data/nuscenes_cam   # 两个 pkl 放这里
```

## 1.5 补装 pointops（GaussianFormer-2 的隐藏依赖，安装文档漏写）

`GaussianLifterV2` 需要 `pointops`（官方 issue #47/#53 确认，installation.md 未列）。
用社区修好编译问题的版本（源自 point-transformer，已去掉 THC/THC.h）：

```bash
conda activate gf2
cd ~/BeliefGauss/third_party/GaussianFormer
git clone https://github.com/xieyuser/pointops.git pointops
cd pointops
TORCH_CUDA_ARCH_LIST="8.6" python setup.py install   # 编译 pointops_cuda（A40=sm_86）
cd ..
# 验证（在 GaussianFormer 根目录下执行）：
python -c "import pointops; from pointops.functions.pointops import furthestsampling; print('pointops OK')"
```

注意：clone 出的 `pointops/` 文件夹必须留在 GaussianFormer 根目录下（import 需要），不要删。
函数名差异（furthestsampling vs farthest_point_sampling）由我们的提取脚本自动加别名，无需手改 `__init__.py`。

## 2. 加载冒烟（数据无关，~1 分钟）

```bash
cd ~/BeliefGauss
python scripts/gf2_load_smoke.py \
    --gf-root third_party/GaussianFormer \
    --config config/prob/nuscenes_gs6400.py \
    --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
    --out outputs/gf2_load_smoke
# 期望最后一行: "OK: checkpoint matches architecture exactly."
```

## 3. mini 上提取 Gaussians（真实数据前向 + 吞吐实测）

前提：`third_party/GaussianFormer/data/nuscenes → $BG_DATA/nuscenes` 软链已建好（R2 已做）。

```bash
export CUDA_VISIBLE_DEVICES=1
cd ~/BeliefGauss
# 先 40 帧试跑：
python scripts/gf2_extract_gaussians.py \
    --gf-root third_party/GaussianFormer \
    --config config/prob/nuscenes_gs6400.py \
    --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
    --pkl third_party/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl \
    --pkl third_party/GaussianFormer/data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl \
    --mini-version v1.0-mini \
    --out '$BG_DATA/cache/gauss_mini_trial' --limit 40

# 试跑成功后全量 mini（~404 帧，去掉 --limit；注意缓存写到本地盘，NAS 写只有 5MB/s）：
python scripts/gf2_extract_gaussians.py \
    --gf-root third_party/GaussianFormer \
    --config config/prob/nuscenes_gs6400.py \
    --ckpt third_party/GaussianFormer/ckpts/state_dict.pth \
    --pkl third_party/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl \
    --pkl third_party/GaussianFormer/data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl \
    --mini-version v1.0-mini \
    --out ~/data_local/cache/gauss_mini
```

## 4. 顺手补测 NAS 顺序读写（R2 只发了小文件随机读 30MB/s 和写 5MB/s）

```bash
dd if=/dev/zero of=$BG_DATA/iotest.bin bs=1M count=4096 oflag=direct 2>&1 | tail -1
dd if=$BG_DATA/iotest.bin of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
rm $BG_DATA/iotest.bin
```

## 5. 发回清单

1. `outputs/gf2_load_smoke/report.json`；
2. 提取试跑的控制台输出 + `~/data_local/cache/gauss_mini/report.json`（重点：ms/帧、显存峰值、单帧 npz 大小）；
3. §4 两个 dd 数字；
4. trainval 下载进度一句话。

**若第 2/3 步报错**：完整 traceback 发回。常见问题预判：pkl 里 image 路径与软链不一致（发回一条报错路径即可，我们改脚本）；显存不够（不太可能，A40 46G）。
