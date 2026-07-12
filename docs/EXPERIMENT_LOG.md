# BeliefGauss 实验记录

> **用途**：每轮 GPU/本地实验的完整档案。论文 Method/Experiments 章节直接从这里写。
> **纪律**：每轮必须记录——动机、commit/tag、配置、完整指标（不许只记结论）、偏差与意外、分析、决策。
> **Method 事实账本**（§0）只收"已被实验验证的设计选择"，写论文时逐条对应消融/引用。

---

## 0. Method 事实账本（随轮次累积）

| # | 设计选择 | 验证来源 | 论文位置 |
|---|---|---|---|
| F1 | Joseph 形式 Kalman 更新在 32 步 BPTT 下保持 PSD、无数值发散 | R0/R1 冒烟 | §3.2 实现细节 |
| F2 | 学习的异方差 Q/R（softplus+floor 对角）显著优于固定噪声：NLL 2.69 vs 30.25，coverage@68 0.66 vs 0.42 | R0/R1 Gate-0 | §3.3 + 消融3 |
| F3 | 可见性门控存在更新（遮挡不衰减 α）+ Q 累积 ⇒ 信念协方差在 96% 遮挡窗内单调增长，再观测后收缩 26× | R0/R1 Gate-0 | §3.4 核心机制 |
| F4 | χ² soft-coverage 正则可微且有效：coverage@68=0.655（名义 0.68）、@95=0.895（名义 0.95） | R0/R1 Gate-0 | §3.5 校准目标 |
| F5 | 关联退火（GT teacher forcing → 学习门控贪心）冷启动稳定，最终关联准确率 97.3% | R0/R1 | §3.3 训练策略 |
| F6 | 整套 pipeline 跨平台确定性可复现（macOS CPU vs Linux 8×3090，同 seed 指标一致到 1e-7） | R1 | 可复现性声明 |

**已知未解问题**（进入后续轮次）：
- P1（R1 发现）：遮挡事件中槽存活率仅 56%（`persistence_rate=0.56`）——疑似遮挡前未出生 or 再观测时门控过紧导致重新出生（身份断裂）。修复目标 ≥85%。
- P2（R1 发现）：coverage@95=0.895 略欠（名义 0.95），95% 分位的校准正则权重或温度需调。

---

## Round 0 — 本地 CPU：数学正确性冒烟（Gate-0 首次通过）

- **日期**：2026-07-10　**执行**：本地 macOS（CPU, torch 2.8.0）　**commit**：`6ddb373`（tag `round-0` 初版）
- **动机**：计划 §9 条 5——在接图像前，用合成观测流验证 predict–associate–update + NLL + χ² 校准的数学正确性；同时预演消融 3（固定噪声经典 KF 基线）。
- **配置**：`configs/smoke_synthetic.yaml`（seed 42；2D 世界；4–8 目标分两类（行人样 σ_a=0.5 / 车样 σ_a=0.1）；观测噪声随距离增长 σ=0.2(1+d/20)；扇形遮挡区；杂波 Poisson(0.15)；32 槽容量；250 iter，前 80 iter GT teacher forcing；BPTT 窗 8；Adam 3e-3；损失权重 innov 1.0 / forecast 1.0 / calib 5.0 / exist 0.2）
- **过程偏差**：首次运行在 iter 1 崩溃——空槽位于原点导致可见性函数 `atan2(0,0)` 反向传播 NaN 污染参数 → S 非正定。修复：对退化输入 `torch.where` 安全替换（`sim/synthetic.py`）。此类"空槽参与全量张量计算"的坑在真实系统中同样存在，记入实现注意事项。
- **结果**（30 条 held-out episode）：

| 指标 | BeliefGauss | 固定噪声 KF |
|---|---|---|
| forecast NLL | **2.687** | 30.252 |
| RMSE (m) | **2.92** | 11.74 |
| coverage@68 | **0.655** | 0.418 |
| coverage@95 | **0.895** | 0.623 |
| 遮挡 P 单调增长比例 | 0.957 (23 事件) | 0.947 (19) |
| trace 增长/收缩比 | 45.0× / 0.038 | 18.8× / 0.060 |
| 再现定位误差 (m) | **0.386** | 0.422 |
| 遮挡存活率 | 0.561 | 0.463 |
| 关联准确率 | 0.973 (n=4886) | 0.983 (n=4647) |

- **Gate-0 判定**：6/6 通过（NLL 胜基线；coverage@68∈[0.58,0.78]；@95∈[0.88,0.995]；P 增长≥0.8；收缩<0.7；关联≥0.95）。
- **分析**：固定 KF 的 NLL 崩坏主要来自异方差错配（行人样目标的机动噪声被低估 5×）——这正是论文叙事"校准需要学习噪声"的最小证据。存活率 56% 是明确短板（→P1）。
- **决策**：进入 GPU 环境验证（R1）；P1/P2 留到 R3 前的本地迭代。

---

## Round 1 — GPU 服务器：环境验证 + 复现冒烟

- **日期**：2026-07-11（服务器时间）　**执行**：朋友，8×RTX 3090 节点　**commit**：`9f2dc8b`（tag `round-0`）
- **动机**：验证服务器环境可用、结果跨平台可复现；采集硬件/磁盘指纹。
- **环境指纹**：torch 2.5.1+cu121，driver 550.144.03（CUDA 12.4），8× RTX 3090 24GB 全部空闲可见；单元测试全过。
- **结果**：Gate-0 **6/6 通过**；全部指标与 R0 本地一致到 ~1e-7（同 seed 同 commit）→ 事实 F6。250 iter 冒烟 wallclock 114s（vs 本地 CPU ≈ 更慢；该冒烟不吃 GPU，仅验证栈）。
- **重大发现（阻塞项）**：`df -h` 显示根分区 878GB 已用 817GB，**仅剩 17GB（99%）**。nuScenes trainval（~550GB）无法落盘。mini(4GB)+Occ3D gts(~10GB 解压后) 勉强可放但无余量。
- **决策**：
  1. R2 第 0 步 = 磁盘处理：让执行人报告全部挂载点与可清理空间；若无第二块盘，需清理 ≥600GB 或加盘——**此项不解决，W5 全量训练不可能**。
  2. R2 主体 = mini + Occ3D gts 下载、GaussianFormer-2 独立环境（py3.8/torch2.0/cu118 + mmcv 2.0.1 栈 + 4 个自定义 CUDA 算子编译）、官方权重 mini 推理，采集实测吞吐（修 §3.5 算力表）。
  3. 本地并行：遮挡子集构建脚本 v0（服务 Plan C）；P1 存活率修复。

---

## Round 1.5 — 机器变更：8×3090 → 4×A40（磁盘阻塞项就此解决）

- **日期**：2026-07-12（服务器时间）　**性质**：基础设施变更，无实验
- **起因**：原 3090 机器内存不足，执行人换到另一台机器。
- **新机器指纹**（来自执行人发回的 `df -h` / `nvidia-smi`）：
  - GPU：**4× NVIDIA A40 46GB**（Ampere sm_86，与 3090 同架构），driver 535.309.01 / CUDA 12.2；
  - **共享机**：GPU0 常驻他人进程（dyslam, ~1.1GB）→ 我们默认 `CUDA_VISIBLE_DEVICES=1,2,3`，大训练前协调 GPU0；
  - 本地盘：846G 用 92%（剩 70G）——只放环境/代码/热数据；
  - **NAS（SMB/CIFS）：311T，127T 空闲** → 数据集落盘问题解决，R1 的阻塞项关闭（方案 A）。
- **影响评估**：
  - 算力：A40 单卡 ≈ 3090（FP32 37 vs 36 TFLOPS，显存带宽 696 vs 936 GB/s），卡数 8→4 ⇒ 数据并行 wallclock ≈ ×1.7–2；46GB 显存允许 micro-batch 2–4/卡 + 更长 BPTT 窗，部分抵消。计划 §3.5 预算表已按 ×1.7 重估，18 周时间线仍可行但 buffer 变薄——特征缓存策略从"省时优化"升级为**必做项**。
  - **新风险 R-IO**：训练数据在 CIFS 网络盘上，几十万小 jpg 的随机读可能成瓶颈。R2 新增三项 IO 测速（顺序写/顺序读/150KB 小文件随机读），若 small-file random read < 2000 files/s，则 W3 起把 samples keyframe 图像或 backbone 特征缓存到本地盘。
- **动作**：`server/ROUND_02.md` 全量重写为新机器版（NAS 布局、IO 测速、GPU 规约、环境从零重建、新机器冒烟指纹）。

---

## Round 2 — GaussianFormer-2 环境落地 + 官方评估流程受阻（转向决策）

- **日期**：2026-07-12　**执行**：朋友，4×A40 节点
- **完成**：双 conda 环境（beliefgauss: py3.10/torch2.5/cu121；gf2: py3.8/torch2.0/cu118）；4 个自定义 CUDA 算子全部编译成功；nuScenes mini + Occ3D gts + annotations.json 落 NAS，软链建好；GaussianFormer-2 Prob-6400 权重（state_dict.pth, 467MB）下载完成。
- **受阻**：官方 `eval.py` 跑不起来。根因分析（本地读源码确认）：
  1. `GaussianLifterV2.__init__` 在**模型构建时**就 `torch.load('out/prob/init/init.pth')`——该文件 README 未提及，作者在 issue #46 补发（清华云盘 `159a3370b4e843ddaec5`）；
  2. 官方数据管线 = 完整 nuScenes + SurroundOcc 标注 + 作者提供的 info pkl；mini 不被直接支持，SurroundOcc 我们本来就不用（主线用 Occ3D 的 mask_camera）。
- **IO 实测（重要）**：NAS 小文件随机读 ~30 MB/s（≈200 张图/s），写 ~5 MB/s。判定：**直接从 NAS 读图训练喂不饱 4×A40**；顺序读写数字缺失，R3 补测。
- **架构事实修正**：GaussianFormer-2 用 **R101-DCN backbone + 1600×864 输入**（计划 §3.1 原假设 R50/704×256 有误）——单帧编码比预想重，进一步强化"提取一次、缓存 Gaussians、下游全部离线训练"的路线。
- **关键源码发现（转向依据）**：
  1. 模型 forward 有 `rep_only=True` 模式，直接返回 `GaussianPrediction(means, scales, rotations, opacities, semantics)`——正是 BeliefGauss 消费的接口；
  2. occupancy GT 加载只是 pipeline 里一个可拆卸的 transform（`LoadOccupancySurroundOcc`），提取 Gaussians 完全不需要 GT；
  3. 数据集类支持自定义 `return_keys`，image-only 前向可行。
- **决策（回应执行人的四个选项）**：
  1. **放弃**复现官方 SurroundOcc 评估（选项1）——baseline mIoU 引论文数字即可，不进我们任何主表；SurroundOcc 数据不下载；
  2. **采纳**选项 2+3 合并：写了 `scripts/gf2_load_smoke.py`（数据无关的构建+严格载权重验证）和 `scripts/gf2_extract_gaussians.py`（mini 上 image→Gaussians 提取，绕过 GT，pkl 自动过滤到 mini 场景）——后者同时就是特征缓存生成器；
  3. **IO 对策**（回应选项4）：不专门"解决"NAS，架构上绕过——图像只在提取时读一遍（顺序、一次性），缓存的 Gaussian npz（fp16，估 <1MB/帧，待 R3 实测）放本地盘，下游 belief memory 训练全部读本地缓存。trainval 全量提取一次 ≈ 34k 帧 × 编码耗时（R3 实测后估算）。
- **下一轮**：ROUND_03 = init.pth + pkl 下载 → 加载冒烟 → mini 提取（40 帧试跑 + 全量 404 帧）→ 补 dd 顺序读写。

---

## Round 3（完成）— GaussianFormer-2 提取链路全线跑通，mini 404 帧落盘

**最终结果（2026-07-11 服务器回传 report.json）**：
| 指标 | 数值 | 推论 |
|---|---|---|
| frames_saved | 404（mini 全量） | 10 场景全覆盖 |
| 纯前向耗时 | **1439 ms/帧**（0.69 fps, fp32, batch 1, A40 单卡） | trainval 34k 帧：单卡 ~13.7h，**4 卡分片 ~3.5h** |
| GPU 显存峰值 | **2.6 GB** | 分片并行无压力；将来 batch>1 还有余量 |
| 单帧 npz | **0.31 MB**（fp16, 无 128 维实例特征） | trainval 全量缓存 **≈11 GB**，本地盘轻松容纳，NAS IO 风险就此关闭 |
| mini 缓存总量 | 127 MB | **可直接打包发回本地**——真实数据的本地 CPU 开发就此解锁（ROUND_04 第 1 项） |

**待定决策入账**：npz 目前只存了 (μ, scale, rot, opacity, semantics)，未存 128 维实例特征 `rep_features`。若 belief memory 的槽特征需要它，trainval 缓存将从 11GB 涨到 ~67GB（本地盘仍可容纳但紧张）。先用 18 维 semantics 作槽特征起步，消融后再决定是否二次提取。

### 过程记录（三个卡点，均已修复）

- **日期**：2026-07-12　**执行**：朋友（步骤 1–2）+ 本地修复
- **通过**：`gf2_load_smoke.py` 输出 "OK: checkpoint matches architecture exactly."（strict_load_ok=true）——权重、架构、环境三者互相咬合确认。init.pth / 两个 pkl 均已就位。
- **失败**：`gf2_extract_gaussians.py --limit 40` 报 `no overlap between pkls and v1.0-mini scenes`。
- **根因（读源码定位，非数据问题）**：官方 pkl 的场景键是 **scene token**（32 位哈希），我的过滤函数拿 scene **name**（"scene-0061"）匹配 → 零交集。pkl 本身没下错（清华云盘的就是全量 trainval 版，无 mini 专用版，本就该由脚本过滤）。
- **修复**（commit 见 tag `round-3` 更新）：
  1. 过滤同时匹配 name 和 token（读 `v1.0-mini/scene.json` 的两个字段），输出目录用可读的 scene name；失配时打印 pkl 键样例与 name/token 样例便于远程诊断；
  2. **补漏（重要）**：提取的 npz 原本没存 `ego2global` 位姿和 `timestamp`——belief memory 的静态 Gaussian 自车对齐（GaussianWorld 式 predict 步）必需。现每帧随存 ego2global(4×4)、timestamp、sample_token（token 同时是与 Occ3D gts 目录对齐的主键）。
- **教训入账**：远程执行前，凡涉及外部数据格式假设（键型、路径、单位），必须先向执行端要 3 行样例数据核对——本次损失一个往返。
- **续(同日第二个卡点)**：过滤修复后提取进到前向，`GaussianLifterV2.forward` 第 194 行 `metas["occ_label"]` KeyError。读源码定位：occ_label/occ_cam_mask 仅用于构造 `pixel_gt`（PixelDistributionLoss 的训练目标，只作为返回值，**不影响 Gaussian 表示**）。两个候选修法：(a) 作者自带的 `benchmarking=True` 逃生门——但它同时把最远点采样切成分块近似 + 随机置换，偏离官方推理路径且非确定，**弃用**；(b) 注入 dummy occ_label（全 empty_label=17，[1,200,200,16]）+ dummy mask——保持精确官方路径，**采用**。此发现同时确认：官方 eval 即使纯推理也强制加载 occ GT，进一步佐证"绕过官方评估管线"决策正确。
- **续（第三个卡点）**：`NameError: farthest_point_sampling`。根因：**pointops 是 GaussianFormer-2 的隐藏依赖**——installation.md 完全没列，lifter 里的 import 被 try/except 吞掉，直到前向才炸（官方 issue #47/#53 多人踩坑）。函数来自 point-transformer 的 pointops 库（原名 `furthestsampling`，三参 offset 式签名与调用处匹配）。修复：① ROUND_03 §1.5 用社区修好 THC 编译问题的 xieyuser/pointops fork 安装（`TORCH_CUDA_ARCH_LIST=8.6`）；② 提取脚本在 `import model` 前自动做名字别名 shim（`pointops.farthest_point_sampling = furthestsampling`），不依赖手改第三方 `__init__.py`。**复现 checklist 入账**：GaussianFormer-2 完整依赖 = 文档 4 算子 + spconv + timm + **pointops** + **init.pth**（两者均在 issue 里而非文档里）。
