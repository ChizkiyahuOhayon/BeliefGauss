# BeliefGauss（v3.1）：CVPR 2027 研究计划

**工作名**：BeliefGauss / BG-Mem
**候选标题**：
1. *Driving by Belief: Calibrated Probabilistic Gaussian Memory for Uncertainty-Aware End-to-End Driving*
2. *Know the Unknown: Belief-State Gaussian World Models for Risk-Aware Planning*
3. *BeliefGauss: Persistent Probabilistic Scene Memory with Closed-Form Risk Propagation for Autonomous Driving*

**日期**：2026-07-09　**硬件**：8×RTX 3090（24GB, Ampere, bf16 可用）　**目标**：CVPR 2027（预计截稿 ≈ 2026 年 11 月上中旬，以官网为准；CVPR 2026 为 11/6 摘要 + 11/13 正文）

---

## 0. 一页结论

**对 v1（Spatial4D）的判定**：可行性高、新颖性不足。"BEV lifting + 4D PE + sparse token + adapter" 每个组件都有成熟公开实现，审稿人会定性为 engineering integration，Borderline Reject 概率极高。另外 v1 伪代码里 `nn.Parameter(torch.randn(1, embed_dim, 4))` 这种 "4D PE" 在数学上是无效的，说明计划还停留在草图层。**不建议按 v1 投。**

**对 v2（4D-GaussianMem）的判定**：直觉正确（要有一个明确的数学对象），但**核心 novelty 声明已被占领**：
- "3D Gaussian 作为驾驶场景表示" = GaussianFormer / GaussianFormer-2（ECCV'24 / CVPR'25）
- "Gaussian 场随时间演化的世界模型" = GaussianWorld（CVPR 2025）
- "Gaussian-centric 端到端驾驶 + Gaussian flow 预测未来 + 规划" = GaussianAD（2024.12）
- "连续 4D Gaussian primitives + 占据预报 + 运动规划统一框架" = **GEM（arXiv 2605.17682, 2026 年 5 月）**——这篇几乎就是 v2 的标题本身。

同时"不确定性感知规划"这条线也已有先占者：SUPER-AD（BEV 像素级 aleatoric 不确定性 → 可行驶图 → 规划, 2025.11）、UncAD（在线地图不确定性, ICRA'25）、Risk-aware World Model MPC（2026.2）、校准 Gaussian 轨迹预测 + 不确定性感知 MPC（2026.3）。

**v2 还有两处数学硬伤**（写进论文会被直接抓住）：
1. $D_{KL}(\mathcal{N}_i^t \| \mathcal{N}_i^{t+1})$ 没有定义 Gaussian 间的对应关系（$N_t \neq N_{t+1}$，且 index $i$ 在两帧间无意义），loss 不 well-defined；
2. $\text{trace}(\Sigma)$ 惩罚会把不确定性直接压塌为 0（uncertainty collapse），得到的是"自信"而不是"校准"。正确目标是 **calibration**（NLL / coverage），不是最小化方差。

**v3 的定位（本计划）**：不再声称"提出新表示"（这条路已死），而是声称——

> **驾驶世界模型应当维护一个『带校准不确定性的持久场景信念（belief state）』。我们把驾驶建模为 POMDP，提出 BeliefGauss：以概率 Gaussian primitives 为载体的可微贝叶斯记忆（predict–associate–update），其信念协方差在遮挡中增长、在再观测时收缩，并可通过闭式公式（$\Sigma_{\text{eff}} = \Sigma_{\text{shape}} + P$）膨胀为规划风险场，使"校准的不确定性"直接转化为"更安全的轨迹"。**

三个贡献（每个都要有独立消融支撑）：
- **C1 持久概率记忆**：区分"形状协方差"（物体占多大空间, aleatoric）与"信念协方差 P"（我们对它位置/速度的确定程度, epistemic）——这是 v2 及 GaussianFormer-2 都混淆的概念；给出可微的 Kalman 结构 predict–associate–update，含可见性感知的 birth/death（被遮挡→保留且 P 膨胀；应见而未见→存在概率衰减）。
- **C2 校准优先的训练目标**：innovation-NLL + 闭式 Wasserstein-2 + χ² coverage 校准正则，替代 v2 的病态 loss；报告 NLL / ECE / AUSE / coverage 曲线。
- **C3 风险的闭式传播 + 即插即用**：belief → 风险场 → 规划（训练时 risk loss，推理时对 diffusion planner 的候选轨迹做 risk-guided 重排/引导）；以 belief tokens 形式接入两个 host（自建轻量栈 + DriveLaW-Act），证明可迁移性。

**两个 go/no-go gate**（见 §6）：W4 末验证"记忆+校准"在遮挡子集上成立；W8 末验证"不确定性→规划安全"成立。任一失败即启动 Plan B/C，不恋战。

---

## 0.5 v3.1 审查修订（2026-07-10，联网核实后）

**核实结果**：GEM（2605.17682）确认存在且确认无不确定性建模；DriveLaW 代码确认已开源（github.com/xiaomi-research/drivelaw，arXiv 2512.23421）；未发现"primitive/instance 级校准信念记忆"的直接撞车工作。**v3 的竞争判断成立，方向批准。** 但审查发现四处必须修订：

### 修订 1（最关键）：粒度错配 → 两层信念设计

v3 §2 把 Kalman 信念挂在每个 Gaussian primitive 上，这在数学上不适定：GaussianFormer-2 的 6400 个 Gaussian 是铺满场景的图元，**跨帧没有身份**（一辆车由数十个图元覆盖，帧间图元集合任意重排），"图元 i 的真实位置"无 GT 可言；而 coverage 评测（§4.2c）需要的是**物体级** GT 中心。若按 v3 原文实现，关联无意义、校准评测无对象，且 6400×N 的 Sinkhorn 代价不可承受。

**修订为两层**：
- **图元层（形状）**：GaussianFormer-2 的 N≈6400 个 Gaussian，携带 $S_i$（形状协方差）、语义、特征。静态图元只做 ego 位姿对齐（GaussianWorld 式）。
- **实例槽层（信念）**：$K \approx 64$–$128$ 个动态实例槽，每个槽 $k$ 携带状态 $\hat{x}_k = (\mu_k, v_k) \in \mathbb{R}^6$、信念协方差 $P_k \in \mathbb{S}^6_+$、存在概率 $\alpha_k$、特征 $h_k$，以及对动态图元的软隶属 $w_{ik}$（图元随所属槽刚性平移）。predict–associate–update 全部发生在**槽层**（Sinkhorn 只有 K×M，便宜且稳定）。
- **风险公式相应修正**：$\Sigma^{\text{eff}}_i = S_i + P^{pos}_{\text{slot}(i)}$（动态图元继承所属槽的位置信念边际；静态图元用小的学习先验）。
- **评测落地**：coverage / 定位误差在**槽 vs nuScenes GT box 中心**上定义——完全适定。
- **对"这不就是 MOT"的回答升级**：槽不是检测框——它锚定一组可渲染的图元，occupancy forecast 由"槽状态位移 + 图元 splatting"闭式产生，且全程可微传到规划风险；经典 KF 消融（消融 3）保留。

### 修订 2：NLL 监督来源拆分（避免自指）

v3 的 innovation-NLL 只对"预测 vs 编码器提案"计算——提案本身是模型输出，校准会自指。拆成两项：
1. **在线 NLL**（预测槽 vs 编码器实例提案）：训练 $R_\theta$（观测噪声头），刻画在线滤波行为；
2. **前瞻 NLL**（predict 分支 vs **GT 未来 box 中心/速度**，nuScenes 2Hz 标注直接可用）：训练 $F, d_\theta, Q_\theta$；**χ² 校准正则只作用于这一项**（对 GT 校准才是真校准）。

关联训练用 nuScenes **GT tracking ID 做 teacher forcing**（训练前期强制正确匹配，退火到学习匹配），砍掉关联学不出来导致整个 pipeline 无梯度的冷启动风险。

### 修订 3：新竞品入列（2026-07-10 检索发现）

- **OWMDrive**（2606.30421, 2026.6）：占据世界模型 + 未来 rollout 提升遮挡/意外场景下规划可靠性。**确定性生成式**，无信念/校准/闭式风险——与我们互补，但 related work 必须正面处理，遮挡叙事措辞避免与其正面撞车（我们的差异：显式概率信念 + 校准证据，非"多看几步 rollout"）。W1 精读。
- **Mimir**（2512.07130）：目标点级 Laplace 不确定性引导扩散规划。层级不同（goal-level vs scene-level），引用即可。

### 修订 4：远程 GPU 协作纪律（见新增 §3.5）

本项目开发者无本地 GPU，代码经 GitHub 交给 8 卡服务器执行，**每轮往返成本 ≈ 1 天**。工程纪律因此升级为一等公民：所有模块必须带 CPU 单测；每个实验 = 一个 config + 一条命令 + 自动打包的结果产物；GPU 轮次只用于"CPU 上无法回答的问题"。

---

## 1. 竞争格局审计（2026-07 核实）

| 工作 | 时间/venue | 做了什么 | 占了 v2 的哪一块 | v3 与它的差异 |
|---|---|---|---|---|
| GaussianFormer / -2 | ECCV'24 / CVPR'25 | 稀疏语义 Gaussian 做 occupancy；-2 把 Gaussian 解释为邻域被占据的概率叠加 | "Gaussian 表示 + 概率解释" | 它的"概率"是占据范围（形状/aleatoric）；我们在其上加**状态信念 P（epistemic）**并做时间贝叶斯传播——正交且可叠加，直接以它为 encoder |
| GaussianWorld | CVPR 2025 | 把 occupancy 预测重构为 Gaussian 空间中的 4D forecasting，显式利用场景演化先验（静态对齐/动态运动/新区域补全） | "Dynamic evolution" | 它是确定性、短窗、streaming；无信念/校准/遮挡持久，无规划风险传播 |
| GaussianAD | 2024.12 arXiv | Gaussian-centric 端到端驾驶：Gaussian flow 预测未来 + 规划 | "Gaussian + planning" | 同上，确定性；我们的贡献在 belief 与 risk，而非表示本身 |
| **GEM** | **2026.5 arXiv** | 连续 4D Gaussian primitives，非自回归地在任意未来时刻查询，统一 forecasting + planning | **几乎 = v2 全部** | GEM 无不确定性、无记忆持久、无校准。**必须正面引用并对比**；若其代码放出，可把 BeliefGauss 套在 GEM 上作为额外实验（化敌为证据） |
| OccWorld / Drive-OccWorld / DOME 等 | '24–'25 | 占据世界模型 forecasting/规划 | 世界模型基线 | 主表对比对象 |
| SUPER-AD | 2025.11 | 相机-only E2E，BEV 像素级 aleatoric 不确定性 → 可行驶图 → 规划 | "不确定性→规划" | 它是**稠密 BEV、单帧、仅 aleatoric**；我们是 primitive 级、随时间传播、aleatoric/epistemic 分解、闭式风险膨胀、报告校准指标 |
| UncAD | ICRA 2025 | 在线地图不确定性用于安全 E2E | 地图不确定性 | 我们针对动态场景 primitives，且带时间信念传播 |
| Risk-aware WM-MPC | 2026.2 arXiv | 风险感知世界模型预测控制 | "risk-aware" 措辞 | W1 精读，明确差异后写入 related work |
| 校准 Gaussian 轨迹预测 | 2026.3 arXiv | KDE/χ² 校准 loss + 不确定性感知 MPC | 校准方法论 | 它作用于 agent 轨迹输出层；我们作用于**场景表示层**且随记忆传播——引用其校准思想并致谢 |
| DriveLaW | **CVPR 2026**（HUST+小米，代码/权重 2026.3 已放出） | 视频生成与规划统一的潜空间世界模型（DriveLaW-Video + DriveLaW-Act 扩散规划器），NAVSIM 纪录 | 我们的 Host-B | 冻结 Video DiT，仅给 Act 注入 belief tokens——3090 可训 |
| DrivePI | **CVPR 2026** | 0.5B MLLM 统一输出 occupancy+flow+planning+文本（LiDAR+相机） | 说明"空间增强 VLA"赛道已极度拥挤 | 备选 Host-C（不确定性问答）；不作为主战场 |
| UniDriveVLA | 2026.4（小米） | MoT 解耦感知与语义，缓解 3D 注入损害推理的问题 | 同上 | 佐证：单纯"给 VLA 加 3D feature"没有故事可讲了 |
| StreamPETR / Sparse4D / EmbodiedOcc（室内 Gaussian 在线记忆） | '23–'24 | 各类时序记忆 | "memory" 措辞 | 它们是 query/特征记忆或室内静态场景；我们是驾驶动态场景的**概率信念**记忆。EmbodiedOcc 需 W1 精读核对 |

**结论：仍然空着的组合**（v3 的落点）= ① primitive 级、可校准、随时间传播的 belief；② 遮挡持久 + 再观测贝叶斯融合；③ belief→风险的闭式传播并与安全指标挂钩；④ 跨 host 的即插即用验证。任何单独一条都不够，四条合起来 + 硬消融才构成一篇 CVPR。

---

## 2. 方法（修正后的数学，可直接改写为论文 §3）

> **v3.1 注**：本节公式按"图元级信念"书写；实施时按 §0.5 修订 1 的两层设计落地——信念状态 $(\hat{x}, P, \alpha)$ 挂在**实例槽**上，$S_i$ 留在图元上，$\Sigma^{\text{eff}}_i = S_i + P^{pos}_{\text{slot}(i)}$。下文的 "$g_i$" 在槽层解读为槽及其成员图元集合。

### 2.1 表示：把"形状"与"信念"分开（核心概念修正）

场景信念在时刻 $t$ 定义为

$$
\mathcal{B}_t = \{ g_i \}_{i=1}^{N_t},\qquad
g_i = \big(\underbrace{\mu_i \in \mathbb{R}^3,\; v_i \in \mathbb{R}^3}_{\text{状态均值 } \hat{x}_i},\;
\underbrace{P_i \in \mathbb{S}^{6}_{+}}_{\text{信念协方差}},\;
\underbrace{S_i = R_i\,\mathrm{diag}(s_i)^2 R_i^\top}_{\text{形状协方差(aleatoric)}},\;
f_i \in \mathbb{R}^{d},\; \alpha_i \in [0,1] \big)
$$

- $S_i$：物体**占据多大空间**（3DGS 意义的 scale/rotation）——这是 GaussianFormer-2 概率叠加所建模的量；
- $P_i$：我们**对它在哪、往哪动有多确定**（POMDP 意义的 belief covariance）——遮挡中增长、再观测时收缩；
- $\alpha_i$：存在概率（Bernoulli）。

v2 用一个 $\Sigma$ 同时表示两者，导致"物体大 = 不确定"的荒谬耦合；这个区分本身就是论文里值得一段的概念贡献。

### 2.2 Predict（先验演化，ego 条件化）

$$
\hat{x}_i^{t+1|t} = F\,\hat{x}_i^{t} + d_\theta\!\big(g_i, \text{ctx}_t, \text{ego}_{t:t+k}\big),\qquad
P_i^{t+1|t} = F P_i^{t} F^\top + Q_\theta(g_i, \text{ctx}_t)
$$

其中 $F$ 为常速度模型的转移矩阵，$d_\theta$ 是轻量 Transformer 输出的残差（建模交互与非线性运动），$Q_\theta \succeq 0$ 是**学习的异方差过程噪声**（对角 + 低秩，softplus 保证正定）。静态 Gaussian 只做 ego 位姿对齐（对齐 GaussianWorld 的先验分解，引用之）。

### 2.3 可见性推理与关联

用当前 Gaussian 场自身渲染期望深度图 → 每个 $g_i$ 得到可见性 $o_i \in [0,1]$（被更近的 Gaussian 遮住则 $o_i$ 低）。图像分支（GaussianFormer-2 式 encoder）产出观测提案 $z_j$ 及**学习的观测噪声** $R_j$（由图像证据质量预测：距离、截断、光照 token 等）。关联用 Mahalanobis 距离门控 + Sinkhorn（训练时可微）/ Hungarian（推理时）：

$$
d_{ij}^2 = (z_j - H\hat{x}_i)^\top (H P_i H^\top + R_j)^{-1} (z_j - H\hat{x}_i)
$$

### 2.4 Update 与 birth/death（遮挡持久机制）

- **匹配上的**：Kalman 更新 $K = P H^\top (HPH^\top + R)^{-1}$；或"Kalman-初始化的门控更新"（可学习，但用 Kalman 解正则/初始化）——两者做消融。特征 $f_i$ 用 GRU 式门控融合。
- **未匹配、但 $o_i$ 低（被遮挡）**：**保留**，$P_i$ 按 $Q_\theta$ 继续膨胀，$\alpha_i$ 不惩罚——这就是"记忆里那辆被公交车挡住的车还在，只是我们越来越不确定它在哪"。
- **未匹配、且 $o_i$ 高（应见未见）**：$\alpha_i \leftarrow \gamma\,\alpha_i$ 衰减，低于阈值删除。
- **未匹配的观测**：birth 新 Gaussian，$P$ 初始化为 $R_j$ 放大。

### 2.5 训练目标（对 v2 的逐条修正）

**不要用**：$D_{KL}(t\|t+1)$（无对应关系）；$\text{trace}(\Sigma)$ 惩罚（塌缩）。

**用**：
1. **Innovation-NLL**（端到端训练 $F, d_\theta, Q_\theta, R_\theta$）：对匹配对
$$
\mathcal{L}_{\text{NLL}} = \tfrac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}} \tfrac{1}{2}\Big[ d_{ij}^2 + \log\det(HP_iH^\top + R_j) \Big]
$$
NLL 自动在"准"与"不过度自信"间权衡——这正是 v2 想要而没写对的东西。
2. **闭式 Wasserstein-2**（可选辅助，匹配对上）：$W_2^2 = \|\mu_1-\mu_2\|^2 + \mathrm{tr}\big(\Sigma_1+\Sigma_2-2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2}\big)$。
3. **渲染占据监督**（当前帧 + 未来 K 帧 forecast）：把 $\{(\mu, S, f, \alpha)\}$ splat 到体素（gsplat/GaussianFormer-2 的 rasterizer），对 Occ3D-nuScenes 做 CE + Lovász；未来帧监督 predict 分支。
4. **校准正则** $\mathcal{L}_{\text{calib}}$：匹配对的归一化 innovation $d_{ij}^2$ 应服从 $\chi^2_{\dim}$；用 batch 内经验分位数与 $\chi^2$ 分位数的差做惩罚（借鉴 2026.3 轨迹校准工作的思路，移植到场景 primitive 层——引用并声明差异）。
5. **存在监督** $\mathcal{L}_{\text{exist}}$：BCE（用可见性 + GT 占据判定正负）。
6. **风险-规划损失**：见 2.6。

### 2.6 风险的闭式传播（论文的"money equation"）

对位置信念边际 $P^{pos}_i \in \mathbb{S}^3_+$，**不确定性直接膨胀障碍物**：

$$
\boxed{\;\Sigma^{\text{eff}}_i = S_i + P^{pos}_i\;}
\qquad
\text{Risk}_t(x) = 1 - \prod_i \Big(1 - \alpha_i\, \bar{\mathcal{N}}(x;\, \mu_i^{t},\, \Sigma^{\text{eff}}_i)\Big)
$$

（$\bar{\mathcal{N}}$ 为峰值归一化的高斯核；连乘沿用 GaussianFormer-2 的概率叠加语义。）直觉：**你越不确定它在哪，它在规划里就"越大"**；而校准（§2.5-4）决定膨胀量是否恰当——校准与安全从此在一个公式里挂钩，这是整篇论文的叙事枢纽。

- **训练**：$\mathcal{L}_{\text{risk}} = \sum_{\tau \in \text{GT轨迹}} \text{Risk}(\tau) + \max(0, m - \text{Risk}(\tau^-))$（负样本 $\tau^-$ 用扰动/对手轨迹）。
- **推理**：对 host 规划器（如 DriveLaW-Act 的扩散采样）输出的 $M$ 条候选按 $\int \text{Risk}$ 重排，或作为 guidance 梯度。全程闭式、可微、便宜。

### 2.7 即插即用接口：belief tokens

每个 Gaussian → 一个 token：$[\,\mu_i;\; v_i;\; \text{eig}(P^{pos}_i);\; \text{主轴};\; \alpha_i;\; f_i\,]$ + 4D PE（正弦式作用于 $(\mu, t)$，不是 v2 那个随机参数）。Top-K（按 $\alpha \cdot \|\text{信息量}\|$）选 256–512 个 token，经 2 层 cross-attention adapter 注入 host。**host 全冻结，只训 memory + adapter**（LoRA 可选）——这是 8×3090 的生命线。

---

## 3. 系统与工程（8×3090 预算）

### 3.1 规模设定
- 图像 encoder：复用 GaussianFormer-2 官方权重（R50, 6 相机 704×256），冻结或 LoRA；
- Gaussian 数：6400（GaussianFormer-2 量级；消融 3200/12800）；
- Memory 模块：3 层 Transformer（d=256）+ Kalman 头 + 噪声头，**可训练参数约 40–80M**；
- 记忆窗：训练时截断 BPTT 至 4–8 帧（2–4s @2Hz），推理时无限流式。

### 3.2 训练配方（3090 专用纪律）
- bf16 + DeepSpeed **ZeRO-2**（3090 多为 PCIe、无全互联 NVLink，避免 ZeRO-3/模型并行的通信墙）；
- gradient checkpointing 全开；micro-batch 1/卡 × 8 卡 × accum 4 = 有效 32；
- **离线缓存 backbone 特征**跑消融（省 60%+ 时间）；主表再端到端微调；
- torch.compile 对 memory 模块（小图，收益大）。

### 3.3 远程 GPU 协作纪律（v3.1 新增，一等公民）

开发环境（macOS, CPU-only）与执行环境（国内 8×3090）分离，经 GitHub + 人工中转，**每轮往返 ≈ 1 天**。纪律：

1. **CPU 先行**：记忆模块、损失、关联、指标全部有合成数据单测/冒烟测试，在本地 CPU 通过后才允许上 GPU；GPU 轮次只回答"CPU 上无法回答的问题"（吞吐、真实数据行为、全量训练）。
2. **一实验一命令**：每个实验 = `configs/*.yaml` + `bash scripts/run_xxx.sh`，无需人工改代码；执行端只需复制粘贴。
3. **结果自动打包**：每次运行结束自动生成 `outputs/<run_name>/report.json`（指标）+ `curves/*.png`（曲线）+ `env.txt`（环境指纹）+ 尾部日志，压缩为单个 zip 传回——避免"缺了一个数再跑一轮"。
4. **确定性**：固定 seed、记录 git commit hash 进 report；执行端只跑 tagged commit。
5. **失败快报**：脚本内置前置检查（数据路径、显存、依赖版本），失败在 60 秒内报清晰错误，而不是训练 3 小时后崩。
6. **国内网络**：SETUP 文档提供 pip 清华源、HuggingFace 镜像（hf-mirror.com）、数据集分卷下载指引。

### 3.4 三个 Host（递进，不并行铺开）
- **Host-A（主战场）**：自建轻量栈 = GaussianFormer-2 encoder + BeliefGauss + 简单轨迹头。出 forecasting 主表 + nuScenes 开环规划 + 全部不确定性/遮挡实验。
- **Host-B**：DriveLaW-Act（扩散规划器）+ belief tokens 条件注入，**冻结 DriveLaW-Video**。出 NAVSIM PDMS 表，证明即插即用。W1 需核实 Act 的参数量与条件接口（代码 2026.3 已开源）。
- **Host-C（stretch，可砍）**：0.5–3B MLLM（如 Qwen2.5-VL）+ LoRA，做"不确定性感知问答"（"左前方被遮挡区域有车的概率？"）小节，只作 qualitative + 小表。

### 3.5 算力估算（诚实标注：均为估计，W1 实测后修正）

| 阶段 | 数据 | 规模 | 估计 GPU 时长（8×3090） |
|---|---|---|---|
| S0 复现 GaussianFormer-2 推理/短训 | mini | — | 1–2 天 |
| S1 记忆预训练（forecast + 校准） | trainval | 12–18 ep | **5–8 天** |
| S2 规划头 + 风险 | trainval | +6 ep | 2–3 天 |
| S3 DriveLaW-Act adapter | navtrain | — | 3–5 天 |
| 消融 ×10 | 1/4 split 或 mini | 短程 | 每组 0.5–1.5 天 |
| 合计 | | | ≈ 6–8 周 GPU 时（与写作重叠可容纳） |

---

## 4. 数据与评测协议

### 4.1 数据
- nuScenes trainval + **Occ3D-nuScenes**（关键：其**相机可见性 mask** 是遮挡实验的免费监督/评测标签）；
- NAVSIM navtrain/navtest（Host-B）；
- 可选：nuScenes-QA / DriveLM（Host-C）。
- **明确砍掉**：CARLA、Bench2Drive（v2 的评测清单在 18 周内不可能完成；NAVSIM 已足以回应"开环指标不可信"的质疑）。

### 4.2 遮挡评测子集（论文的招牌实验，W2 就要把脚本写好）
1. 用 Occ3D 可见性 mask + nuScenes 标注，筛出"目标被遮挡 ≥T 帧后重现"的片段（预计数百段）；
2. 指标：(a) 遮挡期间该目标区域的 forecast IoU；(b) 重现时刻的定位误差 vs 遮挡时长曲线；(c) **coverage：遮挡期间 GT 中心落入预测 68%/95% 信念椭球的频率**（校准的直接证据）；(d) 该子集上的规划 collision rate。
3. 基线在此子集上应显著劣化，BeliefGauss 的差距应最大——这是"记忆 + 校准"价值的最干净展示。

### 4.3 指标与主表
- **表 1**：4D occupancy forecasting（mIoU/IoU @ 0/0.5/1/2/3s）vs OccWorld、GaussianWorld、GEM（若可比）；
- **表 2**：nuScenes 开环规划 L2/collision——**双协议（ST-P3 与 UniAD 式）+ 有/无 ego-status 各报一遍**，正文明写该基准的已知缺陷并以 NAVSIM 为主证据（预防审稿人引 BEV-Planner 批评）；
- **表 3**：NAVSIM PDMS：DriveLaW-Act 原始 vs +BeliefGauss tokens vs +tokens+risk 重排；
- **表 4**：不确定性质量：NLL / ECE / coverage@68/95 / **AUSE**（sparsification）/ risk-coverage 曲线；
- **表 5**：遮挡子集（§4.2）；
- **表 6**：效率：延迟、显存、#tokens vs 稠密 BEV adapter。

---

## 5. 消融（10 组，全部在 1/4-split 上跑，主结论用全量复验 2 组）

1. 无记忆（单帧，= GaussianFormer-2 + 轨迹头）——记忆的价值；
2. 记忆但 $P \equiv 0$（确定性演化，≈ GaussianWorld 式）——**belief 的价值，最重要的一组**；
3. **经典 Kalman（固定手调噪声）替代学习噪声**——预防"这不就是 KF 吗"的审稿必问；
4. 纯学习门控更新（去 Kalman 结构）——结构先验的价值；
5. 去可见性感知 birth/death（统一衰减）——遮挡机制的价值；
6. $\mathcal{L}_{\text{NLL}} \to$ MSE——概率目标的价值；
7. 去 $\mathcal{L}_{\text{calib}}$——报告校准指标恶化幅度；
8. Evidential（NIG）头替代 $P$ 传播——回应"为何不用 evidential"；
9. 风险重排 on/off；$\Sigma^{\text{eff}}$ 不膨胀（$P$ 不入规划）——**C3 的价值**；
10. #Gaussians / BPTT 窗长 / token 数扫描；Host-A vs Host-B 增益一致性。

---

## 6. 时间线（2026-07-13 → 11 月中，18 周，双 gate）

| 周 | 日期 | 任务 | 里程碑 / Gate |
|---|---|---|---|
| W1 | 7/13– | 环境；跑通 GaussianFormer-2 与 DriveLaW 推理；精读 GEM/SUPER-AD/Risk-WM-MPC/EmbodiedOcc 并写差异备忘；核实 DriveLaW-Act 接口 | 差异备忘录定稿（related work 骨架） |
| W2–3 | | 实现 predict/associate/update + NLL + 可见性 birth-death；遮挡子集脚本；mini 上迭代 | mini 上端到端收敛 |
| **W4** | ~8/9 | **Gate-1**：mini/遮挡子集上，(a) 遮挡区 forecast 优于无记忆基线 ≥ 明显幅度；(b) coverage@68 落在 [58, 78] | 否 → 一周排障，再否 → 启动 Plan C |
| W5–6 | | S1 全量训练；表 1 初版；校准曲线 | forecasting 结果成形 |
| W7–8 | | 规划头 + 风险场；nuScenes 开环；AUSE/risk-coverage | **Gate-2**：AUSE 显著正相关 且 遮挡子集 collision ↓ |
| W9–10 | | DriveLaW-Act 集成 + NAVSIM（Plan B：换 DiffusionDrive/Transfuser 为 host） | 表 3 初版 |
| W11–12 | | 10 组消融（缓存特征加速）；Host-C 可选 | 消融全表 |
| W13–14 | | 全量复验、可视化（遮挡持久的定性图是招牌）、失败案例 | 实验冻结 |
| W15–16 | | 写作（先图后文）；按 §8 预答清单内部红队评审一轮 | 完整初稿 |
| W17–18 | 11 月上 | Buffer、补充材料、打磨、投稿 | 提交 |

每周五：arXiv 监控（关键词见 §7.1）+ 15 分钟风险复盘。

---

## 7. 风险与 Plan B/C

### 7.1 被 scoop 监控（每周）
`Gaussian world model driving`、`uncertainty occupancy forecasting`、`belief state driving`、`risk-aware end-to-end driving`、`calibrated world model`、`occlusion-aware planning`、`probabilistic scene memory`。发现近似工作 → 24h 内写差异备忘，决定"引用+差异化"还是"调转叙事重心"。

### 7.2 主要风险矩阵

| 风险 | 概率 | 缓解 |
|---|---|---|
| DriveLaW-Act 集成超预期困难 | 中 | Host-B 换 NAVSIM 上文档完善的规划器（DiffusionDrive/Transfuser）；即插即用故事不变 |
| GEM 等释出带不确定性的后续版本 | 中 | 叙事重心可整体滑向"遮挡持久记忆 + 校准评测协议"（Gate-1 已备好证据） |
| 校准做好了但规划收益小（Gate-2 失败） | 中 | 论文改叙事为"可信世界模型"：forecasting + 遮挡 + 校准分析为主，规划为辅；或转 Plan C |
| 3090 训练慢于估计 | 中高 | 降输入分辨率至 512×192 出主结论、全量只跑最终表；特征缓存 |
| 开环规划指标被质疑 | 高（必然） | §4.3 双协议 + NAVSIM 主证据 + 正文自曝其短 |

### 7.3 Plan C（保底论文，Gate-1 失败时启动）
**《Do Driving World Models Know What They Don't Know?》**：对 OccWorld / GaussianWorld / GEM / DriveLaW 做遮挡与分布外条件下的系统性校准评测（复用 §4.2 协议与已写的评测代码），附一个轻量校准修正。算力需求低一个量级，CVPR 接收难度更高但周期可控，且与主线代码 90% 复用——这就是为什么评测脚本要在 W2 写好。

---

## 8. 写作与 rebuttal 预案

**要做的 claims**：belief 与 shape 的概念区分；校准指标全套；风险闭式传播；遮挡子集上的因果证据；跨 host 一致增益。
**不要做的 claims**：不说"提出新的场景表示"（GEM/GaussianAD 在前）；不说"首个不确定性感知驾驶"（SUPER-AD/UncAD 在前）；措辞用 "the first to maintain a *calibrated, temporally-propagated* belief over scene primitives and *propagate it in closed form to planning risk*"——每个限定词都有文献支撑其必要性。

**预答审稿问题**：
1. *vs GEM？* 确定性 vs 概率信念；无遮挡持久/校准/风险传播；若代码可用，附"GEM+BeliefGauss"实验。
2. *vs SUPER-AD？* 稠密 BEV 单帧 aleatoric vs primitive 级时间传播的 aleatoric+epistemic；机制不同、指标集不同（我们有 coverage/AUSE）。
3. *这不就是 Kalman/MOT 跟踪？* 消融 3 直接回答：学习异方差噪声 + 端到端进规划 + 场景 primitive 而非检测框；经典 KF 在表中作为 baseline 存在。
4. *vs GaussianFormer-2 的"概率"？* 形状占据概率 ≠ 状态信念；我们建在它之上（encoder 复用即证明）。
5. *开环指标不可信？* 见 §4.3。
6. *为何不用 evidential？* 消融 8。
7. *增益来自哪里？* 消融 1/2/9 三级分解：记忆、belief、风险传播各自贡献。
8. *为何没有 CARLA 闭环？* NAVSIM 覆盖闭环式评测且社区正在迁移；CARLA 训练开销与本文贡献正交，列为 future work。

---

## 9. 本周行动清单（7/9–7/15）

1. 克隆并跑通 GaussianFormer-2（权重推理 + mini 短训）与 DriveLaW 推理；记录 3090 实测吞吐，修正 §3.5；
2. 精读 4 篇并各写半页差异备忘：GEM（2605.17682）、SUPER-AD（2511.22865）、Risk-aware WM-MPC（2602.23259）、校准轨迹预测（2603.10407）；顺带核对 EmbodiedOcc；
3. 下载 Occ3D-nuScenes，验证可见性 mask 读取；
4. 写遮挡子集构建脚本 v0（这份代码同时服务主线与 Plan C）;
5. 实现 §2.2–2.4 的最小版本（先不接图像，用 GT 检测模拟观测流，单卡验证 Kalman 结构与 NLL 收敛——一天内可完成的"数学正确性"冒烟测试）；
6. 核实 DriveLaW-Act 的条件注入接口与参数量；
7. 建 arXiv 周报警（§7.1 关键词）。

---

## 附录 A：v1 / v2 / v3 对照

| 维度 | v1 Spatial4D | v2 4D-GaussianMem | v3 BeliefGauss |
|---|---|---|---|
| 核心对象 | "更好的 feature" | Dynamic Gaussian Field（已被 GaussianWorld/GaussianAD/GEM 占领） | **校准的持久信念** $\mathcal{B}_t$（形状/信念分离） |
| 数学 | 无效 4D PE | KL 无对应关系、trace 塌缩 | innovation-NLL + χ² 校准 + 闭式 $W_2$ + 闭式风险 |
| 与规划的关系 | 无 | 泛泛的 uncertainty-weighted | $\Sigma^{\text{eff}} = S + P$ 闭式膨胀 + 可微重排 |
| 招牌实验 | 无 | 常规消融 | 遮挡子集 + coverage/AUSE + 跨 host |
| 评测面 | 未定 | 5 个基准（不可行） | nuScenes(+Occ3D) + NAVSIM，聚焦 |
| 8×3090 可行性 | 高 | 中（未做预算） | 高（host 冻结、可训参数 <100M、有预算表） |
| 被拒主因预测 | engineering | "novelty 已被 GEM 等占领" | 需靠 Gate-1/2 证据说话 |

## 附录 B：W1 必读清单
GaussianFormer / GaussianFormer-2、GaussianWorld、GaussianAD、**GEM**、OccWorld、Drive-OccWorld、**OWMDrive（2606.30421，v3.1 新增）**、SUPER-AD、UncAD、Risk-aware WM-MPC（2602.23259）、校准 Gaussian 轨迹预测（2603.10407）、Mimir（2512.07130，v3.1 新增）、EmbodiedOcc、StreamPETR、DriveLaW、DrivePI、UniDriveVLA、BEV-Planner（开环指标批评）、NAVSIM（v1/v2）、深度学习 Kalman 系（KalmanNet、Backprop-KF）。

## 附录 C：命名与标题备选
方法名：BeliefGauss / BG-Mem / GaussBelief。若后续发现名字冲突，备选 PBM（Probabilistic Belief Memory）。标题倾向候选 1（"Driving by Belief..."），副标题里必须同时出现 calibrated 与 risk，让 AC 一眼看到差异化。
