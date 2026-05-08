# Spectrum-RL: Three-Layer HRL for Integrated EW Radar Resource Scheduling

用强化学习实现相控阵雷达的侦察、干扰、探测、通信一体化控制，提升作战效能。

## Background

相控阵雷达 (Phased Array Radar) 需要在有限的频谱、功率、波束、时间、波形、编码六维资源约束下，同时执行四项核心任务：

| 功能 | 英文 | 目标 |
|------|------|------|
| **探测** | Detection | 在噪声和干扰中发现目标，最大化检测概率 Pd |
| **侦察** | Reconnaissance | 感知频谱占用，识别敌方发射源位置和参数 |
| **干扰** | Jamming | 对敌方雷达/通信链路释放干扰能量，降低其效能 |
| **通信** | Communication | 维持与友方单位的数据链，保证信道容量 |

传统方法使用固定资源分配方案（如均匀分配、按优先级静态划分），无法适应动态变化的战场环境和敌方行为。本项目将四功能一体化资源调度建模为**部分可观测马尔可夫决策过程 (POMDP)**，使用**三层分层强化学习 (HRL)** 进行端到端优化。

### 为什么需要 HRL？

单体 (flat) RL 直接在 136 维连续动作空间上优化，面临严重的维度灾难和信用分配问题。三层 HRL 将决策分解为：

- **Strategic (战略层)**：低频次、粗粒度——选择当前的功能激活模式（Option）
- **Tactical (战术层)**：中频次——在选定的 Option 下分配 24 维资源预算
- **Executive (执行层)**：高频次——将资源预算转化为 136 维具体参数

这种分解模仿军事指挥链，且大幅降低了每层的搜索空间。

---

## Environment (雷达环境)

### Observation Space (107 维)

| 维度 | 组件 | 说明 |
|------|------|------|
| 0–63 | `spectrum_occupancy` | 64 个频道的信号功率 (dB) |
| 64–79 | `threat_levels` | 8 个波束方向的威胁等级（方位+俯仰各 8） |
| 80–95 | `target_params` | 4 个目标的状态（距离/速度/角度/RCS）× 4 |
| 96–99 | `function_status` | 4 个功能（探/侦/干/通）的历史成功率 |
| 100–105 | `resource_usage` | 6 维资源使用率（频/时/空/功/波形熵/编码熵） |
| 106 | `time_step` | 当前时间步（归一化） |

### Action Space (136 维)

| 维度 | 组件 | 说明 |
|------|------|------|
| 0–63 | `freq_alloc` | 64 频道功率分配（softmax 归一化） |
| 64–79 | `beam_direction` | 8 波束方位角 + 8 波束俯仰角 |
| 80–83 | `power_levels` | 4 功能功率分配（softmax → dBm） |
| 84–115 | `waveform_select` | 4 功能 × 8 波形类型 logits |
| 116–119 | `time_alloc` | 4 功能时间片分配 |
| 120–135 | `code_select` | 4 功能 × 4 编码类型 logits |

### Reward Function

奖励由课程学习 (Curriculum Learning) 动态加权，随训练阶段逐步引入更多功能：

$$r = w_0 \cdot r_{\text{detect}} + w_1 \cdot r_{\text{recon}} + w_2 \cdot r_{\text{jam}} + w_3 \cdot r_{\text{comm}} + 0.1 \cdot r_{\text{balance}} - r_{\text{power\_penalty}}$$

**各分量详解：**

| 分量 | 计算方式 | 取值范围 |
|------|----------|----------|
| $r_{\text{detect}}$ | 成功检测的目标数 / 总目标数 | [0, 1] |
| $r_{\text{recon}}$ | 频谱感知覆盖率 (TP / (TP+FN)) | [0, 1] |
| $r_{\text{jam}}$ | 有效干扰敌方数 / 总敌对数 (JSR > 3dB) | [0, 1] |
| $r_{\text{comm}}$ | $\tanh(\text{data\_rate} / 100)$，数据率越高越好 | [0, ~1] |
| $r_{\text{balance}}$ | $1 - \sigma$(四个功能归一化得分)，鼓励均衡 | [0, 1] |
| $r_{\text{power\_penalty}}$ | 总功率超过 $P_{\max}$ 时的线性惩罚 | ≤ 0 |

**课程权重 (4 阶段)：**

| Stage | 名称 | $w_{\text{detect}}$ | $w_{\text{recon}}$ | $w_{\text{jam}}$ | $w_{\text{comm}}$ | 最大奖励/步 |
|-------|------|---------------------|--------------------|------------------|--------------------|-------------|
| 0 | detect_only | 1.0 | 0.0 | 0.0 | 0.2 | ~1.3 |
| 1 | detect_jam | 0.6 | 0.0 | 0.4 | 0.0 | ~1.1 |
| 2 | detect_recon_jam | 0.4 | 0.3 | 0.3 | 0.0 | ~1.1 |
| 3 | full_integrated | 0.25 | 0.25 | 0.25 | 0.25 | ~1.1 |

### 物理模型

- **雷达方程**：单脉冲 SNR 通过标准雷达距离方程计算，包含路径损耗、天线增益、RCS
- **检测概率**：Albersheim 近似 (Swerling 1 目标模型)，$P_{fa}=10^{-6}$
- **信道容量**：Shannon 公式 $\log_2(1+\text{SINR})$ bps/Hz
- **传播模型**：自由空间路径损耗 + 时间相关 Rayleigh 衰落
- **敌方模型**：Markov 意图转移（搜索/规避/火控制导/静默侦察）× 自适应功率控制

### 仿真参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_freq_channels` | 64 | 频率通道数 |
| `num_beams` | 8 | 同时波束数 |
| `max_power_db` | 60 dBm | 总功率上限 |
| `num_targets` | 4 | 目标数量 |
| `num_opponents` | 2 | 敌方数量 |
| `max_steps` | 200 | 每 episode 最大步数 |

---

## Network Architecture (网络结构)

### Strategic Network (战略网络)

```
obs (107)
  └─ Linear(107, 256) + ReLU
       └─ Linear(256, 256) + ReLU
            ├─ option_head:  Linear(256, 8)       → Option 离散策略 π_Ω(ω|s)
            ├─ term_head:    Linear(256+8, 1)     → 终止概率 β(s, ω)
            └─ value_head:   Linear(256, 1)       → 状态价值 V(s)
```

- **输入**：obs (107 维)
- **输出**：8 个离散 Option（对应 8 种功能激活模式）
- **Option 持续时间**：4–32 步，或由终止概率 β 触发提前切换
- **激活函数**：ReLU
- **参数**：~100K

### Tactical Network (战术网络) — PPO Policy

```
obs (107) + option_onehot (8) = 115 维
  └─ Linear(115, 128) + Tanh
       └─ Linear(128, 128) + Tanh
            ├─ actor_mean:   Linear(128, 24)           → 资源分配均值
            ├─ actor_logstd: Parameter(1, 24)           → 各维度独立标准差
            └─ critic:       Linear(128, 1)             → V(s, ω)
```

- **输入**：obs ⊕ option_onehot (115 维)
- **输出**：24 维资源分配（4 功能 × 6 资源维度），服从对角高斯分布
- **激活函数**：Tanh（CleanRL 标准）
- **初始化**：正交初始化，actor 输出层 std=0.01，critic 输出层 std=1.0
- **参数**：~66K

### Executive Network (执行网络) — PPO Policy

```
obs (107) + resource_alloc (24) = 131 维
  └─ Linear(131, 128) + Tanh
       └─ Linear(128, 128) + Tanh
            ├─ actor_mean:   Linear(128, 136)           → 动作均值
            ├─ actor_logstd: Parameter(1, 136)          → 各维度独立标准差
            └─ critic:       Linear(128, 1)             → V(s, resource)
```

- **输入**：obs ⊕ resource_alloc (131 维)
- **输出**：136 维具体执行参数，服从对角高斯分布
- **参数**：~85K

**总参数量：~250K**

---

## Training Algorithm (训练算法)

### 整体架构

```
Strategic (离散 Option)     → REINFORCE + Value Baseline + Entropy
Tactical   (24 维连续)      → PPO (CleanRL, Clipped Objective)
Executive  (136 维连续)     → PPO (CleanRL, Clipped Objective)
```

### PPO 算法细节 (Tactical + Executive)

基于 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的 `ppo_continuous_action.py` 实现：

1. **Rollout 收集**：On-policy 采样 `rollout_length=512` 步（含 +200 安全余量），存储 (state, action, log_prob, value, reward, done)
2. **GAE 优势估计**：`λ=0.95, γ=0.99`，末尾状态用当前 Critic 进行 Bootstrap
3. **PPO 更新**（每轮 rollout 满时触发）：

   - **Policy Loss (Clipped)**：
     $$L^{\text{CLIP}}(\theta) = \mathbb{E}[\max(-A \cdot r(\theta), -A \cdot \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon))]$$
     其中 $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$，$\epsilon=0.2$

   - **Value Loss (Clipped)**：
     $$L^{\text{VF}} = \frac{1}{2}\mathbb{E}[\max((V_\theta - R)^2, (V_{\text{old}} + \text{clip}(V_\theta - V_{\text{old}}, -\epsilon, +\epsilon) - R)^2)]$$

   - **Entropy Bonus**：系数 0.0（标准 CleanRL 默认不加熵正则）

4. **优化参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `update_epochs` | 10 | 每轮 rollout 的更新轮数 |
| `minibatch_size` | $\max(B/32, 64)$ | 32 个 mini-batch |
| `learning_rate` | 3e-4 | Adam 优化器，$\epsilon$=1e-5 |
| `max_grad_norm` | 0.5 | 梯度裁剪 |
| `norm_adv` | True | Mini-batch 内优势归一化 |
| `clip_vloss` | True | 启用 Clipped Value Loss |
| `target_kl` | None | 不启用早停 |

### Strategic 算法细节 (REINFORCE)

1. **Option 级回报**：Option 持续期间的累积折扣奖励，在 Option 终止时计算并写入持久化 Buffer
2. **从 Buffer 采样训练**：`batch_size=256`，从 `HRLReplayBuffer (capacity=1M)` 中采样
3. **损失函数**：
   $$L_{\text{policy}} = -\mathbb{E}[\log\pi(\omega|s) \cdot (R_{\text{norm}} - V(s))]$$
   $$L_{\text{value}} = \text{MSE}(V(s), R_{\text{norm}})$$
   $$L_{\text{total}} = L_{\text{policy}} + 0.5L_{\text{value}} - 0.01 \cdot H(\pi)$$
4. 梯度裁剪：`max_norm=10.0`

### 课程学习 (Curriculum Learning)

4 阶段渐进式训练，基于**实际奖励 / 理论最大奖励**的比率自动推进：

- **成功率阈值**：65%
- **每阶段最少步数**：50,000 步
- **检测窗口**：最近 1000 步

### 超参数总览

| 类别 | 参数 | 值 |
|------|------|-----|
| **PPO** | $\gamma$ | 0.99 |
| | $\lambda$ (GAE) | 0.95 |
| | $\epsilon$ (clip) | 0.2 |
| | Rollout Length | 512 |
| | Update Epochs | 10 |
| | Hidden Dim | 128 |
| **Strategic** | Hidden Dim | 256 |
| | Num Options | 8 |
| | Option Duration | 4–32 |
| **Optimization** | LR (all layers) | 3e-4 |
| | Buffer Capacity | 1,000,000 |
| | Batch Size | 256 |
| | Reward Scale | 2.0 |

---

## Usage

```bash
# 训练（默认 200k 步）
python -m artifacts.main --mode train --device cuda --seed 42

# 评估
python -m artifacts.main --mode evaluate --checkpoint models/hrl_step200000.pt

# 运行基线
python -m artifacts.main --mode baselines

# 完整实验流程（基线 + 训练 + 消融 + 可视化）
python -m artifacts.main --mode full_experiment --total_steps 200000

# 多种子扫描
python run_experiments.py --seeds 42,123,456,789,1024 --total_steps 50000
```

训练过程通过 [W&B](https://wandb.ai) 实时监控，日志和模型自动同步到 wandb 服务器。

---

## Project Structure

```
artifacts/
├── config.py              # 实验配置 (dataclasses)
├── main.py                # CLI 入口 (train/evaluate/baselines/full_experiment)
├── run_experiments.py     # 多种子网格扫描
├── run_experiments.sh     # Bash 版扫描脚本
├── agents/
│   ├── networks.py        # Strategic/Tactical/Executive/Attention/OpponentEncoder
│   └── buffers.py         # HRLReplayBuffer, RolloutBuffer (GAE)
├── env/
│   ├── radar_env.py       # IntegratedRadarEnv (Gymnasium)
│   ├── channel_models.py  # RadarChannel, PropagationModel
│   └── target_models.py   # Target, Opponent (Markov intents)
├── training/
│   ├── trainer.py         # HRLTrainer (主训练循环)
│   ├── ppo_trainer.py     # PPOTrainer + PPOPolicy (CleanRL)
│   ├── sac_trainer.py     # SACTrainer (legacy, 未使用)
│   ├── cleanrl_ppo.py     # CleanRL 参考实现
│   └── curriculum.py      # CurriculumScheduler
├── baselines/
│   ├── random_baseline.py
│   ├── fixed_policy.py    # 4 种固定策略 (uniform/balanced/detect_heavy/jam_heavy)
│   └── independent_rl.py  # Flat RL (无层级)
└── eval/
    ├── metrics.py          # Evaluator, 指标计算
    └── visualize.py        # Visualizer, 对比图/训练曲线/消融
```

## Known Issues & Next Steps

1. **奖励区分度不足**：当前功率预算 (60dBm) 过大，随机策略即可获得接近理论上限的奖励，导致 PPO 无法学到有效策略。需降低 `max_power_db`，增加目标/敌方数量以制造真实的资源竞争。
2. **课程推进困难**：Stage 0 的检测任务过于简单，成功率始终接近 100%，但阈值条件需要 `连续窗口均值 > 65%`，推进逻辑有改进空间。
3. **136 维动作空间的 PPO**：对角高斯假设可能不适合具有强耦合结构的雷达参数空间，考虑使用 Normalizing Flow 或更复杂的分布族。
