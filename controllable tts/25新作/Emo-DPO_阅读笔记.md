# Emo-DPO: 将直接偏好优化（DPO）引入情感语音合成以提升情感可控性

- **来源**: arXiv:2409.10157v1
- **作者**: Xiaoxue Gao, Chen Zhang, Yiming Chen, Huayun Zhang, Nancy F. Chen
- **发表**: arXiv preprint, 2024

---

## 1. 研究背景与问题

当前情感文本到语音（emotional TTS）模型主要依赖监督学习，在==捕捉情感表达的细微差异（nuanced emotional details）==方面存在明显瓶颈。具体而言，现有方法通常仅从文本和期望情感标签出发生成语音，模型仅学习"生成正确输出"，而缺乏对"为何该输出更优"的深入理解。这导致合成语音在情感可控性（controllability）和表现力（expressiveness）上受限，难以区分同一语义内容下不同情感之间的微妙差别。

本文的核心假设是：==如果让模型直接从成对人类偏好数据中学习区分"更优"与"较差"的情感表达，就能更好地捕捉情感表达的细微差异。== 基于此，作者将大语言模型（LLM）领域兴起的直接偏好优化（Direct Preference Optimization, DPO）引入情感 TTS，提出 Emo-DPO 方法。该方法结合指令微调（instruction tuning）与 DPO 训练，在基于 LLM 的 TTS 神经架构中整合情感感知的 LLM-TTS 模块，通过成对偏好数据优化模型对情感语音 token 的生成概率分布。

---

## 2. 方法概述

### 2.1 整体架构

![Emo-DPO Overview](images/Emo-DPO_overview.png)

Emo-DPO 包含三个核心阶段，如图 1 所示：
**各模块说明：**

- **Speech Tokenizer**：将参考语音转换为离散的语音 token 序列，作为 LLM-TTS  decoder 的目标输出。
- **Emotion-aware LLM-TTS**：核心生成模型，包含文本编码器和基于 LLM 的解码器。其关键设计在于将情感信息显式注入 LLM-TTS 架构，使模型在生成语音 token 时能感知目标情感。
- **Flow-matching Vocoder**：将生成的语音 token 序列解码为最终波形。

### 2.2 训练策略

Emo-DPO 的训练分为两个阶段：

**阶段一：指令微调（Instruction Tuning）**

使用情感 TTS 数据集 $D_{ins}$ 对 LLM-TTS 模型 $\pi$ 进行监督微调。数据格式为三元组 $(x_j, E, y_j^E)$，其中 $x_j$ 为文本输入，$E$ 为情感提示词（如 Happy, Angry），$y_j^E$ 为对应情感的语音 token 序列。训练目标包含：

1. **SFT loss** $\mathcal{L}_{SFT}$：标准的 next-token prediction 交叉熵损失。
2. **KL 散度损失** $\mathcal{L}_{KL}$：最小化模型生成的情感语音 token 分布与目标分布 $P$ 之间的 KL 散度。具体采用标签平滑的 KL（label-smoothing Kullback-Leibler）实现：
   $$\mathcal{L}_{KL} = KL(P_\pi || P) = \mathbb{E}_{t_j \sim D_{ins}} \left[ p(y_j^E | E, x_j) \log \frac{p(y_j^E | E, x_j)}{p_\pi(y_j^E | E, x_j)} \right]$$

该损失确保模型学会生成与指定情感提示对齐的语音 token 序列。

**阶段二：Emo-DPO 训练**

仅做指令微调不足以让模型理解"为什么某个情感输出更正确"。因此，作者引入成对偏好数据集 $D_{pref}$ 进行 DPO 训练。对于每个正例 $(x_j, E, y_j^E)$，从训练数据中采样一个共享相同文本输入 $x_j$ 但情感不同的负例（如 Neutral）。数据格式为：

- 正例 $d_j^+ = E\langle\text{endofprompt}\rangle x_j \langle/s\rangle y_j^E \langle/s\rangle$
- 负例 $d_j^- = E'\langle\text{endofprompt}\rangle x_j \langle/s\rangle y_j^{E'} \langle/s\rangle$

DPO 目标函数定义为：

$$\mathcal{L}_{DPO}(\pi; \pi_{ref}) = -\mathbb{E}_{(d_j^+, d_j^-) \sim D_{pref}} \left[ \log \sigma \left( \beta \log \frac{\pi(y_j^E | E, x_j)}{\pi_{ref}(y_j^E | E, x_j)} - \beta \log \frac{\pi(y_j^{E'} | E', x_j)}{\pi_{ref}(y_j^{E'} | E', x_j)} \right) \right]$$

其中 $\pi_{ref}$ 为指令微调后的冻结参考模型，$\sigma$ 为 sigmoid 函数，$\beta$ 控制偏好优化的强度。

为进一步稳定训练，作者引入 ==Jensen-Shannon (JS) 散度== 替代原始 DPO 中的 log 比值运算，得到 JS-DPO 目标：

$$\text{JS-DPO} = \log \left( 1 + \exp \left( -\log \frac{\pi(y_j^E | E, x_j)}{\pi_{ref}(y_j^E | E, x_j)} + \log \frac{\pi(y_j^{E'} | E', x_j)}{\pi_{ref}(y_j^{E'} | E', x_j)} \right) \right)$$

最终训练目标为三项损失的加权和：

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{DPO} + \gamma \mathcal{L}_{KL} + \theta \mathcal{L}_{SFT}$$

其中 $\alpha, \gamma, \theta$ 为超参数，用于平衡偏好优化、分布对齐与监督学习。

---

## 3. 核心概念

### 3.1 Emotion-aware LLM-TTS

**定义**：Emotion-aware LLM-TTS 是 Emo-DPO 的骨干生成模型，指在传统的基于大语言模型的文本到语音（LLM-TTS）架构中，==显式地将情感信息注入到文本编码器和 LLM 解码器中==，使模型在生成语音 token 序列时能够感知并遵循目标情感提示。

**与相关概念的关系**：

- 与传统 FastSpeech2-based TTS 的区别：后者通常将情感嵌入（emotion embedding）直接拼接或相加到音素/文本特征上，而 Emotion-aware LLM-TTS 将情感提示词作为自然语言指令的一部分，利用 LLM 的 in-context learning 能力来理解和生成情感语音。
- 与通用 LLM-TTS（如 VALL-E、SoundStorm）的关系：通用 LLM-TTS 主要关注 speaker similarity 和语音自然度，通常不针对情感表达做专门设计。Emotion-aware LLM-TTS 是在通用架构基础上的情感感知扩展。

**消融验证**：论文 Table III 的消融实验表明，移除 instruction tuning（$\mathcal{L}_{SFT}$）后，模型在情感相似度（Emo SIM）、韵律相似度（Prosody SIM）和可懂度（Intelligibility）上均出现明显下降（Emo SIM 从 98.87 降至 98.74，Prosody SIM 从 3.89 降至 3.72，Intelligibility 从 4.54 降至 4.55）。这说明 instruction tuning 对于捕捉领域内情感特征至关重要。

**边界**：Emotion-aware LLM-TTS 的有效性依赖于预训练 LLM 已经具备的语义理解和生成能力。如果底层 LLM 规模过小或预训练数据中缺乏足够的情感相关文本，情感注入的效果可能受限。

### 3.2 Emo-DPO（情感直接偏好优化）

**定义**：Emo-DPO 是本文提出的核心训练方法，指==将直接偏好优化（DPO）应用于情感语音合成任务==，通过构建成对的"期望情感输出 vs. 非期望情感输出"偏好数据，直接优化 LLM-TTS 模型生成更符合人类情感偏好的语音 token 分布。

**与相关概念的关系**：

- 与标准 DPO（用于文本 LLM）的关系：标准 DPO 针对文本生成中的成对偏好数据进行优化（如 helpfulness、harmlessness）。Emo-DPO 将其适配到语音 token 空间，正例为期望情感的语音 token 序列，负例为同一文本下不同情感的语音 token 序列。
- 与 RLHF（基于人类反馈的强化学习）的关系：DPO 本身是对 RLHF 中 PPO 阶段的简化，省去了显式训练奖励模型的步骤。Emo-DPO 继承了这一优势，直接用偏好数据优化策略模型。

**消融验证**：Table III 显示，移除 DPO 损失（$\mathcal{L}_{DPO}$）后，Emo SIM 从 98.87 降至 98.77，Prosody SIM 从 3.89 降至 3.78，Intelligibility 从 4.54 降至 4.52。这表明 DPO 损失对提升语音清晰度、韵律相似度和情感相似度均有贡献，尤其是帮助模型捕捉"多样化、时间变化的韵律变化"。

**边界**：

- Emo-DPO 的有效性依赖于成对偏好数据的质量。如果负例选择不当（如与正例情感差异过大），模型可能学到的是粗粒度情感区分而非细微差异。
- 本文的负例采样策略是从训练集中选取同一文本、不同情感的样本，这种策略假设训练集中已包含足够的情感对比样本。对于数据稀疏的情感类别，该方法效果可能下降。

---

## 4. 核心洞见

**一句话洞见**：==在情感 TTS 中，"生成正确输出"的监督学习只能教会模型模仿，而"区分优劣输出"的偏好优化才能教会模型理解情感的细微差异。==

展开说明：本文的核心认知贡献在于将 NLP 领域的 DPO 范式迁移到语音合成领域，并验证了==成对偏好信号对于提升情感可控性的必要性==。实验表明，单纯依靠指令微调（SFT）虽然能让模型生成与情感标签对应的语音，但加入 DPO 后，模型在情感相似度、韵律表现和主观听感上均有进一步提升。这说明情感语音合成不仅需要"条件生成"（给定标签生成对应样本），还需要"判别学习"（学会区分同一文本下不同情感表达的优劣）。这一洞见对后续情感可控生成任务具有方法论层面的参考价值。

---

## 5. AI审稿评价

### 5.1 选题价值

**真实性与紧迫性**：情感 TTS 是一个真实存在的研究问题，随着 LLM-TTS 的兴起，如何提升情感可控性和表现力确实是一个紧迫课题。本文将 DPO 引入情感 TTS，切入点清晰，具有一定的创新性。

**缺口判断**：本文指出的"监督学习难以捕捉细微情感差异"是一个合理的缺口，但作者并未与近年来其他情感 TTS 工作（如基于扩散模型、基于流匹配的方法）做深入对比，缺口论述略显单薄。

### 5.2 方法贡献

**方法论层面**：本文的方法论贡献主要是**将已有技术（DPO）适配到新领域（情感 TTS）**，而非提出全新的训练范式。Emotion-aware LLM-TTS 的架构细节描述不够充分（如情感信息具体如何注入 LLM 的哪一层），可复现性受限。

**不可替代性**：DPO 在 NLP 中已被广泛验证，其在情感 TTS 中的有效性虽然得到了实验支持，但方法本身的不可替代性一般——理论上其他偏好学习或对比学习方法也可能达到类似效果。

### 5.3 实验严谨性

**Baseline 选择**：客观实验对比了 emoespeech 和 cosyvoice 两个 baseline。主观实验增加了 AB 偏好测试，设计较为完整。

**消融实验**：Table III 的消融实验覆盖了 DPO、SFT、instruction tuning 的移除操作，能够较好地定位各组件的贡献。但消融实验缺少对 ==$\beta$、$\alpha, \gamma, \theta$ 等超参数的敏感性分析==，也缺少对不同负例采样策略的对比。

**指标全面性**：客观指标包括 WER（可懂度）、Prosody SIM（韵律相似度）、Emo SIM（情感相似度）、Speech Emotion Recognition（情感识别准确率）；主观指标包括 MOS 和 AB preference test。指标覆盖较为全面。

**数据支撑追问**：
- 论文声称 Emo-DPO 在 Emo SIM 上达到 98.87，高于 emoespeech（98.26）和 cosyvoice（98.73），提升幅度分别为 0.61 和 0.14。提升幅度不大，但方向一致。
- AB 偏好测试中，Emo-DPO 相比 emoespeech 获得 85.6% 的偏好率，相比 cosyvoice 获得 88.7% 的偏好率，这是较强的主观证据。
- 然而，==实验仅在 ESD 数据集的英文数据上进行==， speaker 数量和情感类别均有限，泛化性存疑。

### 5.4 写作质量

**清晰度**：论文整体结构清晰，方法描述较为完整。但部分关键细节缺失：
- Emotion-aware LLM-TTS 的具体架构（情感嵌入如何与文本/语音 token 交互）描述不够详细。
- JS 散度替代 log 比值的具体推导和实现细节未给出。
- 训练数据的规模、DPO 数据对的构造细节（如何从 10 个 speaker 和 5 种情感中采样负例）描述不够充分。

**可复现性**：由于代码和预训练模型尚未公开（论文提到"codes will be released"），且部分超参数和架构细节缺失，当前可复现性一般。

### 5.5 判决

**weak accept**。理由：选题有实际价值，将 DPO 引入情感 TTS 的尝试具有启发性，实验结果总体支持方法有效性。但方法创新性有限（主要是已有技术的领域适配），实验规模较小且泛化性验证不足，部分关键实现细节缺失影响可复现性。

---

## 6. 启发

### 迁移

Emo-DPO 的核心机制——==用成对偏好数据训练生成模型区分同一条件下的优劣输出==——可以迁移到其他需要精细控制的语音/音频生成任务中。例如：

- **风格控制 TTS**：将 DPO 应用于说话风格（如正式 vs. 随意、快速 vs. 缓慢）的控制，通过构建风格偏好对提升风格可控性。
- **音乐生成**：在符号音乐生成或音频生成中，用 DPO 区分"更富表现力"和"更平淡"的音乐片段，提升生成音乐的情感表现力。

具体接入方式：在已有 LLM-based 生成模型后增加一个 DPO 微调阶段，构造任务相关的成对偏好数据集（正例为期望输出，负例为同一条件下较差的输出），并引入参考模型冻结的 DPO 目标函数。

### 混搭

Emo-DPO 的 DPO 训练阶段可以与==扩散模型或流匹配声码器==结合。当前 Emo-DPO 的 DPO 优化发生在语音 token 空间（LLM-TTS decoder 的输出），而最终的波形由 flow-matching vocoder 生成。一个可能的组合是：

- 将 DPO 思想扩展到声码器层面，直接在梅尔谱或波形空间上进行偏好优化，使得情感表达的细微差别不仅在 token 层面得到优化，也在最终的声学细节（如基频变化、能量包络）中得到增强。

### 反转

本文的一个默认假设是：==负例必须来自训练数据中的其他情感样本==。但如果我们反转这一假设，考虑用**模型自身生成的较差样本作为负例**（self-generated negatives），可能会带来额外收益：

- 这样可以让模型学习到"即使是我自己能生成的样本，也有优劣之分"，可能进一步提升情感表达的精细度。
- 这与 NLP 中的 DPO 变体（如 IPO、KTO）或自举偏好学习（self-play preference learning）的思路一致。

---

## 7. 同领域/同系列对比

| 方面 | Emo-DPO (本文) | emoespeech | cosyvoice |
|------|---------------|------------|-----------|
| **架构基础** | Emotion-aware LLM-TTS | FastSpeech2-based | LLM-based |
| **情感控制方式** | 指令微调 + DPO 偏好优化 | 情感嵌入 + 监督学习 | 文本/情感提示 + 监督学习 |
| **训练信号** | SFT + KL + DPO | SFT | SFT |
| **主观偏好率** | baseline | 14.4% vs. Emo-DPO | 11.3% vs. Emo-DPO |
| **Emo SIM** | 98.87 | 98.26 | 98.73 |
| **Prosody SIM** | 3.89 | 3.35 | 3.69 |
| **Intelligibility** | 4.54 | 7.17 | 4.94 |
| **数据规模** | ESD (1,500 utterances) | 推测更大 | 推测更大 |

**本文的增量贡献**：

1. **首次将 DPO 引入情感 TTS**：在情感语音合成领域验证了直接偏好优化的有效性，为主观可控性提供了新的训练范式。
2. **构建成对情感偏好数据集**：设计了针对情感 TTS 的 DPO 数据构造方式（同一文本、不同情感的语音对）。
3. **在 LLM-TTS 架构中整合情感感知**：提出 Emotion-aware LLM-TTS，利用 LLM 的 in-context learning 能力增强情感理解和生成。

