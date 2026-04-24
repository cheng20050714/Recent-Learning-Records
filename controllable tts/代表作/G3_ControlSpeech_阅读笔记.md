### **1. title && abstract**

**Title**:
ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control

**Abstract**:
==本文关注一个比传统零样本 TTS 或传统风格可控 TTS 更强的任务设定：**在未见说话人条件下，同时、独立地控制语音内容（content）、音色/说话人特征（timbre）与说话风格（style）**。==

本文提出 **ControlSpeech**，其核心思想是在**预训练的解耦 codec 空间**中分别建模内容、风格和音色，并通过**并行离散 codec 生成器**完成语音合成。


并基于 FACodec 将目标语音分解为内容码、韵律码、声学细节码与音色向量。
其中，作者将 `prosody codec + acoustic codec` 视为 style codec，并引入 **Style Mixture Semantic Density (SMSD)** 模块，以高斯混合分布建模 style text 与 style audio 之间的 many-to-many 对应关系。

实验表明，ControlSpeech 在 **style controllability、speaker similarity、语音质量、鲁棒性与泛化性** 等方面达到可比或最优结果，并在 many-to-many style control 场景中表现出更高的风格准确性与风格多样性。

---

### 2. **intro**
#### **研究背景与动机**

| 问题                                              | 说明                                                             | 影响                                                |
| ----------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| **零样本 TTS 与风格控制长期割裂**                           | 生成语音风格通常随参考语音一并复制                                              | 难以实现“保持音色不变而修改风格”                                 |
| **style prompt 与 speech prompt 存在表示纠缠**         | 参考语音本身已包含韵律、情绪、语速等风格因素                                         | 当 style text 与参考语音风格冲突时，模型控制能力显著下降                |
| **style text 与目标语音之间并非一一映射**                    | **多对多问题**：在使用文本描述风格时，必然会遇到自然语言的模糊性问题[[G3_NaturalSpeech3_阅读笔记]] | 确定性 style embedding 难以同时保证准确性与多样性                 |
| **缺乏同时支持 speaker cloning 与 style control 的数据集** | 文本风格描述数据通常较小，而零样本克隆依赖大规模多说话人数据                                 | 现有系统难以同时获得鲁棒 timbre cloning 与开放式 style control 能力 |
|                                                 |                                                                |                                                   |

>[!info]
>在 ControlSpeech 出现之前，TTS 领域处于一种“鱼与熊掌不可兼得”的状态 ：
>- 之前的 **Zero-Shot TTS 模型**（包括 VALL-E 等）可以完美克隆陌生人的音色，但风格是固定的，无法通过文本调整 。
>- 之前的 **风格可控 TTS 模型**（如 PromptTTS 等）可以通过文本改变风格，但它们无法进行 Zero-Shot 音色克隆（通常只能在有限的几个说话人中切换） 。


本文的出发点是：若要在一个统一系统内实现 **内容、音色、风格三者的独立控制**，则必须首先在表示层面削弱三者之间的耦合，并进一步在 style 建模中显式考虑自然语言描述所固有的分布性与模糊性。

#### ==**核心创新**==

1. **统一任务定义**：提出同时控制 `content / style / timbre` 的 zero-shot TTS 框架，使“未见说话人音色克隆”与“文本风格控制”在同一系统中成立。
2. **解耦 codec 建模**：基于预训练 FACodec，将目标语音分解为内容、风格与音色子空间，降低 speech prompt 与 style prompt 的耦合。
3. **SMSD 风格分布建模**：将 style text 映射为高斯混合分布，而非单一确定向量，从而建模 many-to-many style correspondence。
4. **VccmDataset 构建与系统评测**：构建同时包含 style prompt 与 speaker prompt 的评测数据，并分别考察 in-domain、out-of-domain speaker、out-of-domain style 与 many-to-many 特例场景。

#### **论文主线概括**
ControlSpeech 的方法论可以概括为以下三步：
1. **在预训练解耦 codec 空间中分离 speech factor**；
2. **用 style text 预测 style distribution，用 content text 约束语义内容，用 speech prompt 提供 timbre**；
3. **在离散 codec 空间中并行生成 content/style token，再以 timbre 条件解码为最终语音**。

这一设计表明，作者并非直接在 waveform 或 mel 频谱上做多条件融合，而是先通过**表示空间重组问题结构**，再进行生成。

---
### 3. **method**
#### 3.1 系统架构概述
ControlSpeech 是一个建立在**离散解耦 codec 表示**上的 encoder-decoder 并行生成框架。
其输入为：
- 内容文本 `X_c`
- 风格文本 `X_s`
- 参考语音 `X_t`

其输出为语音 `Y`，满足：
- 内容由 `X_c` 指定；
- 风格由 `X_s` 指定；
- 音色由 `X_t` 指定。


![[Pasted image 20260410161757.png|658]]
**Figure 1: ControlSpeech 的任务设定及其与既有模型的区别**
>Figure 1: The voice prompt, the content description, and the style description correspond to the timbre, content, and style representations in the discrete codec space in the left panel . The right panel compares ControlSpeech with previous style-controllable TTS and zero-shot TTS systems. In this comparison, we use the amplitude and color of the waveform to represent the styleand timbre.

图 1 的核心信息是：本文不再将“style-controllable TTS”与“zero-shot speaker cloning”视为两个独立任务，而是要求模型在一个统一框架内同时满足两者。

**Figure 2: ControlSpeech 整体架构、SMSD 模块与 codec generator 细节**
![[Pasted image 20260410163102.png|598]]

结合图 2，可将整体流程概括为：
- **Separate Encoders：** 文本内容通过 Text Encoder 提取音素级表征；Style Prompt通过 BERT 提取词级语义表征；Speech Prompt通过预训练的 FACodec 编码器和 Timbre Extractor 提取全局的音色嵌入 Global Timbre Embedding) 。
    
- ==**SMSD 模块 (Style Mixture Semantic Density)：** 这是解决文本控制中“多对多”问题的核心。它基于混合密度网络 (MDN)，将 BERT 提取的风格语义表征映射为混合高斯分布，从而实现对目标风格表征的层次化和多样性建模 。==
    
- **交叉注意力机制与时长预测器 (Cross-Attention & Duration Predictor)：** 全局风格表征与文本表征通过 Cross-Attention 融合，随后输入 Duration Predictor，将文本长度扩展为与音频帧对齐的长度 。
    
- **编解码生成器 (Codec Generator)：** 基于 Conformer 架构，采用掩码机制 (Mask-based) 进行非自回归的并行离散 Token 生成 。

#### 3.2 解耦 codec 表示空间：FACodec

##### 3.2.1 为什么要先做 codec 解耦？
[[G3_NaturalSpeech3_阅读笔记]]
##### 3.2.2 语音因子的定义方式
- 在训练时，模型利用预训练的 FACodec 作为特征解耦器，获取目标音频的真实内容特征 ($Y_c$)、韵律特征 ($Y_p$)、声学细节特征 ($Y_a$) 以及音色特征 ($Y_t$) ,风格特征被定义为 $Y_s = concat(Y_p, Y_a)$ 。

这一选择具有明确的工程含义：
- `Y_c` 保留与文本对应最紧密的语义信息；
- `Y_t` 承担全局 speaker identity；
- `Y_p + Y_a` 则承载语速、节奏、音高变化、能量分布及局部表现差异。

因此，ControlSpeech 的生成目标并不是直接预测 waveform，而是生成：

$$
Y_{codec} = \mathrm{concat}(Y_s, Y_c)
$$

==随后再在解码阶段注入 `Y_t`。== 
- [?] 这点或许是实现 ZS 的关键？

> [!summary]
**优势**：
>- 降低 style 与 timbre 的直接耦合；
>- 允许生成器专注于内容与风格，而将音色控制留给 decoder；
>- **有利于在未见 speaker 上维持 speaker similarity。**
>
>**潜在代价**：
>- 若 FACodec 的 factorization 并不充分，则 pitch、prosody 与 speaker trait 仍可能互相泄漏；
>- 这一点在实验中已有体现：ControlSpeech 的综合表现较强，但 **pitch accuracy 并非最优**。

#### **3.3 SMSD：Style Mixture Semantic Density**

##### 3.3.1 SMSD 的结构
1. 全局语义对齐 (Global Semantic Alignment) ——==解决“多对一”==

输入的风格文本 $X_s$ 序列在句首拼接 `[CLS]` 标记后，输入预训练的 BERT 模型。网络提取 `[CLS]` 对应的隐藏层向量作为全局风格语义表征 $X_s^{\prime} \in \mathbb{R}^n$。

- **作用：** 利用大型语言模型的先验知识进行**语义聚类**，将字面不同但语义一致的描述压缩至统一的隐空间，消除字面歧义，并增强对分布外 (OOD) 文本的泛化能力。
    
2. 混合密度网络 (Mixture Density Network, MDN) —— ==解决“一对多”==

模块假设目标风格表征 $Y_s^{\prime} \in \mathbb{R}^d$ 服从由输入 $X_s^{\prime}$ 驱动的混合高斯分布 (GMM)。通过神经网络 $f_{\theta}$，回归出 $K$ 个高斯分量的参数：混合权重 $\pi_k$、均值 $\mu^{(k)}$ 和方差 ${\sigma^2}^{(k)}$。

$$P_{\theta}({Y_s}^{\prime}|{X_s}^{\prime})=\sum_{k=1}^{K}\pi_k\mathcal{N}(\mu^{(k)},{\sigma^2}^{(k)})$$

其中，$\pi_k$ 经过 Softmax 归一化以保证 $\sum \pi_k = 1$。

- **作用：** 将确定的文本映射为连续的概率分布，为后续的采样提供数学基础。

---
##### 3.3.3 SMSD 的作用机理

1. **语义对齐**：利用 BERT 将同义或近义 style text 映射到相近语义区域；
2. **分布建模**：利用混合高斯显式表示“同一描述可能对应多个风格强度”的事实。

因此，SMSD 的意义不只是“增加随机性”，而是在 style control 中显式引入**条件分布建模**。这一点相对于仅用 deterministic style embedding 的方法更符合自然语言风格描述的统计特性。

##### 3.3.4 Noise perturbation 的设计

为进一步调节风格采样的稳定性与多样性，作者在 SMSD 中增加噪声扰动模块，并比较了四类噪声模式：

1. **Fully factored**
2. **Isotropic**
3. **Isotropic across clusters**
4. **Fixed isotropic**

>实验表明，**Isotropic across clusters** 在风格准确性与风格多样性之间取得了更好的平衡，因此被选为默认配置。

##### 3.3.5 SMSD 的训练目标

论文主文给出的 SMSD 优化目标是 style observation 的负对数似然：

$$
L_{SMSD} = - \log P_\theta(Y'_s|X'_s)
$$

>[!summary] 核心流程总结
>文本 $\rightarrow$ 产生一个数学概率分布 $\rightarrow$ **通过采样选定一种风格向量** $\rightarrow$ 生成器根据这个唯一向量去生成最终的 Token。
>通过 **SMSD 模块**，成功地在数学上构建出了一个既有边界（受文本控制）、又有弹性（允许采样扰动）的风格概率空间。
>这为一对一的语音赋予变化，使之更接近人类水平。

#### 3.4 Codec 生成与语音解码
##### 3.4.1 生成目标的定义
ControlSpeech 的离散生成目标为：

$$
Y_{codec} = \mathrm{concat}(Y_s, Y_c) = C_{1:T,1:N}
$$

其中：
- `T` 为扩展后的帧长度；
- `N` 为每帧的离散 code channel 数。

这说明模型需要在每个时间步、每个离散 codebook 维度上预测正确 token。

##### 3.4.2 Mask-based 并行 codec generator
作者采用类似 SoundStorm/MaskGIT 的**并行掩码生成范式**。
训练时：

- 随机选择一个 channel `i` 进行优化；
- 对该 channel 的 token 进行随机 mask；
- 掩码比例按 cosine schedule 采样；

即：

$$
M_i \sim \mathrm{Bernoulli}(p), \quad p=\cos(u'),\; u' \sim U(0,\pi/2)
$$

模型学习在以下条件下恢复被 mask 的 token：

$$
P(M_i C_{1:T,i} \mid C_{1:T,<i}, X_{1:T}, \bar{M_i}C_{1:T,i};\theta)
$$

其中 `X_{1:T}` 是由文本内容与 style 条件融合后得到的帧级表示。

这一设计的优势在于：

- 避免完全自回归带来的推理时延；
- 保留离散 codec 建模的稳定性；
- 允许在多轮迭代中逐步细化低置信度位置。

##### 3.4.3 Cross-Attention 与 Duration Predictor 的作用

在生成 codec 之前，作者先将：
- 文本编码器输出的内容表示；
- SMSD 生成的全局 style representation；
 
通过 **cross-attention** 融合，再送入 **Duration Predictor** 完成长度扩展。

这一点十分关键。风格控制不仅影响声学细节，也会直接影响：

- 语速；
- 节奏；
- 停顿分布；
- 句内时间结构。

因此，style 条件必须在 duration modeling 之前就进入系统，否则模型难以真正控制 speaking rate 与 prosodic rhythm。

##### 3.4.4 Timbre 注入方式：Conditional Normalization

在生成 `Y_{codec}` 后，模型并不直接解码，而是先通过 timbre embedding `Y_t` 做条件归一化：

$$
Y = \mathrm{CodecDecoder}\left(W_\gamma Y_t \cdot \frac{Y_{codec}-\mu_c}{\sigma_c^2}+W_\beta Y_t\right)
$$

这一设计的核心意义是：**将 speaker identity 的控制责任从 token generator 转移到 decoder 端的全局条件调制**。

与直接让生成器同时学习 content/style/timbre 相比，这一方案更符合 factorized control 的目标：

- token generator 主要负责“生成何种内容、以何种风格表达”；
- decoder 负责“以何种音色实现这些离散表示”。

#### 3.5 训练范式与推理流程

##### 3.5.1 总体损失函数

ControlSpeech 的总损失为：

$$
L = L_{codec} + L_{dur} + L_{SMSD}
$$

其中：

- `L_{codec}`：codec generator 的交叉熵损失；
- `L_{dur}`：duration predictor 的 MSE 损失；
- `L_{SMSD}`：style density 的负对数似然损失。

##### 3.5.2 训练流程

训练阶段的核心步骤如下：

1. 用冻结的 FACodec encoder 从目标语音提取 `Y_c, Y_p, Y_a, Y_t`；
2. 用 style codec 经 style extractor 得到全局 style target `Y'_s`；
3. 用 MFA 获取音素时长作为 duration supervision；
4. 用 GT duration 与 GT style representation 训练 duration predictor 和 codec generator；
5. 用 SMSD 逼近 `P(Y'_s|X'_s)`。

##### 3.5.3 推理流程

推理阶段的流程可写为：

1. `style text -> BERT -> SMSD`，得到 style distribution 参数；
2. 从 mixture distribution 中采样 style latent；
3. `content text -> phoneme -> text encoder`；
4. 将 style latent 与 text representation 融合，并预测 duration；
5. 采用 confidence-based iterative decoding 并行生成 codec token；
6. 将 timbre embedding 通过 conditional normalization 注入 decoder；
7. 解码得到最终 waveform。

##### 3.5.4 训练-推理不对齐问题

复现时需要特别注意两类不对齐：

1. **训练时使用 GT duration，推理时使用 predicted duration**；
2. **训练时使用 GT style representation，推理时使用 SMSD 采样结果**。

这意味着：

- duration 误差会影响节奏稳定性；
- SMSD 采样质量会直接影响 style accuracy；
- 当 timbre 与 style 对 pitch 的约束发生竞争时，模型更容易出现控制误差。

实验中 pitch 指标不占优，与这一点具有一致性。

---
### 4. **experiment**

#### 4.1 数据集与实验设置

##### 4.1.1 VccmDataset

由于现有数据集缺少同时包含 speaker prompt 与 style prompt 的大规模资源，作者在 TextrolSpeech 基础上构建 **VccmDataset**。

其主要处理流程包括：

- 以 LibriTTS 和 TextrolSpeech 情感数据为基础；
- 为每条语音标注 `gender / volume / speed / pitch / emotion` 五类属性；
- 对 pitch、speed、volume 进行区间划分，并剔除边界样本；
- 使用更精细的属性标签重新对齐 style description 与音频片段；
- 构建适用于 speaker cloning 与 style control 联合评测的数据集。

##### 4.1.2 四类测试集

| 测试集 | 作用 |
|------|------|
| **Test set A** | 主 style controllability 评测 |
| **Test set B** | out-of-domain speaker cloning |
| **Test set C** | out-of-domain style generalization |
| **Test set D** | many-to-many style control 特例评测 |

其中：

- Test set B 中的 speaker 不出现在训练集；
- Test set C 中的 style prompt 由专家重写，不出现在训练集；
- Test set D 用于评估同一 style 描述的多样化实现能力与 timbre 稳定性。

##### 4.1.3 Baselines 与评测指标

**风格控制基线**：

- PromptStyle
- Salle
- InstructTTS
- PromptTTS 2

**speaker cloning 基线**：

- VALL-E
- MobileSpeech

**客观指标**：

- Pitch / Speed / Volume / Emotion accuracy
- WER
- Spk-sv

**主观指标**：

- MOS-Q
- MOS-S
- MOS-TS
- MOS-SA
- MOS-SD

#### 4.2 主实验结果

##### 4.2.1 Style controllability 主结果（Test set A）

**Table 1: 主 style controllability 结果**
![[G3_ControlSpeech_Table1.png|760]]

从 Table 1 可得到以下结论：

1. **ControlSpeech 在大多数关键综合指标上最优**：
   - Speed accuracy：`0.829`
   - Volume accuracy：`0.894`
   - Emotion accuracy：`0.557`
   - WER：`2.9`
   - Spk-sv：`0.89`
   - MOS-Q：`3.91`

2. **Pitch 并非最优指标**：
   - PromptTTS 2 的 pitch accuracy 为 `0.867`
   - ControlSpeech 为 `0.833`

这表明，ControlSpeech 的主要优势在于**综合控制质量与 timbre cloning 的兼容性**，而不是在所有 style factor 上都绝对领先。特别是 pitch 控制，仍然是 simultaneous timbre cloning + style control 场景中的薄弱维度。

##### 4.2.2 Speaker cloning 与 many-to-many style control 结果

**Table 2 / Table 3: Out-of-domain timbre cloning 与 many-to-many style control**
![[G3_ControlSpeech_Table3.png|620]]

**关于 Test set B（unseen speaker cloning）**：

- ControlSpeech 的 `WER = 3.3`，优于 VALL-E (`6.7`) 与 MobileSpeech (`4.1`)；
- `MOS-Q = 3.95`，与 MobileSpeech (`3.94`) 接近；
- `MOS-S = 3.96`，与 MobileSpeech (`4.01`) 可比。

这说明 ControlSpeech 在 unseen speaker 条件下保持了较强的鲁棒性与质量，其 timbre cloning 能力并未因额外的 style control 目标而明显退化。

**关于 Test set D（many-to-many style control）**：

- MOS-TS：`4.01`
- MOS-SA：`3.84`
- MOS-SD：`4.05`

ControlSpeech 在三项指标上均优于 PromptStyle、InstructTTS 以及去除 SMSD 的版本，说明 SMSD 对**风格准确性**与**风格多样性**均有直接贡献。

##### 4.2.3 Out-of-domain style generalization（Test set C）

附录 Table 5 表明，在训练中未出现的 style prompt 改写条件下，ControlSpeech 仍然在以下指标上优于基线：

- Speed accuracy
- Volume accuracy
- WER
- Spk-sv
- MOS-Q

但 pitch accuracy 依然不是最优项。

这说明 BERT + SMSD 的确增强了 style semantics 的外域泛化能力，但并未完全解决 pitch 与 timbre/风格之间的耦合问题。

#### 4.3 消融实验

##### 4.3.1 去除 codec decoupling 的影响

Table 4 显示，去除 decoupling 后：

- Pitch：`0.833 -> 0.492`
- Speed：`0.829 -> 0.517`
- Volume：`0.894 -> 0.582`
- Emotion：`0.557 -> 0.237`

该结果说明：

- ControlSpeech 的性能提升并非仅来自更大的生成器或更多条件输入；
- **预训练解耦 codec 表示空间**是其成立的必要条件。

换言之，若不先解决 speech factor disentanglement，style prompt 与 speech prompt 的冲突会显著破坏 controllability。

##### 4.3.2 去除 SMSD 的影响

在 many-to-many 测试中，去除 SMSD 后：

- MOS-SA：`3.84 -> 3.59`
- MOS-SD：`4.05 -> 3.66`

这表明 SMSD 并非仅提供随机采样能力，而是有效提升了：

- style description 到 target style 的映射精度；
- 同一 style description 下的生成多样性。

##### 4.3.3 Mixture component 数量的影响

附录中比较了 `K=3,5,7` 三种 mixture 数量。结果表明：

- `K=5` 在 MOS-SA 与 MOS-SD 之间取得最佳折中；
- mixture 数过大并不会持续带来收益，反而可能削弱 style control 的准确性。

这说明，SMSD 的作用不是简单增加模型容量，而是在有限的 mixture 数量下学习一个更稳定的风格分布近似。

##### 4.3.4 噪声模式的影响

附录中比较四种 noise mode 后发现：

- **Isotropic across clusters** 在准确性和多样性之间表现最佳；
- fully factored 与 fixed isotropic 均非最佳选择。

这一结果进一步支持作者的观点：**style control 的关键不只是“是否采样”，而是“采样分布如何被约束”**。

#### 4.4 实验总结

综合实验部分，论文的证据链条较为完整，可以得出以下判断：

1. **ControlSpeech 的主要优势是综合能力而非单一指标极值**。
2. **codec decoupling 与 SMSD 是两项真正起作用的关键设计**。
3. **pitch 仍然是当前框架下最难完全独立控制的因子**。
4. **SMSD 在 many-to-many 场景中的收益是本文最有说服力的实验结果之一**。

---

### 5. **conclusion** && **takeaway**

#### 综合评价

若从创新类型看，ControlSpeech 更接近于 **架构创新**，而非底层生成范式的根本突破。其主要价值在于：

- 将 zero-shot speaker cloning 与 style-controllable TTS 两条研究线统一到同一系统；
- 通过**解耦 codec + 风格分布建模 + 并行离散生成 + timbre 条件解码**形成完整而自洽的技术闭环；
- 为后续“多因素独立控制”TTS 系统提供了可复用的系统设计范式。

#### 方法优势

1. **任务定义清晰且具有实际价值**：能够更自然地对应工业场景中的“指定说话人 + 指定说话风格 + 指定文本内容”需求。
2. **方法分工明确**：content、style、timbre 分别由不同条件路径建模，系统结构较为清晰。
3. **实验设计完整**：不仅报告主实验结果，还单独验证了 out-of-domain speaker、out-of-domain style 与 many-to-many 特例场景。

#### 主要局限

1. **pitch disentanglement 仍不充分**  
   虽然模型总体表现较强，但 pitch accuracy 并未达到最优，说明当前 factorization 对 pitch 的独立建模仍有不足。

2. **对预训练 codec 质量依赖较强**  
   ControlSpeech 的成立高度依赖 FACodec 的解耦质量与大规模预训练能力，若迁移到其他 codec 或其他语言环境，效果未必可以直接保持。

3. **style 语义空间仍然相对受限**  
   当前 style 描述主要围绕 pitch、speed、volume、emotion 等属性展开，与开放域自然语言风格控制仍存在差距。

4. **many-to-many 评测仍以主观指标为主**  
   MOS-SA / MOS-SD 虽然有效，但若未来能够进一步设计更客观、可重复的 style diversity 指标，则论证会更充分。

#### 对后续研究的启发

1. **先解耦，再控制**  
   对于多因素可控生成问题，表示空间的结构化设计往往先于生成器本身的重要性。

2. **自然语言风格控制应优先采用分布建模**  
   只要文本描述存在模糊性与语义同义性，采用 mixture / density modeling 往往比单点回归更合理。

3. **speaker identity 可在 decoder 端注入**  
   将 timbre 作为全局条件调制，而非完全交由 token generator 学习，是一条值得继续探索的技术路线。

4. **评测集设计必须覆盖冲突条件与分布外条件**  
   若只在 in-domain 条件下评测，模型的“独立控制能力”往往会被高估。

#### 复现时最值得关注的点

若后续需要复现本文，建议优先关注以下部分：

1. FACodec 的版本及其解耦质量；
2. style extractor 与全局 style target 的实现细节；
3. SMSD 的 mixture 数量与 noise mode；
4. confidence-based iterative decoding 的调度策略；
5. 训练时 GT duration / GT style 与推理时预测结果之间的不对齐问题。

#### 总结

ControlSpeech 的核心贡献并不在于提出一种全新的生成范式，而在于**首次较系统地证明：在 zero-shot speaker cloning 的前提下，style control 可以被设计为一个相对独立的控制变量，而非参考语音的附属属性**。这一点对后续 controllable TTS 的研究方向具有较强的启发意义。
