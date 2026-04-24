### 1. **title** && **abstract**
**Abstract**:
近年来，基于大语言模型（LLM）的文本转语音（TTS）技术因其高自然度和零样本能力成为主流。
语音token在LLM-based TTS模型中扮演关键角色。

现有语音token通过**无监督学习**获得，缺乏显式语义信息且与文本对齐不佳。
本文提出使用**监督语义token（Supervised Semantic Tokens, S3）**，通过在多语言语音识别模型编码器中插入向量量化层获得。

基于这些token，进一步提出**CosyVoice**合成器，包含：
1. **LLM**：用于文本到token生成
2. **条件流匹配模型（Conditional Flow Matching）**：用于token到语音合成

实验结果表明，监督语义token在零样本语音克隆的内容一致性和说话人相似度上显著优于现有无监督token。此外，大规模数据可进一步提升合成性能，表明CosyVoice的可扩展性。

---
### 2. **intro**
#### 研究背景与动机

| 问题         | 说明                              | 影响      |
| ---------- | ------------------------------- | ------- |
| **缺乏显式语义** | 无监督学习（如HuBERT、EnCodec）不直接关联文本内容 | 内容一致性差  |
| **文本对齐差**  | 语音token与音素/字符没有明确对应关系           | 发音错误、漏字 |
#### ==核心创新==
1. **监督语义token（S3 Tokenizer）**：从多语言ASR模型（SenseVoice）提取，通过编码器中插入向量量化层获得
2. **CosyVoice架构**：LLM + 条件流匹配模型，无需音素器和强制对齐器
3. **x-vector分离建模**：将语音分解为语义、说话人、韵律组件，LLM建模语义和韵律，流匹配模型建模音色和环境信息
4. **指令微调版本（CosyVoice-instruct）**：支持说话人身份、说话风格、细粒度副语言特征控制

---
### 3. **method**
#### 3.1 系统架构概述
CosyVoice由四个核心组件组成：
**Figure 1: CosyVoice整体架构**
![[Pasted image 20260408194518.png]]


**(a) S3 Tokenizer（监督语音分词器）**
- 输入：Mel谱图 → Encoder1 → VQ → Encoder2 → ASR Decoder
- 输出：离散语音token序列

**(b) CosyVoice LM概览**
- **Text Encoder**：对齐文本和语音token的语义空间
- **LLM**：自回归生成语音token
- **Flow Matching Model**：将token转换为Mel谱图
- **HiFi-GAN Vocoder**：Mel谱图 → 波形

**(c) 条件流匹配模型细节**
- 基于ResNet1D和Transformer Block
- 条件：说话人嵌入v、语义token、掩码Mel谱图、时间步t

#### 3.2 监督语义Token（S3 Tokenizer）

##### 3.2.1 ASR
**任务**：将语音转换为文本
**CosyVoice中的ASR**：
- 作为S3 Tokenizer的backbone
- 关键作用：提供**监督信号**，确保语音token与文本内容对齐

##### 3.2.2 核心设计：在ASR编码器中插入向量量化层
- [?] **为什么需要两层Encoder？**

| 层级 | 作用 | 说明 |
|------|------|------|
| **Encoder1** | 提取**低级声学特征** | 频谱特征、局部时序模式 |
| **VQ（量化）** | **离散化** | 将连续表示转为离散token |
| **Encoder2** | 提取**高级语义特征** | 上下文建模、语义理解 |

1. **信息层次不同**：Encoder1输出局部、低级、连续；需要Encoder2提取全局、高级、离散后重新编码
2. **ASR任务要求**：ASR Decoder需要**上下文感知的语义表示**，直接在量化后解码会损失太多信息
3. **S3 Token的质量**：在Encoder1之后量化保留足够信息，经过Encoder2再解码验证token质量

>[!hint]
>**类比**：Encoder1 = 耳朵（听到声音），VQ = 大脑编码（转为内部语言），Encoder2 = 理解（理解语义），ASR Decoder = 转述（输出文字）
>


##### 3.2.3 数学表达
1. **编码器第一部分**：
$$H = \text{Encoder1}(\text{PosEnc}(X))$$

2. **向量量化**（对第l帧隐藏表示$h_l$）：
$$\mu_l = \text{VQ}(h_l, C) = \arg\min_{c_n \in C} ||h_l - c_n||_2$$

3. **Codebook更新**（指数移动平均EMA）：
$$c_{\mu_l} := \alpha c_{\mu_l} + (1-\alpha)h_l$$

**EMA作用**：
- **稳定性**：避免codebook剧烈变化
- **平滑更新**：新样本逐步影响codebook，而非直接替换
- **防止崩溃**：避免某些codebook向量长期不被使用（codebook collapse）

**对比**：直接更新 $c_{new} = h_l$ 噪声敏感、不稳定；EMA更新 $c_{new} = 0.99 \cdot c_{old} + 0.01 \cdot h_l$ 平滑累积、稳定

4. **编码器第二部分**：
$$\tilde{H} = \text{Encoder2}(\text{PosEnc}(\bar{H}))$$

5. **ASR解码**（训练时）：
$$P(Y|X) = \text{ASRDecoder}(\tilde{H}, Y_{-1})$$

>[!hint] 
> 首次引入**监督学习**，这里可以借鉴思路。

| | 无监督（EnCodec/HubERT） | **监督（S3）** |
|---|---|---|
| **训练目标** | 重建语音 / 掩码预测 | **ASR：语音→文本** |
| **是否需文本标签** | 否 | **是** |
| **token语义** | 不明确 | **与文本强对齐** |
| **内容一致性** | 差（WER高） | **好（WER低）** |

#### 3.3 大语言模型用于TTS

##### 3.3.1 为什么需要LLM？ASR不是已经对齐了吗？
>[!note]
**类比理解**：
>- **S3 Tokenizer = 字典（词汇表）**：建立"apple"→[a-p-p-l-e]怎么发音的映射，但字典不会"造句"
>- **LLM = 作家（生成器）**：给定主题（文本），写出一篇文章（语音token序列），决定用什么词、什么语气、什么节奏


**推理时的流程**：
```
用户输入："Hello world" + 参考语音（提供说话人v）

步骤1：提取说话人v（x-vector）
       ↓
步骤2：LLM生成：[S, v, 文本编码, T, μ₁, μ₂, ..., μ₁₀₀, E]
       ↑ 这里LLM是核心！
       ↓
步骤3：S3 token → Flow Matching → Mel谱图 → 波形
       ↑ S3 Tokenizer只在这里用到（Encoder1 + VQ）
```

##### 3.3.2 序列构建与训练目标

**序列构建**：

$$[S, v, \{\bar{y}_u\}_{u \in [1:U]}, T, \{\mu_l\}_{l \in [1:L]}, E]$$

其中：
- $S, E$：序列开始/结束标记
- $v$：说话人嵌入（x-vector提取）
- $\bar{Y} = \{\bar{y}_u\}$：BPE分词后的文本编码
- $T$：语音转向标记（Turn of Speech）
- $\{\mu_l\}$：S3 tokenizer提取的语音token

**文本编码器**：
$$\bar{Y} = \text{TextEncoder}(\text{BPE}(Y))$$

**训练目标**（仅计算语音token和E的交叉熵）：
$$L_{\text{LLM}} = -\frac{1}{L+1} \sum_{l=1}^{L+1} \log q(\mu_l)$$

**自回归语言建模**（类似GPT生成文本）

**关键设计**：
- 使用BPE token而非音素，无需音素器
- 文本编码器对齐文本和语音token语义空间
- Teacher-forcing训练

#### 3.4 最优传输条件流匹配（OT-CFM）

**动机**：相比DDPM，流匹配具有更简单的梯度、更易训练和更快的生成速度

**连续时间归一化流（CNF）**：

概率密度路径由时间相关向量场$\nu_t(X): [0,1] \times \mathbb{R}^{L \times D} \to \mathbb{R}^{L \times D}$定义，生成流$\phi_t$：

$$\frac{d}{dt}\phi_t(X) = \nu_t(\phi_t(X), t)$$
$$\phi_0(X) \sim p_0(X) = \mathcal{N}(X; 0, I)$$
$$\phi_1(X) \sim p_1(X)$$

**最优传输流**：
$$\phi_t^{\text{OT}}(X_0, X_1) = (1 - (1-\sigma)t)X_0 + tX_1$$
$$\omega_t(\phi_t^{\text{OT}}(X_0, X_1)|X_1) = X_1 - (1-\sigma)X_0$$

**训练目标**：
$$L_{\text{OT-CFM}} = \mathbb{E}_{t, p_0(X_0), q(X_1)} ||\omega_t(\phi_t^{\text{OT}}(X_0, X_1)|X_1) - \nu_t(\phi_t^{\text{OT}}(X_0, X_1)|\theta)||$$

**神经网络条件**：
$$\nu_t(\phi_t^{\text{OT}}(X_0, X_1)|\theta) = \text{NN}_\theta(\phi_t^{\text{OT}}(X_0, X_1), t; v, \{\mu_l\}_{1:L}, \tilde{X}_1)$$

其中$\tilde{X}_1$是掩码Mel谱图（从随机起点到结束置零）。

**余弦调度器**：
$$t := 1 - \cos(\frac{1}{2}t\pi)$$

使生成过程在开始时有更多步骤。

**Classifier-Free Guidance（CFG）**：

训练时以0.2概率丢弃条件$\Psi = \{v, \{\mu_l\}, \tilde{X}_1\}$。

推理时向量场修改：
$$\tilde{\nu}_t(\phi_t^{\text{OT}}(X_0, X_1)|\theta; \Psi) = (1+\beta) \cdot \nu_t(\phi_t^{\text{OT}}(X_0, X_1)|\theta; \Psi) - \beta \cdot \nu_t(\phi_t^{\text{OT}}(X_0, X_1)|\theta)$$

其中$\beta = 0.7$为引导强度。

#### 3.5 零样本上下文学习

**Figure 2: 序列构建方式**
![[Pasted image 20260408204001.png|433]]
**(a) 同语言零样本克隆**：
- 将提示语音token和输入文本合并
- 提示语音token作为已生成的前缀
- LLM自回归生成后续token

**(b) 跨语言语音克隆**：
- 省略提示文本和token（防止源语言韵律影响目标语言）
- 保留说话人嵌入v

**推理流程**：
1. 提取提示语音的S3 token和说话人嵌入v
2. 构建输入序列（提示token + 输入文本编码 + T）
3. LLM自回归生成语音token
4. 将生成的token与提示token拼接，作为流匹配模型的条件
5. 流匹配模型生成Mel谱图
6. HiFi-GAN声码器生成波形

#### 3.6 指令微调版本（CosyVoice-instruct）

在CosyVoice-base基础上进行指令微调，支持：

| 控制类型 | 示例 |
|---------|------|
| **说话人身份** | "Selene 'Moonshade', is a mysterious, elegant dancer..." |
| **说话风格** | "A happy girl with high tone and quick speech." |
| **细粒度副语言** | "Well that's kind of scary [laughter]." |

**训练数据**：
- 说话人身份：101小时
- 说话风格：407小时
- 细粒度副语言：48小时

**关键设计**：微调时不使用说话人嵌入，让模型从指令中学习说话人特征。

---

### 4. **experiment**

#### 4.1 数据集

**小规模单语数据集（LibriTTS）**：
- 585小时，2,456英语说话人
- 用于与基线模型公平对比

**大规模多语数据集（内部）**：

| 语言 | 时长（小时） |
|------|------------|
| 中文（ZH） | 130,000 |
| 英语（EN） | 30,000 |
| 粤语（Yue） | 5,000 |
| 日语（JP） | 4,600 |
| 韩语（KO） | 2,200 |

**总计约17万小时**

#### 4.2 模型配置
**S3 Tokenizer**：
- 小规模：基于ESPNet Conformer，6层encoder + VQ + 6层encoder
- 大规模：基于SenseVoice-Large，预训练后微调210K步
- Codebook：单codebook，4096码本

**CosyVoice模型**：

| 配置 | Tiny | Normal |
|------|------|--------|
| Text Encoder层数 | 6 | 6 |
| Text Encoder维度 | 512 | 1024 |
| LLM层数 | 12 | 14 |
| LLM维度 | 512 | 1024 |

#### 4.3 S3 Tokenizer评估

**Table 5: 向量量化对ASR性能影响（LibriTTS）**

| Model | dev_clean | test_clean | test_other |
|-------|-----------|------------|------------|
| Conformer | 2.62 | 2.89 | 6.57 |
| Conformer-VQ | 3.13 | 3.18 | 7.56 |

插入VQ后WER仅轻微上升，证明S3 token保留足够语义信息。

**Table 6: 多语言S3 token语义保留能力（Common Voice）**

| Model | zh-CN | en |
|-------|-------|-----|
| Whisper-L-V3 | 12.55 | 9.39 |
| SenseVoice-L | 8.68 | 9.77 |
| **S3 tokens** | **12.06** | **15.38** |

在中文上S3 token优于Whisper-Large V3（相对错误率降低4.14%）。

#### 4.4 与基线模型对比

**Table 7: LibriTTS test-clean上的内容一致性和说话人相似度对比**

| Model           | Text Token | Speech Token | WER (%)  | #INS+DEL | #SUB    | SS        |
| --------------- | ---------- | ------------ | -------- | -------- | ------- | --------- |
| Original        | -          | -            | 3.01     | 66       | 200     | 69.67     |
| VALL-E          | Phone      | EnCodec      | 18.70    | 342      | 1312    | 53.19     |
| UniAudio        | Phone      | EnCodec      | 8.74     | 254      | 519     | 47.56     |
| SpearTTS        | Phone      | HuBERT       | 6.14     | 133      | 410     | 51.71     |
| Exp-1           | Phone      | HuBERT       | 7.41     | 325      | 409     | 67.85     |
| **Exp-2**       | Phone      | **S3**       | **5.05** | **122**  | **325** | **67.85** |
| **Exp-3**       | **BPE**    | **S3**       | **3.93** | **108**  | **239** | **67.85** |
| **Exp-4-Large** | **BPE**    | **S3**       | **3.17** | **96**   | **184** | **69.49** |

**关键发现**：
1. S3 token（Exp-2）vs HuBERT（Exp-1）：WER从7.41%→5.05%，内容一致性显著提升
2. BPE token（Exp-3）vs Phone（Exp-2）：WER从5.05%→3.93%，无需音素器也能达到更好效果
3. 大规模数据（Exp-4）：WER降至3.17%，接近人类水平（3.01%），说话人相似度超越人类

#### 4.5 生成质量评估

**Table 8: 英语（LibriTTS test-clean）**

| Model | WER (%) | #Ins.&Del. | SS |
|-------|---------|------------|-----|
| Original | 2.66 | 92 | 69.67 |
| ChatTTS | 8.32 | 441 | - |
| **CosyVoice** | **2.89±0.18** | **88.60±3.88** | **74.30±0.15** |
| + 5× re-ranking | 1.51 | 47 | 74.30 |

**Table 9: 中文（AISHELL-3 test set）**

| Model | CER (%) | #Ins.&Del. | SS |
|-------|---------|------------|-----|
| Original | 2.52 | 25 | 74.15 |
| ChatTTS | 3.87 | 111 | - |
| **CosyVoice** | **3.82±0.24** | **24.4±2.24** | **81.58±0.16** |
| + 5× re-ranking | 1.84 | 11 | 81.58 |

**关键发现**：
- CosyVoice达到人类水平的内容一致性
- 说话人相似度（SS）超越人类（英语74.30 vs 69.67，中文81.58 vs 74.15）
- ChatTTS存在说话人泄露问题（插入删除错误多），CosyVoice无此问题
- ASR重排序可进一步提升内容一致性

#### 4.6 情感可控性

**Table 10: 情感控制准确率对比**

| Emotion | CosyVoice-base | CosyVoice-instruct | w/o instruction |
|---------|---------------|-------------------|-----------------|
| Happy | 1.00±0.00 | 1.00±0.00 | 0.98±0.01 |
| Sad | 0.45±0.05 | **0.98±0.02** | 0.77±0.04 |
| Angry | 0.59±0.03 | **0.83±0.04** | 0.49±0.12 |
| Surprised | 0.26±0.02 | **0.64±0.03** | 0.28±0.06 |
| Fearful | 0.88±0.01 | 0.87±0.03 | 0.83±0.04 |
| Disgusted | 0.46±0.06 | **0.93±0.02** | 0.45±0.16 |

CosyVoice-instruct在情感控制上显著优于base版本，尤其是Sad、Angry、Surprised、Disgusted等情感。

#### 4.7 作为数据生成器

**Table 11: 使用CosyVoice生成数据训练ASR（Librispeech）**

| Training Data | dev_clean | dev_other | test_clean | test_other |
|--------------|-----------|-----------|------------|------------|
| Librispeech | 2.77 | 5.84 | 2.79 | 5.97 |
| Syn on LS text | 2.79 | 6.37 | 3.00 | 6.59 |
| LS + Syn on LS text | 2.44 | 5.52 | 2.56 | 5.68 |
| LS + Syn on LS, MLS text | **1.93** | **4.43** | **2.04** | **4.53** |

**关键发现**：
- 仅用合成数据可达到与真实数据相当的效果
- 合成数据与真实数据结合可显著提升ASR性能
- 引入MLS文本的多样性比单纯增加时长更重要

---

### 5. **conclusion**

#### 5.1 核心贡献总结
1. ==**监督语义token（S3）**：首次将监督学习引入TTS语音token，通过ASR编码器+VQ获得，内容一致性和说话人相似度显著优于无监督token==
2. **CosyVoice架构**：LLM + 条件流匹配，无需音素器和强制对齐器，支持零样本上下文学习和跨语言克隆
3. **可扩展性**：大规模数据（17万小时）训练可显著提升性能，达到人类水平
4. **指令控制能力**：CosyVoice-instruct支持说话人身份、说话风格、副语言特征控制