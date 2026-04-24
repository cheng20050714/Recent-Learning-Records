# EmoVoice: 基于 LLM 的自由文本提示情感可控语音合成

- **来源**: arXiv:2504.12867
- **作者**: Guanrou Yang, Chen Yang, Qian Chen, Ziyang Ma, Wenxi Chen, Wen Wang, Tianrui Wang, Yifan Yang, Zhikang Niu, Wenrui Liu, Fan Yu, Zhihao Du, Zhifu Gao, Shiliang Zhang, Xie Chen
- **发表**: ACM MM 2025
#自然语言控制 #情感控制 #音素并行 #组建模

---
## 1. 研究背景与问题

情感可控 TTS 面临的核心瓶颈是**控制粒度与自然度的矛盾**。
已有方法主要依赖离散情感类别标签（如 angry, happy, sad）进行控制，但人类的情感表达具有连续性和高度语境依赖性，粗粒度标签无法捕捉细腻的情感状态（如 "略带讽刺的失望" 与 "愤怒的失望" 的区别）。
同时，==自然语言描述==虽然能表达丰富情感，但现有工作在如何利用自由文本情感描述实现==细粒度控制==方面探索不足。



==已有方法的两个主要缺陷==：
1. **控制信号粗粒度**：类别标签只能覆盖有限情感空间，无法表达强度、混合情感或情境化细微差别
2. **数据稀缺**：高质量、带有细粒度自然语言情感标注的语音数据集极度匮乏，限制了监督学习的可能性

本文的**核心假设**是：如果将 LLM 的强大语言理解能力和自回归 TTS 建模结合，并用自然语言情感描述和自由风格的上下文提示来引导，同时==引入并行的音素（phoneme）token 输出来增强文本一致性==，就能在仅用少量训练数据的情况下实现细粒度的情感可控语音合成。

---
## 2. 方法概述
### 2.1 整体架构
EmoVoice 以 Qwen2.5-0.5B/1.5B 为骨干 LLM，通过扩展词表和增加 LoRA 适配器，使其能够自回归地预测音频 token 序列。
![[Pasted image 20260413162758.png|437]]

1. **核心骨干网络 (Backbone)**
EmoVoice 的基础架构建立在 **Qwen2.5-0.5B** 模型之上 。
- 这是一个包含 24 层 Transformer 层的因果预训练语言模型，参数量约为 4.9 亿 。
- 模型的所有参数在训练阶段都会进行更新 。    
- 利用 LLM 强大的文本语义和情感分析能力，模型不再需要专门的提示词编码器（Prompt Encoder）来理解情感指令 。
    

2. **输入与词表扩展 (Input & Vocabulary)**
- **输入格式**：`<SYSTEM>: Say this sentence with emotion of <Description>. \n <Text>.` 
- **词表扩展**：加入了一个专门用于音频 token 的新代码本 $V_a$，合并为新的词表 $V = V_t \cup V_a$ 。文本 token 的嵌入（embeddings）保持不变，而新加入的音频 token 的嵌入则是随机初始化的 。

 3. **生成机制与组建模 (Generation & Group Modeling)**
为了提高效率，该模型采用了**语义组建模 (Semantic Group Modeling)** 技术：
- 在每一个预测步骤中，模型不会只预测一个 token，而是同时预测一个组（Group）的语义 token 。
- 本模型中组的大小 $G=3$ 。
- [!]  $x_a = logits[..., |V_t|:]$ 强行截取音频 Token，只看 $V_a$ 部分
- ==预测全部词表，但只保留音频部分，完美复用了 LLM 强大的自回归生成能力==
    
4. **声学波形生成 (Acoustic Generation)*

LLM 生成语义 token 后，还需要将其转换为我们能听到的声音波形：

- EmoVoice 使用了 CosyVoice 系统中的 **流匹配模块 (Flow Matching module)** 和 **HiFi-GAN 声码器 (Vocoder)** 来完成这一转换 。
- 在推理时，系统会输入一段来自同一说话人的中立情感音频作为提示词（Prompt），来引导流匹配模块==控制生成语音的音色 ==。


### 2.2 训练策略：两阶段

**阶段一：标准 TTS 预训练**
在 VoiceAssistant（英文）和 Belle（中文）等中性 TTS 数据上预训练，让模型掌握基本的文本到语音映射能力。采用标准的 next-token prediction 目标。

**阶段二：情感 TTS 微调**
- 英文：在 EmoVoice-DB（本文构建的 40 小时合成数据集）和 LAION's Got Talent（真实情感语音数据）上微调
- 中文：在 Seed-TTS 的情感数据子集上微调

微调数据采用 GPT-4o 生成的自由风格情感描述（freestyle natural language emotion descriptions）。输入格式为：
```
Say this sentence with emotion of <Description>. <Text>
```

### 2.3 音素增强变体（Phoneme Boost）

为了缓解 LLM-based TTS 中常见的文本漂移（text drift）和发音错误问题，论文引入了并行==音素 token 预测机制==。具体有四种变体：

![[Pasted image 20260413184034.png|378]]

| 变体 | 输出结构 |
|---|---|
| EmoVoice | 仅音频 token |
| EmoVoice-ST | 文本 + 音频 token 交错 |
| EmoVoice-SP | 音素 + 音频 token 交错 |
| EmoVoice-PT | 文本和音素作为辅助输入，并行预测音频 token |
| EmoVoice-PP | 文本和音素 + 音频 token 并行输出 |

消融实验（Table 6）表明，**EmoVoice-PP** 在情感指标上表现最优，**EmoVoice-PT** 在 WER 上最低。

### 2.4 数据构建：EmoVoice-DB

论文构建了一个高质量的英文情感语音数据集，核心特点：
- **规模**：约 40 小时音频，22,000+ utterances
- **情感覆盖**：六个基础类别（angry, happy, sad, surprised, disgusted, fearful）+ neutral
- **标注粒度**：每个样本配有一段详细的自然语言情感描述（由 GPT-4o 生成），而非简单标签
- **多样性**：包含小说片段、对话、观察性短语等
- **说话人**：覆盖多种音色和说话风格

数据构建流程（三步）：
1. **文本与情感描述生成**：用 GPT-4o 按类别生成文本和详细情感描述
2. **语音合成**：用 GPT-4o-audio 将带情感描述的文本合成为高保真语音
3. **后处理**：计算 WER、情感相似度，过滤低质量样本

---
## 3. 核心概念及实验

### 3.1 自由文本情感提示（Freestyle Text Prompting）

**定义**：用自然语言自由描述来指定待合成语音的情感状态，而非预设的离散类别标签。例如："whispered through the parched cornstalks, its voice fraying like worn silk" 或 "emotionally charged dialogue excerpts representing natural spoken lines"。

**与相关概念的关系**：
- 相较于类别标签（categorical labels），自由文本提示将情感控制从有限离散空间扩展到了连续、高维的自然语言空间
- 相较于风格迁移（style transfer）或参考音频克隆（prompt TTS），自由文本提示不需要提供参考音频，仅通过文本即可精确操控情感
- 本质上是将 LLM 的 instruction-following 能力迁移到了语音合成领域

**消融验证**：
- Table 9 显示，通过对 EmoVoice-DB 的情感描述进行数据增强（用 GPT-4 改写同义描述），WER 从 2.83 降至 2.73，情感相似度和 UTMOS 均有提升，说明模型从多样化的语言描述中学习到了更鲁棒的情感语义映射。**（多对一）**
- Table 5 显示，不同输出结构变体在情感指标上的差异较小，说明自由文本提示的情感控制效果相对稳定，不完全依赖特定的 token 输出方式

**边界**：
- 依赖 LLM 对情感描述的理解能力。若描述过于抽象或与文本语义冲突（如用 "欢快" 描述悼词），模型可能产生不一致结果
- 情感描述的粒度与数据分布强相关。训练数据中未见的情感概念（如特定文化语境下的情感）可能无法准确表达
- [?] 论文未对 **out-of-distribution 的情感描述** 做系统测试，自由文本提示的泛化边界尚不明确

### 3.2 音素增强输出（Phoneme Boost Variants）

**定义**：在 LLM 自回归生成音频 token 的同时，引入音素（phoneme）token 作为并行的预测目标或辅助输入，利用音素与文本的严格对应关系来增强合成语音的文本忠实度（content consistency）。

**与相关概念的关系**：
- 传统 TTS 中，音素序列是文本的前端表示，自回归 LLM-based TTS（如 VALL-E、CosyVoice）通常直接从文本/字符预测音频 token，跳过了显式音素监督
- EmoVoice 将音素重新引入生成流程，但不同于传统 TTS 的两阶段（文本→音素→声学特征），这里音素是 LLM 自回归输出的一部分或多任务目标
- 与 Seed-TTS 的 phoneme-level prompt 不同，EmoVoice 强调的是 **预测阶段** 的音素增强，而非输入阶段

**消融验证**（Table 6）：
- EmoVoice-PT（并行音素作为辅助输入）WER 最低（16.34），但未达到最优情感指标
- EmoVoice-PP（并行音素 + 音频 token 输出）WER 11.68，情感相似度最高（0.9100），说明**在输出层显式预测音素 token 能同时提升情感和文本一致性**
- 作为对比，仅音频 token 的 baseline（EmoVoice）WER 18.07，情感相似度 0.9084——音素增强对 WER 的改善最为显著

**边界**：
- 音素信息主要解决发音准确性和文本漂移问题，对情感自然度的上限提升有限。Table 6 中各变体的 UTMOS 差异很小（4.21-4.29），说明音素增强不直接改善感知质量
- 中文场景中（Table 4），音素的使用可能受限于拼音系统的覆盖度，论文未深入讨论多语言音素统一问题
- [?] 增加了输出序列长度，推理成本和延迟会上升，论文未量化这一开销

### 3.3 EmoVoice-DB：合成情感数据集
**定义**：一个通过 GPT-4o 生成文本/情感描述、再通过 GPT-4o-audio 合成语音的 40 小时英文情感语音数据集，特点是**细粒度自然语言情感描述 + 高表达性语音**。

**与相关概念的关系**：
- 相较于 ESD（Emotional Speech Dataset）等使用真实录音的数据集，EmoVoice-DB 规模更大、标注更细，但语音完全由 AI 合成
- 相较于其他合成数据集（如 InstructSpeech），EmoVoice-DB 强调 **freestyle natural language descriptions** 而非固定模板或类别标签
- 本质上是用"AI 教 AI"的方式扩展情感 TTS 的监督信号

**消融验证**：
- Table 8 中，对比了仅使用真实数据 LAION's Got Talent（约 44.4k 样本）vs. 加入 EmoVoice-DB 后的效果。加入 EmoVoice-DB 后，英文测试集上 WER 从 3.94 降至 2.73，情感相似度从 0.9079 提升至 0.9100，UTMOS 从 4.35 提升至 4.36
- 这表明合成数据不仅扩展了数据量，更重要的是丰富了情感描述的多样性，帮助模型学习更鲁棒的文本-情感-语音映射

**边界**：
- ==合成数据的上界问题==：GPT-4o-audio 合成的语音可能在某些声学细节（如极端情感的声带紧张度）上与真实人类表达存在差异，这可能成为模型性能的上界
- 数据集目前只有英文，中文模型使用了不同的数据来源（Seed-TTS 子集），跨语言一致性未验证
- [?] 论文未对 EmoVoice-DB 和真实录音数据做系统的人类偏好对比，合成数据相对于真实数据的"天花板"未被量化

---
## 4. 核心洞见

**本文的核心认知贡献是：在情感可控 TTS 中，自由文本情感提示可以替代甚至超越离散类别标签，而 LLM 的文本理解能力是实现这一替代的关键媒介；同时，==通过在 LLM 的输出空间中引入音素 token 作为并行监督信号，可以在不牺牲情感表达力的前提下显著提升内容一致性。**==

但如果更严格地审视：后半句（音素增强）确实存在方法论层面的价值——它指出了 LLM-based TTS 中一个此前被忽视的问题（文本漂移），并给出了一种轻量级的多任务学习解决方案。前半句（自由文本提示）则更多是**对 LLM 固有能力的领域迁移应用**，其认知层面的新颖性相对有限。

>[!question]
>论文默认"情感由自然语言描述定义"，但 human perception 研究表明，LLM 对情感的文本描述理解与人类主观感知并不完全一致。这提示一个反转思路：**与其不断优化文本描述→语音的合成链路，不如直接以人类反馈（RLHF）或情感嵌入对比学习来优化**，绕过 LLM 的语义瓶颈。这意味着可以停下对"更巧妙的 prompt engineering"的追求，开始探索 TTS 领域的 reward modeling。


---
## 5. AI审稿

### 5.1 选题价值

情感可控 TTS 是一个真实且活跃的研究方向。从离散标签到自然语言描述的过渡是领域内的自然演进，论文选题具有一定的前瞻性和应用价值。但"用自然语言描述控制语音情感"这一思路在 PromptStyle、InstructSpeech 等工作中已有探索，所以论文需要回答的关键问题是：**相比已有工作，增量贡献在哪里？**

论文的增量主要体现在：
1. 将 LLM（Qwen2.5）作为核心生成模型，而非仅作为控制编码器
2. 引入音素增强机制
3. 构建了新的细粒度数据集 EmoVoice-DB

### 5.2 方法贡献

方法层面的主要贡献是**音素增强变体**和对**LLM 初始化规模的消融**。这些实验设计较为扎实，能够为后续 LLM-based TTS 研究提供参考。

但在创新性上存在局限：
- 整体架构（LLM + VQ token + 自回归生成）与 CosyVoice、Seed-TTS、VALL-E 等高度相似
- 自由文本提示的思路与 PromptTTS2、InstructSpeech 等相近工作有重叠
- [?] 论文未明确说明其与 CosyVoice 的关系。从实现细节看，EmoVoice 大量借鉴了 CosyVoice 的 tokenizer、decoder、training pipeline，这是否应被更明确地定位为"基于 CosyVoice 的扩展工作"？

### 5.3 实验严谨性

**优点**：
- 消融实验比较充分：输出结构变体（Table 5, 6）、LLM 规模（Table 7）、LLM 初始化（Table 8）、数据增强（Table 9）均有覆盖
- 主实验同时包含客观指标（WER, Emo_Sim↑, Recall↑, UTMOS↑）和主观 MOS 评测
- 在英文和中文两个语种上进行了验证

**不足**：
1. **Baseline 选择的公平性问题**：Table 2 对比了 PromptStyle、PromptTTS、CosyVoice、GPT-4o-mini-tts 等方法，但 EmoVoice 使用了 EmoVoice-DB + LAION 的混合训练数据，而 baselines 是否使用了相同数据增强不清楚。如果 baselines 仅在公开数据集上训练，这种对比不公平。
2. **硬案例测试集的客观性问题**：论文自建了一个 hard-case test set（包含绕口令、技术术语、生僻词），并在该测试集上优于 GPT-4o-mini-tts（Table 6）。但硬案例的选取标准和标注过程未详细说明，自封 hard-case 然后自己胜出，说服力有限。
3. **主观评测的样本量**：MOS 评测仅请了 30 名参与者，每人评 10 个样本×6 类情感 = 60 utterances，总计 1800 个评分点。对于 ACM MM 级别的论文，这个样本量偏小。
4. **情感评估指标的可靠性讨论**：论文第 7 节专门讨论了现有情感评估指标与人类感知不一致的问题，这是一个很好的自省。但也暗示了论文报告的情感相似度/Recall 等指标的可靠性本身存疑——既然如此，主实验结论的坚实性也随之打折扣。

### 5.4 写作质量

论文结构清晰，方法描述详细，实验部分组织有序。但存在两个明显问题：
1. **创新性表述的清晰度**：摘要和引言中对"novel"一词使用过多，但实际上核心方法论的原创性并不突出。审稿人可能会质疑 novelty。
2. **与相关工作的区分度**：Related Work 和 Method 部分对 CosyVoice、Seed-TTS 等近缘工作的引用充分，但对"EmoVoice 相比它们的独特之处"的阐述不够尖锐。

### 5.5 判决

**weak accept**

理由：选题有价值，实验相对扎实，数据集建设有贡献，音素增强的消融有参考意义。但方法层面的原创性一般，baseline 对比的公平性存疑，创新性表述略显夸大。

---
## 7. 同领域/同系列对比

| 方面       | EmoVoice                    | PromptTTS / PromptTTS2 | CosyVoice             | InstructSpeech / PromptStyle |
| -------- | --------------------------- | ---------------------- | --------------------- | ---------------------------- |
| **基座模型** | Qwen2.5 LLM                 | 非 LLM (扩散/自回归)         | 非 LLM (flow matching) | LLM / 非 LLM                  |
| **控制信号** | 自由文本情感描述                    | 风格提示文本                 | 无情感控制（原论文）或扩展         | 自然语言指令/风格提示                  |
| **音素监督** | **输出层并行音素预测**               | 无                      | 无显式音素增强               | 无                            |
| **数据集**  | EmoVoice-DB (合成, 40h, NL描述) | 自有数据集                  | AliMeet 等中性数据         | 自有数据集                        |
| **核心增量** | LLM + 自由文本 + 音素增强           | 开创性文本→风格控制工作           | 高质量中性语音合成             | 指令化语音合成                      |

