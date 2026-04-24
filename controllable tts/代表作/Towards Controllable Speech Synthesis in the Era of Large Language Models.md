>[!Conclusion]
这篇综述把可控 TTS 的核心问题概括为“如何在不牺牲音质的前提下，精确控制语音的**音高、节奏、情感、音色、风格乃至环境属性**”，并将方法演进总结为两条主线：
>1. `标签/参考音频/文本描述/自然语言指令` 的控制接口升级
>2. `连续声学特征 -> 离散 codec token`、`传统神经 TTS -> LLM/混合架构` 的模型升级。

### **1.基本信息**
- 研究范围：面向 `controllable TTS` 的系统综述，覆盖
- 控制任务：`prosody`、`timbre`、`emotion`、`style`、`language`、`environment`
- 方法视角：`模型架构`、`控制策略`、`特征表示` 、`数据集`、`评测指标`、`挑战与未来方向`
- 论文定位：作者声称这是“第一篇专门面向 controllable TTS 的综合综述”。


> [!info] 核心术语速记
> 
> - **Controllable TTS**：可按指定属性生成语音的 TTS。
>     
> - **Prosody**：韵律，主要含音高、时长、节奏、重音等。
>     
> - **Timbre**：音色，决定“是谁在说话”的主观声纹感。
>     
> - **Style**：更高层的说话方式，如播报、正式、轻松、叙事。
>     
> - **Zero-shot TTS**：不给目标说话人微调，只靠少量参考音频完成克隆。
>     
> - **Voice Cloning**：复制目标说话人的音色特征。
>     
> - **Disentanglement**：把内容、情感、音色、韵律等因素分开编码。
>     
> - **Codec Token**：把语音离散化成 token，便于用语言模型处理。
>     
> - **Prompt-based TTS**：用文本描述或参考音频控制生成。
>     
> - **Instruction-guided TTS**：用自然语言指令统一表达内容和控制目标。
>     
> - **Flow Matching**：近两年很强的生成范式，兼顾质量和速度。
>     
> - **Speech Editing**：不是从头生成，而是按指令修改已有语音片段。
>

### **2. 背景与研究动机**
- 为什么写这篇综述：
	- 过去综述大多关注传统 TTS、声学模型或 vocoder，本质上更关注“能不能生成自然语音”，而不是 **“能不能按用户意图控制生成结果”**。
	- 近两年 diffusion、codec LM、LLM、prompt/instruction-based TTS 快速发展，已有综述跟不上。
- 领域背景：
	- TTS 已从“自然发声”走向“可控、个性化、表达性强”的阶段。
	- 工业界需求显著增加，特别是影视、游戏、机器人、虚拟助手、有声书、跨语种交互。
- 驱动问题：
	- 如何控制**情感、音色、语速、停顿、风格、方言、环境感**。
	- 如何做零样本 voice cloning、风格迁移、可编辑语音生成。
	-  如何让用户直接用自然语言而不是手工调参去控制 TTS。

### **3. 方法与研究分类**

- [!] 分类很清晰的一张图
![[image-216.png]]


| 维度   | 子类                   | 核心思路                 | 代表模型                                       |
| ---- | -------------------- | -------------------- | ------------------------------------------ |
| 模型架构 | 非自回归 Transformer     | 并行生成，强调速度与稳定性        | FastSpeech, FastSpeech 2, FastPitch        |
| 模型架构 | VAE 类                | 用连续隐变量建模风格/情感/韵律     | VAE-Tacotron, Parallel Tacotron, CLONE     |
| 模型架构 | Diffusion 类          | 通过逐步去噪生成高保真语音        | NaturalSpeech 2/3, DEX-TTS                 |
| 模型架构 | Flow/Flow Matching 类 | 可逆映射或流匹配，兼顾速度与质量     | VoiceBox, E2-TTS, F5-TTS, SpeechFlow       |
| 模型架构 | 自回归 RNN 类            | 序列建模强，早期表达性控制代表      | Prosody-Tacotron, GST-Tacotron, MsEmoTTS   |
| 模型架构 | LLM/Codec LM 类       | 把语音 token 化，当作语言建模问题 | VALL-E, VoxInstruct, CosyVoice, VoiceCraft |


| 维度   | 子类                            | 核心思路             | 代表模型                                               |
| ---- | ----------------------------- | ---------------- | -------------------------------------------------- |
| 控制策略 | Reference Speech Prompt       | 用参考语音控制音色/风格     | GenerSpeech, MegaTTS 2, ControlSpeech              |
| 控制策略 | Natural Language Descriptions | 用描述文本指定语音风格      | PromptTTS, InstructTTS, PromptTTS++                |
| 控制策略 | Instruction-Guided            | 用统一自然语言指令控制或编辑语音 | VoxInstruct, CosyVoice, InstructSpeech, Step-Audio |



| 维度   | 子类         | 核心思路            | 代表模型                           |
| ---- | ---------- | --------------- | ------------------------------ |
| 特征表示 | Continuous | 保留细腻声学细节，利于自然度  | mel、latent feature             |
| 特征表示 | Discrete   | 便于和 LLM/多模态模型对接 | EnCodec/FACodec/SQ codec token |

作者还特别强调了一个底层问题：`speech attribute disentanglement`。主要做法有：
- `对抗训练`：把不想要的属性从隐空间中“挤出去”
- `信息瓶颈/分支编码器`：让不同分支分别编码内容、情感、prosody 等
- `KL 正则/量化`：强化因素分离

### **4. 对比与总结**
从控制策略看，作者给出的优缺点很清楚：
![[image-217.png]]

| 方法类                           | 核心思路                              | 优点                | 局限                 | 更适合的场景                       |
| ----------------------------- | --------------------------------- | ----------------- | ------------------ | ---------------------------- |
| Style Tagging                 | 将连续的声音风格特征，转化为离散的、带有明确人类语义的“文本标签” | 简单直接，可控性强         | 只能控制预定义属性，表达空间小    | 音高、语速、能量、基础情感控制              |
| Reference Speech Prompt       | 用参考语音控制音色/风格                      | 个性化强，适合零样本克隆      | 依赖高质量参考音频，控制维度不够直观 | voice cloning、style transfer |
| Natural Language Descriptions | 用描述文本指定语音风格                       | 对用户友好，可读性高        | 描述容易模糊，模型可能误解      | 文本提示驱动的风格/情感控制               |
| Instruction-Guided            | 用统一自然语言指令控制或编辑语音                  | 自由度最高，可统一内容+风格+编辑 | 系统复杂、算力高、依赖 LLM 对齐 | 通用指令式 TTS、语音编辑、多轮交互          |

从架构看：
- `NAR` 更适合部署和高效生成。
- `AR/LLM` 更适合零样本、长上下文和自然语言控制。
- `Diffusion/Flow` 在高保真和表达性上更强，但训练和采样更复杂。
- `Hybrid` 是作者最看好的方向，例如 `CosyVoice` 这类把 `LLM 的指令理解` 和 `flow-based 高保真生成` 结合起来的方法。

作者给出的演化脉络很明确：
- 控制接口：`标签 -> 参考语音 -> 文本描述 -> 指令驱动`
- 语音表示：`连续特征 -> 离散 token`
- 模型形态：`RNN/CNN -> Transformer/VAE -> Diffusion/Flow -> LLM/Hybrid`
- 任务边界：`合成 -> 零样本克隆 -> 语音编辑 -> 多模态/对话式生成`

### **5. 趋势与未来方向**
论文总结的未来方向主要有 4 个：
- `指令驱动的细粒度语音合成与编辑`
- ==`表达性多模态语音合成` ==
- `零样本长语音与情感一致的对话语音合成`
- `大规模数据集自动构建`

当前主要挑战有 3 个：
- `细粒度属性控制难`
	- 情感、prosody、风格往往交织，难以单独精确控制
- `特征解耦难`
	- 改 pitch 往往会连带影响情感和自然度
- `数据集稀缺`
	- 缺少同时覆盖多风格、多情感、多场景、同一说话人多条件变化的数据

值得重点关注的新技术：
- `LLM + flow matching` 的混合式 TTS
- `codec token` 驱动的 speech language model
- `instruction-to-speech` 与 `speech editing`
- `多模态控制`，例如图像/视频/环境联合控制
- `MLLM-based evaluation`，作者甚至用 Gemini 评估 instruction following、naturalness、expressiveness

### **6. 学习与应用**
- 选题不要只做“更自然”，而要明确你想解决哪类“控制接口”问题。
	- 是标签控制？
	- 参考音频控制？
	- 描述式 prompt？
	- 还是统一 instruction？
- 真正有研究价值的问题，集中在 `可解释控制 + 表征解耦 + 长语音一致性 + 用户意图对齐`。
- 实验设计不能只看 `MOS/WER`，还要评估 `instruction following` 和 `expressiveness`。
- 很适合的研究切口：
	- 细粒度情感/韵律解耦
	- 长文本/对话式 controllable TTS
	- 多模态可控 TTS
	- 数据自动标注与指令生成
	- 统一“合成+编辑”的 instruction-based speech model


- 数据集：有 Table 3，整理了 `IEMOCAP, ESD, GigaSpeech, WenetSpeech, TextrolSpeech, Parler-TTS, SpeechCraft` 等。
- 评测：有 Table 4，整理了 `MCD, FDSD, WER, Cosine, PESQ, MOS, CMOS, AB/ABX`。
- 工具/开源底座：有 Table 5，但重点是 `speech tokenizer / codec`，如 `HuBERT, Wav2Vec 2.0, EnCodec, SoundStream, SpeechTokenizer, Mimi Codec, WavTokenizer`，不是完整的 TTS 框架清单。
- 论文列表：[awesome-controllabe-speech-synthesis](https://github.com/imxtx/awesome-controllabe-speech-synthesis)



研究脉络时间线：
- `2016 年前`：规则法、拼接法、HMM，主要做基础 prosody control。
- `2016-2020`：WaveNet、Tacotron、FastSpeech 系列，进入神经 TTS 与显式属性控制阶段。
- `2021-2022`：风格迁移、参考语音 prompt、zero-shot cloning、属性解耦开始成熟。
- `2023-2024`：PromptTTS、VALL-E、NaturalSpeech、VoiceLDM、InstructTTS，描述式/LLM 式控制成为主流。
- `2024-2025`：VoxInstruct、CosyVoice、Step-Audio、VoiceCraft，进入“统一 instruction + 可编辑 + 多模态 + 混合架构”阶段。
