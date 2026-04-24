# **VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts**
### 1. **title** && **abstract**
- 模型结构的优化：提出了一种名为VLMO的统一视觉-语言预训练模型，即模态专家混合（Mixture-of-Modality-Experts, MoME）的Transformer网络架构
- 分阶段预训练策略
### 2. **intro**
`写作思路：现有痛点 -> 本文解决方案 -> 训练策略创新`
- 现有主流框架
	- Dual-encoder 浅层交互 检索任务有效 
	- Fusion-encoder 深度联合编码 VL分类任务有效
- 通过 MoME 在同一个Transformer块内处理图像、文本以及图文对这三种不同的输入模态
- 考虑到两种主流框架善于不同的任务，为了利用二者的优势，MoME 摒弃了单一的前馈网络（FFN），代之以三个模态专家：用于图像编码的视觉专家（V-FFN）、用于文本编码的语言专家（L-FFN），以及用于图文融合的视觉-语言专家（VL-FFN）


### 3. **method**

![[image-157.png|463]]
**Model Architecture**
- **Input Representations**：[[Vit]] 的 Image embedding / [[BERT]] 中的 WordPiece 分词器 / 图像和文本的输入向量在序列维度上 Concatenate 
- **Shared Multi-Head Self-Attention**：无论是输入图像还是文本，自注意力层及其参数在所有模态中都是共享的。这有助于在底层空间对齐视觉和语言的信息特征
- [?] 这里共享 attention 直接输入就可以吗？不需要像[[ALBEF]]一样进行对齐吗

- **Switching Modality Expert**：
	- **专家路由 (Expert Routing)**：$MoME\text{-}FFN$ 会根据当前输入数据的模态（纯图、纯文或图文对）以及当前所处的网络层级（$l$）来自动选择专家 。
	- **V-FFN（视觉专家）**：专门用于处理图像特征
	- **L-FFN（语言专家）**：专门用于处理文本特征
	- **VL-FFN（视觉-语言专家）**：专门负责处理已经过自注意力层初步对齐的、复杂的图文融合特征


 **Pre- Training**
- 网络会根据当前输入数据的三种模态以及网络所在的层数，自动切换并激活对应的专家进行前向传播，借鉴 [[ALBEF]] 的 
$$\mathcal{L}=\mathcal{L}_{itc}+\mathcal{L}_{mlm}+\mathcal{L}_{itm}$$
	进行预训练，在统一的参数框架下同时执行三种截然不同的预训练任务
- [?] 大多数操作都沿袭 [[ALBEF]] 大概是吧？再读一下

**Stagewise Pre-Training**
- 由于高质量图文对数据规模有限，且文本通常是短小简单的标注，直接用其训练容易导致模型泛化能力不足（经典问题，noisy data）

- **阶段 1 (视觉预训练)**：先用海量纯图像数据，采用 BEIT 提出的掩码图像建模任务，把自注意力模块和 V-FFN 来==预热==
    
- **阶段 2 (语言预训练)**：**冻结**刚才练好的自注意力模块和视觉专家，用海量纯文本数据，通过掩码语言建模任务，专门训练 L-FFN
    
- **阶段 3 (视觉-语言预训练)**：用前两步的参数作为初始化，在图文对数据上全面放开，重点学习模态间的对齐与融合
### 4. **experiment**

### 5. **conclusion**