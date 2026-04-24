# **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**

1. **title**
- 去除卷积特征/区域性特征，仍能让准确率几乎保持不变，简化流程。
- 移除目标检测器，防止其在提取视觉特征过程中带来的局限性
1. **abstract**
- **当前痛点**：
	- **效率与速度低**：仅仅提取输入图像的特征，就需要消耗比后续多模态交互步骤更多的计算资源。
	- **表达能力受限**：模型的表达能力上限被死死限制在了视觉嵌入器（Visual Embedder）及其预定义的视觉词汇表上。
- **解决方案**：提出了极简VLP模型，将视觉输入处理大幅简化，借鉴nlp领域


3. **intro**
>To be fed into VLP models, image pixels need to be initially embedded in a dense form alongside language tokens.

- Vit 分割为patch传入transformer
- 目标检测 bonding marks

>To this date, most VLP studies have focused on improving performance by increasing the power of visual embedders.
- 因此ViLT聚焦于通过将image快速、轻便embedding

![[image-132.png]]


- **数据增强**：

4. **background**
**4.1 Taxonomy of Vision-and-Language Models**
（1）视觉和语言这两种模态，在专用的参数或计算量上是否对等
（2）两种模态是否在深度网络中交互。

==总结出以下四种类别：==
![[image-133.png]]

VSE
CLIP
VLP
ViLT

**4.2 模态交互模式**
- **单流（Single-stream）**：将图像和文本的输入直接拼接（concat）在一起，让网络层统一处理（如 VisualBERT, UNITER）。
    
- **双流（Dual-stream）**：两种模态在输入层面不拼接，各自走各自的流（如 ViLBERT, LXMERT）。

**4.3 视觉嵌入模式 (Visual Embedding Schema)**
- 最大瓶颈（figure1）是模型的视觉嵌入，**ViLT 的核心突破**是借鉴了 ViT 模型的思路，完全抛弃了目标检测和 CNN 主干网络。它采用最简单的方案：直接把图像切成Patch，然后通过简单的线性投影（Linear projection）转化为向量 。


5. **method**
```shell
模型结构 -> 模型训练 -> 模型训练技巧（whole word masking）/ 数据增强（rand argument）
```

**模型结构：**
![[image-134.png]]
- [?] 这里没太懂输入的处理是怎样的，为什么是三个相加？

**全词掩码（whole word masking）**
- **原因**：如果只掩码长单词被切分后的部分子词（例如 "giraffe" 被切分为 "gi", "##raf", "##fe"，只掩盖 "##raf"），模型可能直接利用相邻的文本 token就能猜出答案，而根本不去看图像 。
    
- **方案**：为了强迫模型充分利用另一种模态（图像）的信息，ViLT 在预训练时以 0.15 的概率将整个单词的所有连续子词全部掩码掉 。

- [?] 疑问：虽然实验验证了这样操作结果就是更优，但是这个假定一定成立吗？


**图像数据增强 (Image Augmentation)**

传统的 VLP 模型因为大多依赖预先提取并缓存好的图像区域特征（Region Features），所以很难在微调阶段使用图像数据增强 。

- ViLT 由于直接处理原始图像块，因此在微调阶段引入了 RandAugment 数据增强策略 。
    
- **细节**：作者去掉了原策略中的“颜色反转（color inversion）”（因为文本常包含颜色描述）和“随机擦除（cutout）”（为了防止删掉图像中微小但关键的物体）。

6. **experiment**
![[image-135.png]]
>多模态数据集 images 对应多个 captions

- [?] 读experiment大概可以读什么呢？

7. **conclusion**
- 表达的是简单的baseline可以达到很好的效果，未来的 VLP 研究不应该再陷入加强单模态特征提取器（比如不断叠加更深、更复杂的 ResNet），而应该把精力集中在 Transformer 模块内部的模态交互机制上。
- 未来的三大方向....