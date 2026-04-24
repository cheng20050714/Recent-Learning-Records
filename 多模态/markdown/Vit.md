# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

### 1. **title：**
- 把一张图像切分成一个个 $16 \times 16$ 像素的小图块（patches），然后把这些图块当作自然语言处理中的Token来对待 。
- Transformer 模型同样能在超大规模的图像识别任务上起到很好的效果

### 2. **abstract：**
- **纯 Transformer 架构：** 作者提出，我们根本不需要依赖 CNN 。只要把图像分割成图块序列，直接喂给一个纯粹的 Transformer 模型，就能把图像分类任务做得非常好 。

### 3. **intro & related work**
- *写作逻辑：背景 -> 遇到了什么瓶颈 -> 怎么破局 -> 结果*
	- **主流范式**：在海量文本上进行预训练（Pre-train），然后在特定的小数据集上微调（Fine-tune）
	- **核心优势**：With the models and datasets growing, there is still no sign of saturating performance.
	- VIT只需要讲图片分割为batch就可以当作token提供给transformer，利用**监督学习**进行图像分类任务
	- **数据量**：在常规数据量下，ViT 打不过 CNN，原因是它没有 CNN 天生的“归纳偏置”；数据量一旦足够大，Transformer 自己学到的规律，足以碾压 CNN。



### 4. **method**

> In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible.An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.



![[image-126.png]]


- **预处理阶段 (Preprocessing)**
    
    - **图像切块与展平 (Patch Extraction & Flattening):** 将原始尺寸为 $224 \times 224 \times 3$ 的图像，按照 $16 \times 16$ 的像素大小切块，得到 $N = 224^2 / 16^2 = 196$ 个图块序列 。每个图块在通道维度展平后，物理大小确为 $16 \times 16 \times 3 = 768$ 。
        
    - **线性映射 (Linear Projection / Patch Embedding):** _（注：这里需要严格区分“物理像素大小”和“模型隐层维度”）_ 展平后的 768 维向量会通过一个可训练的线性投影层映射到模型的恒定隐藏层维度 $D$ 。在 ViT-Base 变体中，设计设定的隐层维度 $D$ 恰巧也是 768 。此时序列形状变为 `196 x 768`。
        
    - **追加分类向量 (Class Token):** 在序列的起始位置，拼接一个额外的、可学习的 `[class]` embedding 。由于加入了一个 Token，序列有效长度变为 $196 + 1 = 197$ 。此时序列形状为 `197 x 768`。
        
    - **加入位置编码 (Position Embedding):** 为了让模型感知图块的原始空间拓扑关系，向这 197 个 token 中逐元素相加（Add）标准的、可学习的 1D 位置编码 。最终输入给 Transformer Encoder 的张量形状精确维持在 `197 x 768` 。
    
- **多头自注意力层 (Multi-Head Self-Attention, MSA)**
    
    - **层归一化 (Layer Normalization):** 在进入注意力机制进行 Q、K、V 计算之前，序列会先统一经过 LayerNorm 处理 。
        
    - **多头拆分:** 以 ViT-Base 为例，隐藏维度 $D = 768$ 被分配给 12 个平行的注意力头 (heads) 。为了保持计算量和参数量在改变 head 数量时恒定，每个头处理的降维特征大小设为 $D_h = D / k = 768 / 12 = 64$ 。
        
    - **注意力计算与拼接:** 这 12 个头分别独立进行自注意力计算，每个头计算出 `197 x 64` 的特征 。将它们拼接 (Concat) 之后，再通过一个多头线性投影矩阵 ($U_{msa}$) 进行融合映射 。最终输出的形状严格对齐输入，依然是 `197 x 768`，这一设计完美契合了随后的残差连接 (Residual Connection) 需求 。

- **消融实验**
	- **位置编码**：实验结果显示，如果不加位置编码，模型性能会大跌；但只要加了位置编码（不管是用 1D 还是 2D），性能几乎没有差别 。
	- **分类头**：作者引入了一个专门的 `[class]` token 来做分类 。但普通的 CNN 通常是用全局平均池化（GAP）来做分类的 。作者去掉了 `[class]` token 换成 GAP发现只要学习率调整好，两者的表现差异不大 。

- **归纳偏置**
	- CNN：归纳偏置、平移等变性
	- vit：所有位置信息必须从头开始学习


#### 5. **experiment**

### 6. **conclusion**
- 论文证明了直接将 Transformer 应用于图像识别是完全可行的 。除了最开始的“切块”操作外，模型没有引入任何专门针对图像的 **“归纳偏置”**（比如 CNN 中特有的局部性和平移等变性）。
	- 优点：框架简单、“相对便宜”
	- vit更大可能有更好的结果 vit-G

> **归纳偏置**
> 模型在学习数据之前就“自带”的假设或偏好，用来帮助模型从有限数据中进行合理推断。

