
[讲解视频](https://www.bilibili.com/video/BV1JY411q72n/?spm_id_from=333.1387.homepage.video_card.click&vd_source=a99ca0f9acb7c542b9d10a0bcc1917e2)
### 一、 香农熵（Shannon Entropy）

**香农熵是对概率分布中“预期信息量（Expected amount of information）”的度量，同时也是对系统不确定性（Uncertainty）的衡量**。

- **数学定义**：对于离散概率分布，熵 $H(p)$ 被定义为： $$H(p) = \sum p_i I_i^p = \sum p_i \log_2\left(\frac{1}{p_i}\right) = - \sum p_i \log_2(p_i)$$ _(注：连续情况下使用积分代替求和)_。
- **直观性质**：
    - 当概率密度函数**越均匀时，系统越随机，香农熵越大**。
	    - 例如，对于一枚绝对公平的硬币（正反面概率均为0.5），其香农熵为 $1$。
    - 当概率密度函数**越集中时，系统确定性越高，香农熵越小**。
	    - 例如，当硬币正面概率为0.2，反面为0.8时，其香农熵下降为 $0.72$。

### 二、 交叉熵（Cross Entropy）

**交叉熵衡量的是：在给定“估计概率分布（Estimated probability distribution，记为 $q$）”的情况下，对“真实概率分布（Ground truth probability distribution，记为 $p$）”预期信息量的估计**。

- **数学定义**： $$H(p, q) = \sum p_i I_i^q = \sum p_i \log_2\left(\frac{1}{q_i}\right) = - \sum p_i \log_2(q_i)$$
- **计算逻辑解析**：
    1. **期望的计算基准**：因为现实中的数据总是根据“真实概率分布 $p$”出现的，所以数学期望（Expectation）必须在 $p$ 上进行计算。
    2. **信息量的计算基准**：由于我们使用的是估计值，因此单次事件的“信息量（Amount of information）”是基于“估计概率分布 $q$”计算的（即 $\log_2(1/q_i)$）。

### 三、 KL散度 / 相对熵（Kullback-Leibler Divergence）

>[!hint]
>**KL散度是一种定量衡量两个概率分布之间差异的方法**，在数学上等于**交叉熵与香农熵的差值**。

- **推导与公式**： $$D(p||q) = H(p, q) - H(p) = \sum p_i I_i^q - \sum p_i I_i^p = \sum p_i \log_2\left(\frac{1}{q_i}\right) - \sum p_i \log_2\left(\frac{1}{p_i}\right) = \sum p_i \log_2\left(\frac{p_i}{q_i}\right)$$。
- **重要数学性质**：
    1. **非负性（吉布斯不等式 Gibbs inequality）**：$D(p||q) \ge 0$。**当且仅当两个分布完全相同时，KL散度才等于0**。
    2. **非对称性**：$D(p||q) \neq D(q||p)$。因此，KL散度不能被视为真正的“距离度量（Distance metric）”。
- **在机器学习优化中的应用**：在机器学习参数估计（如参数为 $\theta$ 的分布 $q_\theta$）中，**最小化KL散度有时等价于最小化交叉熵损失函数**。因为真实分布的熵 $H(p)$ 是一个常数，其对参数 $\theta$ 的梯度为0（即 $\nabla_\theta H(p) = 0$），所以 $\nabla_\theta D(p||q_\theta) = \nabla_\theta H(p, q_\theta)$。

### 四、 KL散度的另一层直观解释

KL散度还可以从“序列概率（Probability of sequences）”的角度来理解。**相似分布产生的序列的概率应该是相似的，反之亦然**。 通过对抛硬币序列的数学推导，可以将KL散度等价理解为：基于真实分布 $p$ 生成的一组特定数据序列，在“分布 $p$ 下发生的概率”与在“分布 $q$ 下发生的概率”的比值的对数期望： $$D(p||q) = \log\left(\frac{P(\text{sequence of distribution } p | \text{distribution } p)}{P(\text{sequence of distribution } p | \text{distribution } q)}\right)$$。

### 五、 核心结论

1. **信息量**是一个事件对数概率的倒数（inversion of log probability）。
2. **香农熵**衡量的是一个概率分布的**预期信息量**。
3. **交叉熵**衡量的是给定估计分布后，对预期信息量的估计。**交叉熵始终大于或等于香农熵**。
4. **KL散度**衡量两个概率分布之间的差异，可理解为同一序列在两种不同分布下的概率差异。它是机器学习和深度学习概率模型中的核心概念，与交叉熵损失函数密切相关。