# DDPM 算法学习笔记

## 1. 整体直观理解

- **核心思想**：给图片加噪是容易的，但从噪声中恢复图片是困难的。DDPM 的核心就是**从正向逐步加噪的过程中，学习其逆向的去噪过程**，最终实现从纯噪声生成真实图片。整个过程通常需要 1000 步。
    
- **采样机制**：无论是给图片加噪还是去噪，都是从一个正态分布（高斯分布）中进行采样。
    ![[截屏2026-04-02 10.12.12.png]]
- **过程对比**：
    
    - **前向过程（加噪）**：是**确定性**的采样过程，逐步向图片中添加高斯噪声。
    - **反向过程（去噪）**：是利用**神经网络**预测出正态分布的均值和方差，然后再从中进行采样以去除噪声。
        

## 2. 正向加噪过程 (Forward Process)

- **最终目标**：给图片逐步加噪声，让图片的每个像素最终都变成从 **均值为0，方差为1的标准正态分布 $\mathcal{N}(0, \mathbf{I})$** 中的采样。
    
- **参数设计与优化**：
    
    - 如果简单地使用 $x_t = x_0 + t\beta\cdot\epsilon$ 进行加噪，会存在两个问题：随着步数 $t$ 增加，均值始终是 $x_0$ 而非 0，且方差会一直无限增大。
        
    - **解决方案**：引入一个随时间递增的方差控制参数序列 $\beta_t$（通常在 0.0001 到 0.02 之间），即 $0 < \beta_1 < \beta_2 < \dots < \beta_T < 1$。保证每一步添加的噪声越来越大，同时对原图进行衰减。
	    - [!] 参数的设置随意，只要能到达最终目标——  **$\mathcal{N}(0, \mathbf{I})$** 即可。
        
- **单步加噪公式**：
    令 $\alpha_t = 1 - \beta_t$，单步的加噪过程可以表示为：
    
    $$q(x_t|x_{t-1}) = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)\mathbf{I})$$
    
- **多步直接加噪 (重参数化技巧)**：
    
    由于独立高斯分布相加的性质（均值相加，方差相加），我们可以跳过中间步骤，直接从原始图像 $x_0$ 一步计算出任意时刻 $t$ 的加噪图像 $x_t$。
    
    令 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，则公式为：
    
    $$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$
    
    $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$
    

## 3. 逆向去噪过程与模型训练 (Reverse Process)

- **反向推导**：我们希望得到逆向概率分布 $q(x_{t-1}|x_t)$，但直接计算是困难的。
    
- **条件贝叶斯公式**：如果在已知原始图像 $x_0$ 的前提下，我们可以通过贝叶斯公式精确计算出上一步的分布 $q(x_{t-1}|x_t, x_0)$：
    
    $$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}$$
    
    经过数学推导，这个真实的后验分布也是一个正态分布，具有确定的均值和方差。
    
- **神经网络的作用**：
    
	逆向过程同样被定义为一个马尔可夫链，但其转移概率是由参数化的神经网络来学习的，起点是标准正态分布 $p(x_T) = \mathcal{N}(x_T; 0, \mathbf{I})$ 。
	
	- **联合分布**：$p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$
	    
	- **单步转移概率**：$p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ _这里 $\mu_\theta$ 和 $\Sigma_\theta$ 是我们需要用神经网络去拟合的均值和方差。_


## 4. 核心优化目标：变分下界 (Variational Bound)

我们通常希望最大化生成真实数据的概率对数 $\log p_\theta(x_0)$。
其中，
$$p_{\theta}(x_{0}):=\int p_{\theta}(x_{0:T})dx_{1:T}$$

由于直接计算很困难，我们转而优化它的**变分下界 (Variational Bound on negative log likelihood)**：

$$\mathbb{E}[-\log p_\theta(x_0)] \le \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] =: L$$

为了让这个公式能够被高效优化，论文利用马尔可夫链的性质和贝叶斯公式，将这个下界 $L$ 进行了重写，转化为多个 KL 散度 (KL divergence) 的和：

$$L = \mathbb{E}_q\left[D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1} D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)\right]$$

这个公式极其重要，它将复杂的优化问题拆解成了三个部分：

1. **$L_T$**：$D_{KL}(q(x_T|x_0) || p(x_T))$。这部分比较的是前向过程的最终状态和标准正态分布。由于前向过程 $\beta_t$ 是固定的常数，这部分没有可学习的参数，可以忽略 。
    
2. **$L_{t-1}$**：$\sum_{t>1} D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$。这是**最核心的训练项**。它要求神经网络预测的逆向分布 $p_\theta(x_{t-1}|x_t)$，要尽可能逼近“在已知原图 $x_0$ 和当前状态 $x_t$ 时，前一步状态 $x_{t-1}$ 的真实后验分布”。
    
3. **$L_0$**：$-\log p_\theta(x_0|x_1)$。这是最后一项重建误差 。
    

---

## 5. 计算真实的后验分布 $q(x_{t-1}|x_t, x_0)$

为了让网络去拟合 $q(x_{t-1}|x_t, x_0)$，我们首先需要知道它的数学表达式。通过贝叶斯公式推导，这是一个易于处理的高斯分布：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t\mathbf{I})$$

其中，它的真实均值 $\tilde{\mu}_t$ 和真实方差 $\tilde{\beta}_t$ 的闭式解为：

- **均值**：$\tilde{\mu}_t(x_t, x_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$
    
- **方差**：$\tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$
    

因为方差 $\tilde{\beta}_t$ 是由固定的超参数组成的常数，所以**神经网络 $p_\theta$ 只需要去拟合均值 $\tilde{\mu}_t$ 即可** 。

---

## 6. 参数化：为什么最终演变成了“预测噪声”？

虽然让网络直接预测均值 $\tilde{\mu}_t$ 是可行的，但论文发现了一种更巧妙的参数化方法。

根据前向过程的性质，我们可以把未知的 $x_0$ 用已知的 $x_t$ 和注入的噪声 $\epsilon$ 表达出来：

$$x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

移项得到 $x_0$：

$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$$

将这个 $x_0$ 代入到上面 $\tilde{\mu}_t$ 的公式中，经过化简，我们得到了一个完全不包含 $x_0$ 的等式：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)$$

这揭示了一个深刻的结论：既然真实的均值是由 $x_t$ 和当前步骤的噪声 $\epsilon$ 决定的，而 $x_t$ 是网络的输入，那么**网络 $\mu_\theta$ 只需要去预测噪声 $\epsilon$ 即可**！

因此，我们将神经网络参数化为：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

(其中 $\epsilon_\theta(x_t, t)$ 就是我们的 U-Net 预测出的噪声 。)

---

## 7. 最终的简化损失函数 (Simplified Objective)

将上述 $\mu_\theta$ 的参数化形式代回原来的 KL 散度公式 $L_{t-1}$ 中，原本复杂的分布比对，化简成了预测噪声与真实噪声之间的均方误差 (MSE)：

$$\mathbb{E}_{x_0, \epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2\right]$$

论文进一步发现，如果**去掉公式前面的复杂权重系数**，不仅能让代码实现变得极其简单，而且能让模型更加关注于困难的去噪步骤（较大的 $t$），从而在实践中产生质量更高的图片 。

最终，DDPM 实际使用的训练损失函数被极大简化为：

$$L_{simple}(\theta) := \mathbb{E}_{t, x_0, \epsilon}\left[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2\right]$$


