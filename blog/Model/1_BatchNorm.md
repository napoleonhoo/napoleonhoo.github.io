# Batch Norm：批标准化

本文的主要内容来源于对 Batch Norm 论文的研读，原文见：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)

*注：论文充斥着大量的数学分析，限于作者水平，本文只摘取部分重点。*

## 1 要解决的问题

深度学习训练过程，大量使用了 SGD（Stochastic Gradient Descent，随机梯度下降）算法，已被证明是行之有效的训练方法。但它需要仔细调整模型的超参数，特别是学习率、模型参数的初始值等。由于每一层的输入都会受到前一层参数的影响，训练过程变得复杂。因此，随着网络深度的增加，网络参数的细微变化会被放大。层输入分布的变化会带来问题，因为层需要不断适应新的分布。当学习系统的输入分布发生变化时，就会发生**协变量偏移**（covariate shift）。

我们把训练过程中，深度神经网络中内部节点的分布变化，成为**内部协变量偏移**（internal covariate shift）。消除它有望更快的训练。本文提出了一种新的机制，称为**批标准化**，是一种减少内部协变量偏移的方法。

- 它显著加快了深度神经网络的训练速度。它通过固定层输入的均值和方差来达到这个目的。
- 批标准化通过减少梯度对参数规模和初始值的依赖，对网络里面的梯度也有益处。这使得我们可以提高更高的学习率还不用担心不收敛的风险。
- 批标准化使模型正则化，减少对 Dropout 的使用。
- 批标准化通过阻止模型陷在饱和模式，使得模型使用饱和非线性成为可能。

## 2 根据 Mini-Batch（小批量） 统计数据进行标准化

批标准化变换（BN Transform）算法如下：

$$
\begin{aligned}
&\textbf{Input:} \quad \text{Values of } x \text{ over a mini-batch: } \mathcal{B} = \{x_1, \ldots, x_m\}; \\
&\quad\quad\quad\quad \text{Parameters to be learned: } \gamma, \beta \\
&\textbf{Output:} \quad \{y_i = \mathrm{BN}_{\gamma,\beta}(x_i)\} \\
\\
&\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i && \text{// mini-batch mean} \\
&\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2 && \text{// mini-batch variance} \\
&\hat{x}_i \leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} && \text{// normalize} \\
&y_i \leftarrow \gamma \hat{x}_i + \beta \equiv \mathrm{BN}_{\gamma,\beta}(x_i) && \text{// scale and shift}
\end{aligned}
$$

注意以下几点：

- 它以一个 mini-batch 为基础，计算均值和方差。
- 标准化（normalize）后，得到一个标准化的数据。
- 使用一个线性变换得到的结果就是 BN 变换之后的结果。

### 2.1 BN 网络的训练

为了在网络中使用 BN，我们指定一个激活子集，并根据上面的算法为每个激活子集插入 BN 变换。下面的算法总结了训练 BN 网络的过程。

$$
\begin{aligned}
&\textbf{Input:}   \quad \text{Network } N \text{ with trainable parameters } \Theta; \\
&\quad \quad \quad \quad \text{subset of activations } \{x^{(k)}\}_{k=1}^K \\
&\textbf{Output:} \quad \text{Batch-normalized network for inference, } N_{\mathrm{BN}}^{\mathrm{inf}} \\
\\
&1:\quad N_{\mathrm{BN}}^{\mathrm{tr}} \leftarrow N && \text{// Training BN network} \\
&2:\quad \mathbf{for}\ k = 1 \ldots K\ \mathbf{do} \\
&3:\quad \quad \text{Add transformation } y^{(k)} = \mathrm{BN}_{\gamma^{(k)}, \beta^{(k)}}(x^{(k)}) \text{ to } N_{\mathrm{BN}}^{\mathrm{tr}} \text{ (Alg. 1)} \\
&4:\quad \quad \text{Modify each layer in } N_{\mathrm{BN}}^{\mathrm{tr}} \text{ with input } x^{(k)} \text{ to take } y^{(k)} \text{ instead} \\
&5:\quad \mathbf{end\ for} \\
&6:\quad \text{Train } N_{\mathrm{BN}}^{\mathrm{tr}} \text{ to optimize the parameters } \Theta \cup \{\gamma^{(k)}, \beta^{(k)}\}_{k=1}^K \\
&7:\quad N_{\mathrm{BN}}^{\mathrm{inf}} \leftarrow N_{\mathrm{BN}}^{\mathrm{tr}} && \text{// Inference BN network with frozen parameters} \\
&8:\quad \mathbf{for}\ k = 1 \ldots K\ \mathbf{do} \\
&9:\quad \quad \text{// For clarity, } x \equiv x^{(k)},\ \gamma \equiv \gamma^{(k)},\ \mu_{\mathcal{B}} \equiv \mu_{\mathcal{B}}^{(k)},\ \text{etc.} \\
&10:\quad \text{Process multiple training mini-batches } \mathcal{B}, \text{ each of size } m, \text{ and average over them:} \\
&\quad \quad \mathbb{E}[x] \leftarrow \mathbb{E}_{\mathcal{B}}[\mu_{\mathcal{B}}] \\
&\quad \quad \mathrm{Var}[x] \leftarrow \frac{m}{m-1} \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2] \\
&11:\quad \text{In } N_{\mathrm{BN}}^{\mathrm{inf}}, \text{ replace the transform } y = \mathrm{BN}_{\gamma, \beta}(x) \text{ with} \\
&\quad \quad y = \frac{\gamma}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot x + \left( \beta - \frac{\gamma \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \right) \\
&12:\quad \mathbf{end\ for}
\end{aligned}
$$


### 2.2 BN 网络的推理

激活函数的标准化依赖于 mini-batch，使得训练高效。但是，在推理过程中，这既无必要又不应该。我们希望输出仅取决于输入。当网络被训练好后，我们使用标准化

$$
\hat{x} = \frac{x - \mathbf{E}[x]}{\sqrt{\operatorname{Var}[x] + \epsilon}}
$$

使用的是总体统计数据，而非 mini-batch 的统计数据。因为在推理中，均值和方差都是固定的，标准化就相当于一个线性变换，应用到每一个激活函数后。所以，可以使用伸缩参数$\gamma$和偏移参数$\beta$，来产生一个线性变换，来代替$\operatorname{BN}(x)$。
