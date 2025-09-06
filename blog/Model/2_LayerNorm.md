# LayerNorm：层标准化

本文的主要内容来源于对论文[Layer Normalization](https://arxiv.org/pdf/1607.06450)的研读。

*注：文中涉及较大量的数学证明，鉴于编者水平与篇幅，本文只总结文章重点部分*。

## 1 摘要

LayerNorm 主要来源于对 BatchNorm 的改进，*如果有对 BatchNorm 还不太了解的小伙伴，欢迎大家参考我的另一篇文章：[《BatchNorm：批标准化》](./1_BatchNorm.html)*。

BatchNorm 可以显著地降低前馈神经网络（feed-forward neural network）的训练时间。但是，BatchNorm 的效果依赖于 batch size 的选择，并且不能够很直接的应用于 RNN（循环神经网络，recurrent neural network）。

LayerNorm 对 BatchNorm 做了变换：标准化时，LayerNorm 使用在单次训练过程中，对一个层的神经元的所有的输入计算均值和方差。像 BatchNorm 一样，我们也给每一个神经元一个自己的自适应的偏移（bias）和增益（gain），这些作用在标准化后，非线性之前。和 BatchNorm 不同，LayerNorm 在训练和测试时，执行相同的操作。

通过在每一个时间步上，分别计算标准化数据，就可以直接应用在 RNN 上。优势主要在于：
- LayerNorm 对于稳定 RNN 中的 hidden state dynamics 十分有效。
- 可以减少训练时间。

## 2 背景：BN 简述

令：
- $l^{th}$：前馈神经网络的第 $l$ 层
- $a^l$：第 $l$ 层的输入相加得到的 vector
- $W^l$：第 $l$ 层的权重矩阵
- $h^l$：从下至上的输入
- $f(\cdot)$：element-wise 的非线性函数
- $w_i^l$、$b_i^l$：第 $l$ 层的第 $i$ 个隐藏单元对应的权重与偏移
有：
$$
a_{i}^{l}=w_{i}^{l}{}^{\top}h^{l}\qquad h_{i}^{l+1}=f(a_{i}^{l}+b_{i}^{l})\qquad(1)
$$

深度学习中一个重大的挑战就是权重的梯度会极大地依赖于上一层神经元的输出。BatchNorm 就是为了解决这种“变量偏移”而提出的。它对训练过程中的每一个隐藏单元的输入进行标准化。对于，第 $l$ 层的第 $i$ 个输入和（summed input），BatchNorm 根据数据分布的方差进行缩放：
$$
\bar{a}_{i}^{l}=\frac{g_{i}^{l}}{\sigma_{i}^{l}}\left( a_{i}^{l}-\mu_{i}^{l}\right)\qquad\mu_{i}^{l}=\underset{\mathbf{x}\sim P(\mathbf{x})}{\mathbb{E}}\left[ a_{i}^{l}\right]\qquad\sigma_{i}^{l}=\sqrt{\underset{\mathbf{x}\sim P(\mathbf{x})}{\mathbb{E}}\left[\left( a_{i}^{l}-\mu_{i}^{l}\right)^{2}\right]}\qquad\qquad(2)
$$
其中：
- $\bar{a}_{i}^{l}$：标准化了的第$l$层的第$i$个输入和
- $g_i$：增益参数，用于在非线性激活函数之前缩放标准化激活函数

注意到这里的期望是根据整个训练数据集的分布得到的，一般根据公式(2)进行精确计算不太现实，因为它需要使用当前的权重在整个训练数据集上进行一般前向传播。所以，$\mu$ 和 $\sigma$ 都是使用当前的一个 mini-batch 的数据来进行估计的。**这就对 mini-batch 的大小产生了限制，并且很难应用到 RNN 上。**

## 3 LayerNorm 的核心内容

注意到一个层的输出会对下一层输入和造成高度相关地影响，特别是 ReLu 单元的输出可以变化很大。**这表示“变量偏移”问题可以通过在每一层内，固定输入和的均值和方差，来减轻。** 所以，我们通过下面的公式来计算同一层的隐藏单元的 LayerNorm 的统计数据：
$$
\mu^{l}=\frac{1}{H}\sum_{i=1}^{H}a_{i}^{l}\qquad\sigma^{l}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}\left( a_{i}^{l}-\mu^{l}\right)^{2}}\qquad(3)
$$
其中：
- $H$：一层中的隐藏单元的个数

公式(2)和公式(3)的区别是 LayerNorm 下，一层中的所有隐藏单元共享相同的标准化参数 $\mu$ 和 $\sigma$，但是不同的训练过程的标准化参数是不同的。和 BatchNorm 不同，**LayerNorm 对 mini-batch 的大小没有任何限制，可以被应用到 batch size 为 1 的纯线上环境。**

### 3.1 使用 LayerNorm 的 RNN

对 NLP 任务，不同的训练过程有不同的句子长度是很常见的。使用 RNN 可以很好解决这个问题，因为在每个时间步里面使用相同的权重。但是，当我们在 RNN 中使用 BatchNorm 时，我们需要在一个序列里面分别计算和存储每一个时间步的统计数据。当一个测试序列的长度比任何一个训练序列的长度都要长时，这就很有问题了。LayerNorm 不会有此类问题，因为它的标准化参数只取决于当前时间步的一个层的输入和。它也只有一个集合的增益和偏移参数，在每一个时间步里面共享。

在标准 RNN 中，一个层的输入和使用当层的输入 $\mathbf{x}^t$ 的和前一个隐藏状态的 vector $\mathbf{h}^{t-1}$，利用公式 $\mathbf{a}^t = W_{hh}h^{t-1} + W_{xh}\mathbf{x}^t$ 来计算。使用 LayerNorm 的 RNN 层，使用如下的方法进行标准化，来对激活函数重新平易近人和重新缩放：
$$
\mathbf{h}^{t}=f\left[\frac{\mathbf{g}^{t}}{\sigma^{t}}\odot\left(\mathbf{a}^{t}-\boldsymbol{\mu}^{t}\right)+\mathbf{b}\right]\qquad\boldsymbol{\mu}^{t}=\frac{1}{H}\sum_{i=1}^{H}a_{i}^{t}\qquad\sigma^{t}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}\left( a_{i}^{t}-\boldsymbol{\mu}^{t}\right)^{2}}\qquad(4)
$$
其中：
- $W_{hh}$：隐藏权重
- $W_{xh}$：从下至上对权重的输入
- $\odot$：两个 vector 的 element-wise 的乘法（点乘）
- $\mathbf{b}$、$\mathbf{g}$：偏移和增益参数

在标准 RNN 中，每一个时间步中，网络中单元输入和的平均维度总是趋近于扩大或缩小，导致梯度的爆炸或消失。在使用 LayerNorm 的 RNN 中，**标准化参数使得对所有的输入和的缩放有了不变性，因此也使得 hiden-to-hidden dynamics 更加稳定。**

## BatchNorm v.s. LayerNorm

这里用下面的表格简单总结一下 BatchNorm 和 LayerNorm 的对比。

|          | BatchNorm            | LayerNorm        |
| :------: | :------------------: | :--------------: |
|  归一化维度   | 单样本层内所有特征            | batch 内所有样本的同一特征 |
| 适用 batch | 支持较小 batch           | batch size 大     |
|   适用网络   | RNN、Transformer、在线学习 | CNN              |

## 英语学习

- NSERC：National Science and Engineering Research Council of Canada，加拿大自然科学与工程委员会
- CFI：Canada Fund for Innovation，加拿大创新基金
- saturate：使饱和
- intersperse：散布，点缀，散置
- affine：仿射的，拟似的，亲合的