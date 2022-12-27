---
layout: default
---
# AdaGrad

英文：Adaptive Gradient

Reference:
- Deep Learning, Ian Goodfellow and Yoshua Bengio and Aaron Courville
- https://zhuanlan.zhihu.com/p/150113660

## 伪代码

**Require**: Global learning rate $\epsilon$  
**Require**: Initial parameter $\theta$  
**Require**: Samll Constant $\delta$, perhaps $10_{-7}$, for numerical stability  
&ensp;&ensp;&ensp;&ensp;Initialize gradient accumulation variable $r=0$  
&ensp;&ensp;&ensp;&ensp;**while** stopping criterion not met **do**  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Sample a minibatch of $m$ examples from the trianing set $\{x^1,...,x^m\}$ with corresponding targets $y^i$  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Compute gradient: $g\leftarrow \frac{1}{m}\nabla\theta \sum_i L(f(x^i;\theta),y^i)$  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Accumulate squared gradient: $r\leftarrow r + g\bigodot g$  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Compute update: $\Delta\theta \leftarrow - \frac{\epsilon}{\delta + \sqrt{r}} \bigodot g$ (Division and square root applied element-wise)  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Apply update: $\theta \leftarrow \theta + \Delta\theta$  
&ensp;&ensp;&ensp;&ensp;**end whille**  

## 思想
独立地适应所有模型参数的学习率，缩放每个参数反比于其**所有梯度历史平均值总和的平方根**。具有代价函数最大梯度的参数相应地有个快速下降的学习率，而具有小梯度的参数在学习率上有相对较小的下降。  
你已经更新的特征（幅度）越多，你将来更新的就越少，这样就有机会让其它特征(例如稀疏特征)赶上来。用可视化的术语来说，更新这个特征的程度即在这个维度中移动了多少，这个概念由梯度平方的累积和表达。稀疏特征的平均梯度通常很小，所以这些特征的训练速度要慢得多。  
这个属性让AdaGrad（以及其它类似的基于梯度平方的方法，如RMSProp和Adam）更好地避开鞍点。Adagrad将采取直线路径，而梯度下降（或相关的动量）采取的方法是“让我先滑下陡峭的斜坡，然后才可能担心较慢的方向”。  
假定有一个多分类问题，$i$表示第i个分类，$t$表示第t次迭代，同时也代表分类i出现的次数。其中，$g_{t,i}=\Delta J(W_{t,i})$表示t时刻，指定分类i，代价函数J关于W的梯度。
$$
W_{t+1}=W_t-\frac{\eta_0}{sqrt{\sum_{v\prime=1}^t(g_{t\prime,i})+\epsilon}}
$$
从表达式可以看出，对出现比较多的类别数据，Adagrad给予越来越小的学习率，而对于比较少的类别数据，会给予较大的学习率。因此Adagrad适用于数据稀疏或者分布不平衡的数据集。Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；缺点在于，随着迭代次数增多，学习率会越来越小，最终会趋近于0。  
