---
layout: default
---
# 协同过滤方法

协同过滤基本方法：
- 计算相似度
- 打分

在推荐系统中，由此衍生出来两种基本方法：
- 基于用户的协同过滤，User-Based Collaborative Filtering，UserCF
- 基于物品的协同过滤，Item-to-Item Based Collaborative Filtering，ItemCF

这两种方法的区别是：UserCF计算用户之间的相似度，而ItemCF计算物品之间的相似度。

## 1 计算相似度的公式
### Jaccard
$$
J(\mu\nu)=\frac{|N_\mu\cap N_\nu|}{|N_\mu\cup N_\nu|}
$$

### cosine
$$
cos=\frac{|N_\mu\cap N_\nu|}{\sqrt{|N_\mu||N_\nu|}}
$$

## 打分
以UserCF为例：  
用以下公式来度量用户对物品的感兴趣程度：
$$
p(\mu,i)=\sum_{\nu\in S(\mu,K)\cap N(i)}Simlarity_{\mu\nu}R_{\nu i}
$$ 
其中，$S_(\mu,K)$代表的是与用户$\mu$最相似的$K$个用户，将与用户$\mu$相似的用户列表按照相似度进行排序就可以得到。$N(i)$代表的是对喜欢物品$i$的用户集合，$Simlarity_{\mu\nu}$代表的是用户$\mu$与用户$\nu$之间的相似度，这个可以直接从用户相似度表中得到。$R_{\nu i}$代表用户$\nu$对物品$i$的兴趣，因为使用的是单一行为的隐反馈数据，所以$R_{\nu i}=1$。对于用户$\mu$最相似的$K$个用户，我们分别计算用户$\mu$与这$K$个用户喜欢的物品集合之间的感兴趣程度，得到用户$\mu$对这$N$个物品的感兴趣程度列表，然后将其逆序排序，取前$m$个物品推荐给用户$\mu$，至此算法结束。