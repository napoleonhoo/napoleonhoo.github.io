# q值的产出

## 1 离线（在线）裁网

离线裁出在线的网络。

根据某个在线需要用到的目标（预估值或向量）裁出网络，和训练网络不同的是：
- 没有label、反向传播、loss计算等
- 输入层不同，如embedding

### 精排网络裁网目标

是一个或多个预估值，这里说的预估值是一个数的向量，离线表示的维度是：`[-1, 1]`。有多个预估值时，它每一个数都是`[-1, 1]`，concat在一起之后是：`[-1, n]`。

### 粗排网络裁网目标

是doc侧向量。离线表示维度是：`[-1, n]`。

## 2 在线预估predictor：输出q值

### 精排

1. 调用`paddle_infer`接口，输入数据，并获取输出`tensor`的数值，以此作为q值。

### 粗排

1. 调用`paddle_infer`接口，输入数据，并获取输出`tensor`的数值，以此作为user侧向量。
2. doc侧向量与user侧向量做点乘，结果作为q值。

### 最终计算

具体有：

$$
qFinal_i = \frac{1}{e^{-qInit_i} + 1}
$$

## 3 调权：Router

最终输出的q值还会在Router（ranking-service）进一步的调权，才是最终预估模块输出的q值。*GR等模块可能还会有进一步的调权。*