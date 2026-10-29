# Lecture 3 Bit Hacks

# 课程内容
#### 备注：以下并未列出课程全部内容，只是将我理解并觉得用处比较多的地方列了出来。

## Binary Representation
虽然大家早已经知道了二进制原码、反码、补码的意思，不过这里提供了另一种解释方法。很有意思。这里记录下来。
设${x=\{x_{w-1}x_{w_2}...x_0\}}$是一个w位的无符号整数，二进制表示为：
$$
x=\Sigma^{w-1}_{k=0}x_k2^k
$$
有符号整数（补码）的表示是：
$$
x=(\Sigma^{w-2}_{k=0}x_k2^k)-x_{w-1}2^{w-1}
$$
其中，${x_{w-1}}$是符号位。

## Set the kth Bit
- 【问题】将x的第k位设为1.
- 【方法】左移、或。
``` c
y = x | (1 << k)
```
- 【备注】`1 << k`就是第k位的`mask`（掩码）。

## Clear the kth Bit
- 【问题】将x的第k位设为清0.
- 【方法】左移、取反、与。
``` c
y = x & ~(1 << k)
```

## Toggle the kth Bit
- 【问题】将x的第k位取反。
- 【方法】左移、或。
``` c
y = x ^ (1 << k)
```

## Extract a Bit Field
- 【问题】从x中提取第shift位
- 【方法】掩码、右移。
``` c
(x & mask) >> shift
```

## Set a Bit Field
- 【问题】将x的shift位设为y。
- 【方法】掩码取反、或上右移之后的值。
``` c
x = (x & ~mask) | (y << shift)
```

## No-Temp Swap
- 【问题】不用临时变量，将x、y交换。
一般有：
``` c
t = x;
x = y;
y = t;
```
使用二进制方式：
``` c
x = x ^ y;
y = x ^ y;
x = x ^ y;
```
- 【原理】xor的特性：`(x ^ y) ^ y = x`
- 【性能】虽然没有创建临时变量，但是在现代CPU上，不能很好的利用指令级并行（Instruction-Level Parallelism, ILP）。

## No-Branch Minimum
- 【问题】找到x、y中的最小值。
``` c
r = y ^ ((x ^ y) & -(x < y))
```
- 【备注】编译器可能会做得更好。

## Modular Addition
- 【问题】计算`(x + y) mod n`。
``` c
// 基础版
r = (x + y) % n;

// 没有除法，无法预测的分支。
z = x + y;
r = (z < n) ? z : z - n;

// use bit hack
z = x + y;
r = z - (n & -(z >= n));
```

## Least-Significant 1
- 【问题】计算x中最低有效位中的1的掩码。
``` c
r = x & (-x)
```

## Population Count
- 【问题】计算x中的1的个数。
``` c
// 基础版
for (r = 0; x != 0; ++r)
	x &= x - 1;
```
- 【注意】最坏情况下，要循环x的位数次。
- 查表的方法
``` c
static const int count[256] = 
{0, 1, 1, 2, 1, 2, 2, 3, 1, ..., 8};
for (int r = 0; x != 0; x >>= 8)
	r += count[x & 0xFF];
```
表中的数据是256个，记录了每个数（数组的index）对应的1的个数。
- 【相关指令】现代CPU上，统计1的数量的函数，都已经有了相应的指令，比任何人写出的代码都要更快。如在GCC中，有函数：
``` c
int __builtin_popcount(unsigned int x);
```

# TODO

# 英语学习
- \^：读作“caret sign”。
- \~：读作“tilde”。
- caveat：警戒、告诫。