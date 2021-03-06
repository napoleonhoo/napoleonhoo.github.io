---
layout: default
---

# Ex10-I 斐波那契数列

## 题目描述

>写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：
>斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
>
>答案需要取模 1e9+7（1000000007）。

## 问题分析

斐波那契数列大家都知道，不过具体到这道题目，需要注意以下几点：

- 输入是：0<=n<=100
- 答案取模1e9+7

说实话，在我一上来做这道题的时候，没有细心观察到这一点。试一下大家会发现，即便是使用`unsigned long long`也会使直接得到的结果溢出。所以取模这一步不能只在返回的时候做，在**迭代**过程中就要做。代码使用循环，具体如下。

## 代码

```c
int
fib (int n)
{
  if (n == 0 || n == 1)
    return n;
  int a = 0, b = 1, c;
  for (int i = 0; i < n - 1; ++i)
    {
      c = (a + b) % 1000000007;
      a = b;
      b = c;
    }
  return b;
}
```

## 结果

> 执行结果：通过
>
> 执行用时：0 ms, 在所有 C 提交中击败了100.00%的用户
>
> 内存消耗：5.1 MB, 在所有 C 提交中击败了66.39%的用户

得到了一个还算不错的结果吧。