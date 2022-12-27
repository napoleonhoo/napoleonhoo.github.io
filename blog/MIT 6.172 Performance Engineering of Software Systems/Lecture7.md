# Lecture 7 - Races and Parallelism

## 课程内容
本节课简述了一些关于竞争的问题，主要讲述了关于并行度的计算。

## Determinacy Races

当两个逻辑上并行的指令访问同样的内存空间，并且至少有一个指令进行了写操作时，会出现Determinacy Races的情况。

### 竞争的种类

| A     | B     | Race Types |
| -     | -     | -          |
| read  | read  | N/A        |
| read  | write | read race  |
| write | read  | read race  |
| write | write | write race |

### 避免竞争
- 尽量避免在多线程中共享变量。
- 在某些地方，同一个struct中的不同元素也会出现竞争的问题。（和机器字的大小相关）

## 什么是Parallelism（并行度）？

### 计算图
- 一个并行的指令流是一个DAG（有向无环图） $G = (V, E)$
- 每一个顶点$v \in V$是一个strand：一个指令序列。
- 每一个边$e \in E$是一个调用、return、continue等的边。

### Amdahl's "Law"
> 如果你的程序50%的是并行的，50%的是串行的，那么，无论你用多少个核，你不可能达到两倍以上的加速比。  —— Gene M. Amdahl

总结：**总体上来说，如果你的程序有$\alpha$占比的内容必须串行执行，那么加速比最多是$1/\alpha$**。备注：*这只是一个很松的上限*。

### 性能衡量

$T_p$：在P个核心上的执行时间

$T_1$：work（总工作量）

$T_\infty$：span（关键路径、最长路径、计算深度）

Work Law：$T_p \ge T_1/P$

Span Law: $T_p \ge T_\infty$

### Speedup（加速比）
加速比：$T_1 / T_p$
- 当$T_1 / T_p < P$，称为次线性（sublinear）加速比。
- 当$T_1 / T_p = P$，称为完美线性加速比。
- 当$T_1 / T_p > P$，称为超线性（superlinear）加速比。

### 并行度
最大可能的加速比：

$T_1 / T_\infty$：parallelism，the average amount of work per step along the span（在最长路径上每一步的平均工作量）

## 调度理论（Scheduling Theory）

### 贪婪调度
核心思想：每一步都做尽可能多的工作

定义：当一个strand的所有前驱都完成了的时候，那么称其为**准备就绪**。

Complete Step：
- 大于等于P个strand就绪了。
- 运行任意P个strand。

Incomplete Step：
- 小于P个strand就绪了。
- 运行所有的strand。

### 分析贪婪调度算法

> 定理：对于任何一个贪婪调度器有：
> $$
> T_p \le T_1 / P + T_\infty
> $$

### 贪婪调度的最优

> 推论：对于任意一个贪婪调度器的执行时间，是最优调度算法的2倍以内。

### 线性加速比

> 推论：当$T_1 / T_\infty >> P$任何一个贪婪调度器可以达到近乎完美线性的加速比。

定义：$T_1 / (P T_\infty)$成为parallel slackness（松弛度、松紧度）。

一般地，这个值大于等于10比较好，才能抵消掉多线程带来的开销。

## 其他
- ostensibly: 表面上地
- corollary: 推论，必然的结果或结论