# Lecture 2 Bentley Rules for Optimizing Work

# 课程内容

本节课主要介绍了一些常见的优化方法，即“New Bentley Rules”。主要介绍的方法总结如下：

|        | Data Structure              | Logic                            | Loops                         | Functions                  |
| -      | -                           | -                                | -                             | -                          |
| CPU    | Augmentation                | Constant Folding and Propagation | Hoisting                      | Inlining                   |
|        | Precomputation              | Common-Subexpression Elimination | Sentinels                     | Tail-Recursion Elimination |
|        | Compile-Time Initialization | Algebraic Identities             | Loop Unrolling                | Coarsening Recursion       |
|        | Caching                     | Short-Circuiting                 | Loop Fusion                   |                            |
|        | Sparsity                    | Ordering Tests                   | Eliminating Wasted Iterations |                            |
|        |                             | Creating a Fast Path             |                               |                            |
|        |                             | Combining Tests                  |                               |                            |
| Memory | Packing and Encoding        |                                  |                               |                            |
|        | Sparsity                    |                                  |                               |                            |

## 1 Data Structures
### 1.1 Packing & Encoding
- 含义：将数据进行打包或压缩，使其占用更少内存。“reduce memory fetching”
- 优化路线：内存优化。
- 注意：要想到unpacking和decoding的花费。
- 这种设计方式常见于一些名为“ValueAccessor”的类。适用于大规模存储的情况。

### 1.2 Augmentation
- 含义：在数据结构中添加一些多余的信息，来将一些常见的优化便利化。
- 优化路线：CPU耗时优化。
- 举例：在链表中添加tail指针将append操作变得简单。

### 1.3 Precomputation
### 1.4 Compile-Time Initialization
### 1.5 Caching
- 含义：预先计算一些可能会用到的数据。
- 优化路线：CPU耗时优化。
- 只能针对比较少量的数据，因为内存有限。

三者比较：
- Precomputation是指的运行时预计算，在初始化的时候完成的。
- Compile-Time Initialization是指的编程时就已经得到了结果，如Look-up Table等，在运行时查找。
- Caching指的是做计算的时候，存储下来的结果。

备注：对于Compile-Time Initialization还有更近一步的优化方法，即metaprogramming（programs a program that writes programs）。也就是，用编程的方法生成代码，代码的核心点事需要用到的数据（如：Look-up Table）。

### 1.6 Sparsity
- 含义：利用计算数据源的稀疏性进行优化。
- 举例：稀疏矩阵的计算。使用Compressed Sparse Row (CSR) / Compressed Sparse Column (CSC)。
- 优化路线：CPU耗时 & 内存优化。
- 注意：优化后，只要矩阵中有0数据，一般计算耗时上肯定有优化。但具体内存是不是有优化，需要取决于存储方法和矩阵的稀疏度。

## 2 Logic
### 2.1 Constant Folding and Propagation
- 含义：这就是常说的常量优化，类似于`1.4 Compile-Time Initialization`，尽量使用一些显式在编译期确定的数据。

### 2.2 Common-Subexpression Elimination
- 含义：将常见的计算提取出来，只计算一次。
- 优化路线：CPU耗时。

### 2.3 Algebraic Identities
- 含义：利用数学上的相等情况化简计算。
- 优化路线：CPU耗时。
- 举例：将开方的比较转换为取平方的比较。

### 2.4 Short-Circuiting
### 2.5 Ordering Tests
### 2.6 Creating a Fast Path
- 含义：在进行一系列测试（bool计算）的时候，越早结束越好。
- 优化路线：CPU耗时。

三者比较：
- 这三个基本思想差不多。
- Ordering Tests讲的核心点就是安排bool表达式的计算顺序。
- Short-Circuiting和Creating a Fast Path侧重点都是利用判断（如：if）之类的，提前结束计算。

备注：对于Ordering Test来讲，`&&`放在最前面的是容易失败的，`||`放在最前面的是最容易成功的。由于其短路的性质，核心点事越早结束判断越好，执行计算越少越好。

### 2.7 Combining Tests
- 含义：将多个判断语句合成一个，减少判断的数量。本质上是减少branch miss。
- 优化路线：CPU耗时。

## 3 Loops
### 3.1 Hoisting (Loop-Invariant Code Motion)
- 含义：将一个循环中的不变量提取到循环外面来做。
- 优化路线：CPU耗时。
- 注：太简单的情况下，编译器也做得到。

### 3.2 Sentinels
- 含义：利用“哨兵”（或者说dummy value）来标明边界情况，或者处理退出循环时的测试，即是因为啥退出循环的。
- 优化路线：CPU耗时。

### 3.3 Loop Unrolling
- 含义：展开循环。目的是减少循环的次数，进而达到减少循环相关控制语句的执行次数。
- 优化路线：CPU耗时。
- 优点：1. 循环控制指令减少。2. 使得更多的编译器优化成为可能。
- 缺点：过多的循环展开会影响指令cache的使用，即增加其miss率。

### 3.4 Loop Fusion (jamming)
- 含义：相同循环范围内的循环整合成一个。
- 优化路线：CPU耗时。

### 3.5 Eliminating Wasted Iterations
- 含义：去掉没必要的循环次数。
- 优化路线：CPU耗时。

## 4 Functions
### 4.1 Inlining
### 4.2 Tail Recursion Elimination
### 4.3 Coarsening Recursion
- 含义：减少函数调用的开销。
- 优化路线：CPU耗时。

三者比较：
- 核心思想相同，即减少函数调用的开销。
- Inlining是指把被调用的小函数直接写在大函数里面，或者使用关键字`inline`提示编译器将函数进行inline。
- Tail Recursion Elimination是指对于递归函数的尾递归调用去除，直接写代码。
- Coarsening Recursion是将粒度太小的递归调用直接写出来，不再递归。


# TODO
- Jon Louise Bentley, *Writing Efficient Programs*

# 英语学习
- vagary: 变换莫测，出乎意料的变化
- orrery: mechanical model of the solar system.
- &符号念做ampersand
- hoist: 提起，升高
- clobber: 狠击，猛揍
- coarsen: 变粗糙，（使）变厚
