---
layout: default
titile: CUDA by Example
---

# CUDA by Example 笔记(零): 总体介绍

# 1 书本介绍

作者是两名nvidia的工程师Jason Sanders、Edward Kandrot，利用一些比较基础又有应用场景的例子，来介绍cuda编程。主要内容是：

- 【不做介绍】GPU发展、CUDA的安装
- 【见第一节】CUDA C基础：基本概念、kernel调用、参数传递、设备属性
- 【见第二节】CUDA并行编程基础：block、gird的概念、kernel调用的第一个参数
- 【见第三节】线程交互：thread的概念、kernel调用的第二个参数、共享内存、线程同步
- 【见第四节】常量内存、事件
- 【见第五节】纹理内存
- 【不做介绍】书本第8章：“Graphics Interoperability”
- 【见第六节】原子操作、相关数据结构、并行优化等
- 【见第七节】CUDA stream（流）
- 【见第八节】多GPU的使用
- 【见附录一】读书本时的疑问与解答

# 2 通读下来我的感受
## 2.1 优点

- 介绍深入浅出，没有特别深奥，很容易懂。没有上来就甩出来一大堆概念。
- 书本较薄，很容易读完。
- 介绍的内容真的很有应用价值。

## 2.2 缺点
- 作者的C++代码看着很奇怪，可能这和这书本的年代比较久远有关，2010年的，C++11可能还没流行起来。
- 同样地，书本的一些理念、概念可能会有些老旧。
- 书本的很多内容都需要用到画图的功能，OpenGL相关组件，几乎每个例子都有。这对于像我这种只有命令行的用户很麻烦。
- 书本的一些例子也很复杂，涉及到图形图像、物理、数学等，光是理解这个例子在干啥已经很麻烦了。

# 3 关于这系列文章

这系列文章主要讲述了我在学习*CUDA by Example*这书本的时候的总结与体会。

- 我是将PDF打印下来读的，因为这样方便写写画画。（链接见最后）
- 按照惯例，凡是直接学习外语原文的文章，我都会在每节的最后加上相关的**英语学习**的内容。一边学计算机，一边学英语。
- 由于我的环境只有一个命令行，没有图形界面，所以一些OpenGL相关的图形展示都没有具体图像。
- 这个系列的文章并没有cover这书本所有章节。

# 4 一些工具

## 4.1 build脚本

``` shell
#!/bin/bash

set -ex

nvcc $1.cu -o $1.gpu -O0 -g $2
export LD_LIBRARY_PATH=/opt/compiler/cuda-11.1/lib64:/usr/lib64
time ./$1.gpu
```

将这个脚本命名为“build”，并加上执行权限（`chmod +x build`），以后就可以很简单地编译+执行了。比如：`./build add_vec`，就可以编译一个`add_vec.cu`的文件，并且生成一个`add_vec.gpu`可执行文件，并且运行了。这里为了计时加上了`time`命令，实际上没有太大用处。

# 5 相关资源

- [书本PDF下载](https://github.com/hcilab-um/STColorCorrection/blob/master/Docs/Cuda/Jason%20Sanders%2CEdward%20Kandrot%E2%80%94CUDA%20by%20Example%20An%20Introduction%20to%20General-Purpose%20GPU%20Programming%E2%80%942010.pdf)。这个源的PDF是比较好的一版，其他的源现在着缺页现象。
- [书本示例代码](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-)。有人（不太确定是不是官方）将代码传到了网上，方便下载，也可以直接查看。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)。官方文档。
- [CUDA C++ Best Practice Guid](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)。官方文档。
- 参考书：
- 课外书：

# 6 英语学习

- HPC：High Performance Computing
- daunting：令人畏惧的
- compartmentalize：划分
- profound：巨大的；极度的
- countdown：最后的准备