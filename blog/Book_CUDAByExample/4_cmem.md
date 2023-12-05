# CUDA by Example 笔记(四): 常量内存与事件

*对应书本第6章*

# 1 大纲

- 常量内存

# 2 内容

## 2.1 常量内存

每个GPU上都有几百个运算单元，一般情况下，一个程序的瓶颈不会是芯片的运算吞吐，而是内存带宽。一些情况下，我们做不到将程序源源不断地输入来保证计算的高速度。所以，如何减少内存的移动呢？

一种新的内存：常量内存（constant memory）。顾名思义，我们用常量内存来保存在kernel执行过程用不会变化的数据。硬件提供了64KB的常量内存，比起使用全局内存，使用常量内存会减少需要的内存带宽。

*备注：书本P101页调用kernel第一个参数应该是`s`，P102页kernel函数漏写第一个参数`Sphere *s`。这里作者应该是和常量内存的写法搞混了。*

### 定义

定义：

``` cpp
__constant__ Sphere s[SPHERES];
```

不许调用如`cudaMalloc`、`cudaFree`来分配、释放内存。对于上述的数组，需要在**编译时**指定长度（即不是动态的长度）。

### 拷贝

``` cpp
cudaMemcpyToSymbol(s, tmp_s, sizeof(Sphere) * SPHERES);
```

需要用`cudaMemcpyToSymbol`而不是`cudaMemcpy`来将CPU上的内存拷贝到常量内存。

### 性能

比起来从全局内存中读取数据，使用常量内存可以节省内存带宽。原因有二：

- 一个从常量内存的读操作可以广播到相邻的线程，节省了15个读操作。
- 常量内存被缓存了，所以连续地从相同的地址读取数据不会产生任何的内存传输。

上述第一条的“相邻”指的是什么意思呢？我们先谈一下warp（线程束）的概念。warp指的是32个线程被“编织”到一起，然后在一个时钟周期内一起执行。在你代码的每一行，warp中的每个线程会在不同的数据上执行相同的指令。

当处理常量内存时，硬件可以将一个内存读操作广播到half-warp（半线程束）。一个half-warp是16个线程。如果一个half-warp内的每个线程都需要同一地址的数据，那么GPU只会发送一次请求并将所有数据广播到每个线程。如果你需要从常量内存中读取很多数据，那么比起来从全局内存读取，仅仅会产生`1/16`（约6%）次读请求。

但是减少的不只是这94%，因为常量内存是不会变的，所以硬件就会将这些数据缓存下来，第一次half-warp读完之后，当另一半的half-warp读取同样的地址的时候，会命中常量缓存，便不会在产生内存传输。*<font color=red>问题3：这个同样的地址要精确到什么程度？每次读数组的整个好，还是分开读好？突然想起来，最好应该是一次性读取16个数据。</font>*

当然，这是一把双刃剑，如果另外的half-warp读的不是同样的地址，这反而会降低性能。允许将一个读操作广播到16个线程的代驾是这16个线程只能一次发送一个读请求。举例来说，当half-warp的16个线程需要的是不同的数据的时候，这16个不同的请求会被串行化，就会需要16倍的读请求时间。如果使用传统的全局内存，这些请求可以同时发出。在这种情况下，从常量内存读取数据反而会更慢一些。

*备注：常量内存不需要作为函数的输入参数传入。*

## 2.2 事件(Event)

## 定义

使用CUDA event API来衡量GPU消耗在一个任务上的时间。

CUDA上的一个event就是一个在用户定义时间的GPU时间戳。

## 使用流程

``` cpp
// 创建
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// 开始
cudaEventRecord(start, 0);

// ……GPU工作代码……

// 结束
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

// 获取human-readble时间
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
printf("Time Elapsed: %3.1fms/n", elapsedTime);

// 销毁
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 事件同步`cudaEventSynchronize`

因为我们在CUDA C里面的一些调用都是**异步的**。举例来说，当我们发起了kernel之后，GPU开始执行代码，但是CPU继续执行程序中的下一行代码。从性能的角度来讲，这确实很好。但是这给计时工作造成了一定的困难。想象一下，`cudaEventRecord()`的调用是将*记录现在时间*的指令放到GPU就绪队列中。所以，这个event不会真的开始记录，直到任何`cudaEventRecord()`调用之前的命令执行完成。从event记录到“正确”的时间的角度上来看，这正是我们想要的。但是在GPU完成这之前的工作和记录下stop事件前，我们不能安全地读取stop事件的值（*这里补充一点，不能安全地读取，是指的在CPU上不能安全地读取。*）。

`cudaEventSynchronize`告诉CPU在一个事件上进行同步。这个函数告诉运行时在GPU没到达stop事件前，阻塞任何的指令。（*这里注意，这个指令是给CPU看的。*）

*其他内容不再赘述，参看代码及注释。*

# 3 总结

本节主要学习了常量内存、事件。

常量内存主要用来减少内存传输。

事件主要用在记录GPU执行时间。

# 4 英语学习

- precipitate：引发
- rasterization：光栅化，栅格化
- translucent：半透明的
- tertiary：第三的
- full-blown：成熟的
- can of worms：一团糟，复杂问题，马蜂窝
- it is worth noting：值得注意的是
