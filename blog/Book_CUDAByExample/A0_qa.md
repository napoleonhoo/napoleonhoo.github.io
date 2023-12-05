# CUDA by Example 笔记(Appendix 0): 疑问与解答

## 内容

这一节的内容是我总结之前文章的问题，并给予解答。并不是对书中"Appendix 0"的笔记，其具体内容参见笔记()。并且，“CUDA by Example 笔记”系列博客从这一节开始，皆是我自己的总结与回顾内容，与原书本内容无直接关系。



### 问题1: 为啥a和b不需要使用`cudaMalloc`分配内存呢？

问题来源：笔记(一)。

更广泛的来讲，这个问题其实是指的cuda kernel的参数传递方式。

问题的解答可以参见[stackoverflow上的问题和回答](https://stackoverflow.com/questions/47485401/when-passing-parameter-by-value-to-kernel-function-where-are-parameters-copied)，还有回答中提到的官方文档[CUDA C Programming Guide, 14.5.10.3 Function Parameters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-parameters)。这里摘录stackoverflow上的回答部分如下：

> ... all arguments to CUDA kernels are passed by value, and those arguments are copied by the host via an API into a dedicated memory argument buffer on the GPU. At present, this buffer is stored in constant memory and there is a limit of 4kb of arguments per kernel launch ...

简单翻译如下：CUDA kernel的所有参数都是传值的，这些参数由host通过API拷贝至GPU上的特定的内存缓冲区。当前，这个缓冲区在常量内存，且每个kernel调用最多是32,764 bytes(约31KB)。*备注：数量上和答主不同，因为答主的版本应该较老，同时参看官方文档可以得到我说的这个值。*



### 问题2: 难道`__global__`只能返回void，`__device__`才能返回int之类的值？

问题来源：笔记(二)

问题的解答可以参见官方文档[CUDA C Programming Guide, 7.1 Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers)。这里摘录一句：

> A `__global__` function must have void return type, and cannot be a member of a class.

即`__global__`的返回值只能是void。

而`__device__`的返回值可以是任意的，如数值、类或结构体、指针等。



### 问题3：同样的地址要精确到什么程度？每次读数组的整个好，还是分开读好？

问题来源：笔记(四)

依我浅见，最好应该是一次性读取16个数据。



### 问题4：`cudaBindTexture`和`cudaUnbindTexture`编译时会提示warning：deprecated，那么应该用啥是最新的函数呢？

问题来源：笔记(五)。

最新版CUDA API对texture有了很大的改动，把它作为一个对象来处理。有兴趣的同学参见[官方文档](问题4：`cudaBind`和`cudaUnbind`编译时会提示warning：deprecated，那么应该用啥是最新的函数呢？)。



### 问题5：它应该有大小限制吧？

问题来源：笔记(七)

锁页内存的大小限制，就是整个系统的可用内存，或者程序的可用内存。



### 问题6：总体看下书中用到“cuda runtime”的地方在哪儿？

问题来源：笔记(七)



### 问题7：意思是至少有一端是device就行，另一端是device或host无所谓？

问题来源：笔记(七)



### 问题8：stream之间有顺序吗？比如说，stream0的拷贝A，必须在stream1的拷贝A之前？

问题来源：笔记(七)

理论上来讲，并不能保证顺序。

回答参考了[NVIDIA论坛上的问答](https://forums.developer.nvidia.com/t/stream-execution-order-in-cuda-exercise/79859)和[stackoverflow上的问答](https://stackoverflow.com/questions/49297293/cudaelapsedtime-with-non-default-streams)。



### 问题9：这个thread同步deprecated了？用啥代替呢？

问题来源：笔记(八)



### 问题10：为啥需要CPU、GPU同步呢？

问题来源：笔记(八)



### 问题11：write-combined是啥意思呢？

问题来源：笔记(八)
