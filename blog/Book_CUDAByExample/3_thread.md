# CUDA by Example 笔记(三): 线程交互

*对应书本第五章*

# 1 大纲

- kernel调用的第二个参数
- `blockDim`
- 共享内存
- 线程同步

# 2 内容

## 2.1 thread使用

CUDA运行了很多kernel的副本。我们把这些并行的副本称为**block**。CUDA允许这些block被分为**thread**。如上一节的例子中：

``` cpp
add<<<N, 1>>>(dev_a, dev_b, dev_c);
```

第二个参数就是每个block中thread的数量。在本例中，我们共运行了：

`N blocks * 1 thread/block = N parallel blocks`

### 那么既然有了block又需要thread？

因为并行的thread可以做到并行的block做不到的事情。

### 获取每个kernel拷贝的索引的方式

对于上一节中的例子，我们运行了N个block，每个block用了1个thraed。那么对每一个kernel的拷贝有：

``` cpp
int tid = blockIdx.x;
```

而我们使用1个block，每个block用N个thread。那么对每一个kernel的拷贝有：

``` cpp
int tid = threadIdx.x;
```

### thread调用的要求

对于一次调用中的block的数量是有硬件要求的：65535.而对于每个block可以调用的thread数量也是有硬件要求的，即device属性中的`maxThreadsPerBlock`。（*在我的机器上，这个数值是1024。*）

### `blockDim`

对于多个block、多个thread的kernel拷贝索引时有：

``` cpp
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

**`blockDim`**：三维数组，对所有的block都是固定值，它代表了一个block每一个维度上的thread数。当我们使用的是一维的block，所以只需要`blockDim.x`。

结合之前的`gridDim`，CUDA可以调用一个2维的block的grid，其中每一个block都有3维的thread。

### thread的设置

当计算两个向量的和的时候，如果每个向量都有N个元素，那么最好的方法就是总共使用N个thread。那么具体如何设置这个值呢？

第一个方法是设置一个固定值。比方说，每个block使用128个thread。所以，我们就可以使用`N / 128`个block，所以总共就有128个thread了。

但是，如果N是127的话，我们就使用了0个block（*上述是整数除法*）。实际上，如果N不是128的整数倍的话，我们就会使用了太少的thread。比较好的方法是使用：

``` cpp
add<<<(N + 127) / 128, 128>>>(dev_a, dev_b, dev_c);
```

实际上，这就会发起了更多的thread。所以我们上一章中提到的判断就很有用了：

``` cpp
if (tid < N)
```

这样就可以避免kernel访问的数据超出边界了。

*另外，注意一点：对于CPU来说，发起几百万个线程是很奇怪的，但对于GPU来说，是正常的。*

## 2.2 共享内存(shared memory)与同步

CUDA中有一块专用内存称为共享内存（shared memory），可以用`__shared__`关键词来修饰。

对于CUDA来讲，编译器对共享内存的变量和其他一般的变量有区别。它为每个block都会创建一个变量的副本。block中的每个thread都会共享这份内存，但是thread不能看到或修改其他block中的内存。这就为block中的thraed之间的沟通合作提供了方法。另外，如果我们想要在thread间通讯，那么就需要一个钟在thread间同步的机制。

### 使用方法

``` cpp
__shared__ float cache[threadsPerBlock];
```

cache数组的大小为`threadsPerBlock`，所以block中的每个thread都已一个地方来存放它的临时结果。

对于每个线程对应于cache中的索引，即：

``` cpp
int cacheIndex = threadIdx.x;
```

这个索引是和block索引没关系的，因为每个block中都有自己的一份私有的拷贝。

### 保证线程间同步的方法

``` cpp
while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
}
cache[cacheIndex] = temp;
```

我们在赋值给cache语句之后，需要一个线程间同步的方法，来保证所有的线程都向cache写入完成了，因为下一个步骤就是从cache中读出数据。

``` cpp
__syncthreads();
```

这个函数调用保证了一个block中的thread都在进行这个函数调用之后的所有的指令前，完成了所有的指令。

### 同步注意点

假设有如下语句：

``` cpp
if (threadIdx.x % 2)
```

当一些线程执行一些指令，而另一些线程不执行或执行其他指令，这叫做thread divergence。这种情况很可能导致一些线程空闲。

对于`__syncthreads()`，CUDA保证了没有任何线程会执行其之后的语句，直到所有的线程都执行完`__syncthreads`。**所以，如果`__syncthreads()`放在分支中，那么就有一些thread肯定执行不了这条语句，那么就会一直在这里等待下去。**

# 3 总结

我们这一节学习了thread的概念、共享内存，都是GPU中很重要的概念，有着很广泛的使用。

另外，这里摘录一句书里的话，对reduction下的定义。这也是我疑惑多时的内容。

> reduction: taking an input array and performing some computations that produce a smaller array of results.

# 4 英语学习

- redux：（自远方或放逐）回来的
- overshoot：冲出、冲过
- refrain：避免，克制，抑制，节制
- forthcoming：乐于助人的
- ripple：波纹，涟漪
- sinusoidal：sine曲线的
- scratchpad：便浅薄，高速缓存存储器
- convoluted：错综复杂的，盘绕的，弯曲的
- conceivably：可以想象地
- sans：（古语）无，等于without
- innocuous：无意冒犯的，无害的
- moral：寓意，教益
- aesthetically：美学上地，审美地
