# CUDA by Example 笔记(六): 原子操作

*对应书本第9章与附录*

# 1 大纲

- 原子操作基本原理
- 原子锁

# 2 内容

## 2.1 编译

全局内存上的原子操作的支持要求计算能力超过1.1，*共享内存*上的原子操作要求计算能力超过1.2。

编译的时候，要告诉编译器，代码不能在低于1.1的硬件上运行。更重要的是，这就告诉了编译器，可以在针对计算能力进行相应的优化。举例：

``` shell
nvcc --arch=sm_11
nvcc --arch=sm_12
```

其实，也就是说，这个`--arch`参数指定的计算能力版本，最好是你的GPU支持的最高的版本。

### CUDA上的`memset`：`cudaMemset`

举例如下，基本上和CPU上的`memset`没有太大区别，其中`dev_histo`是指向device的内存的指针。

``` cpp
cudaMemset(dev_histo, 0, 256 * sizeof(int));
```

*备注：P174页的第二个分配内存函数的size应该是`256 * sizeof(int)`，即最后面不应该是`long`。*

## 2.2 原子操作相关函数

原子操作的基本概念想必大家都很熟悉，这里只对GPU上的应用进行介绍。要注意：是**硬件**保证了院子操作不会被中断。

### `AtomicAdd(addr, y)`

其中`addr`是被加数的地址，`y`是加数。

当很多个线程同时对同一个内存地址进行读写时，为了保证原子性，硬件会将所有操作串行化，这可能会导致产生一个很长的等待队列，并且获得的性能提升可能会随之消失。

### 减少内存冲突的方法

利用书中计算直方图的方法举例来讲：

1. 每个block内现将自己的结果暂存到共享内存上。
2. 最后将每个block的结果存放到全局内存上。

这样就避免了每个thread都频繁地发起原子加操作，并且每次只是加1.

## 2.3 原子锁

先看书中代码`"lock.cuh"`：

``` cpp
#pragma once

struct Lock{
    int *mutex;

    Lock(void) {
        int state = 0;
        HANDLE_ERROR( cudaMalloc((void**)&mutex, sizeof(int)) );
        HANDLE_ERROR( cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice) );
    }

    ~Lock(void) {
        cudaFree(mutex);
    }

    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
        __threadfence();
    }

    __device__ void unlock(void) {
        __threadfence();
        atomicExch(mutex, 0);
    }
};
```

首先唯一一个成员变量，即`int* mutex`，它是一个在GPU上分配的int类型指针。

构造函数的作用就是分配这个内存，而析构函数的作用就是释放这个内存。

`lock()`函数调用了`atomicCAS()`，`unlock()`函数则调用了`atomicExch()`。

### `atomicCAS(int* address, int compare, int val)`

atomic compare and swap，和CPU上对应的函数是很类似的。它的第一个参数即为在GPU上分配的内存，第二个参数为需要和前面指针指向的数据进行compare的数，第三个参数：当compare返回true的时候，将其赋值给指针指向的内存。简单来讲，就是下列操作是通过一个原子操作完成的。

``` cpp
int old = *address;
*address = (old == compare) ? val : old;
return old;
```

### `atomicExch(int* address, int val)`

它将`val`赋值给第一个参数指向的地址，并返回原来的值。简单来讲，就是下列操作是通过一个原子操作完成的。

``` cpp
int old = *address;
*address = val;
return old;
```

Lock结构体，通过向`mutex`写1实现加锁，通过向`mutex`写0实现解锁。因为可能出现的抢锁情况，所以要加一个`while`循环一直判断。

总体逻辑和CPU上是一致的，主要的区别就是使用了CUDA上的API。

### 使用方法

如你需要保护的是变量`count`的复制：

``` cpp
lock.lock();
++count;
lock.unlock();
```

注意：这里的`Lock`对象要在所有线程之间共享。

## 2.4 哈希表中多线程读写的一些优化

``` cpp
while (tid < ELEMENTS) {
    unsigned int key = keys[tid];
    size_t hashValue = hash( key, table.count );
    for (int i=0; i<32; i++) {
        if ((tid % 32) == i) {
            Entry *location = &(table.pool[tid]);
            location->key = key;
            location->value = values[tid];
            lock[hashValue].lock();
            location->next = table.entries[hashValue];
            table.entries[hashValue] = location;
            lock[hashValue].unlock();
        }
    }
    tid += stride;
}
```

主要注意到这里的第五行：`if ((tid % 32) == i)`。

在GPU中，32个线程为一个warp，一个warp中的所有线程在一个时钟周期内同时执行。所以，对于一个warp内的线程来讲，只有一个线程可以获取锁，所以如果我们让warp中的每个线程都同时竞争锁是没什么必要的。从上面的代码中不难发现，对应于一个`i`的值，一个warp中的线程中只有一个会进入到`if`语句括起来的语句。

# 3 总结

本节学习了原子操作的基本知识，简单优化方法，还有原子锁，关于哈希表中多线程读写的一些优化。

书中对无锁队列的讲解很好，这是对CPU和GPU通用的内容，如果对相关内容不熟悉的话，可以认真看下。

# 4 英语学习

- drastic：极端的，剧烈的
- splurge：挥霍
- tabulate：列表显示，将……列成表格
- hullabaloo：喧嚣
- at length：最终，最后，长久地，详尽地
- stipulation：条款，规定
- fret：苦恼
- primer：启蒙本
- warrant：使有必要，使正当，使恰当
- leaves much to be desired：令人不满意，有待改善，还有许多待改进之处，不够好
- trounce：彻底打败
- blanket：总括的，综合的，包括所有情形（或人员）的
- blanket rule：通用规则