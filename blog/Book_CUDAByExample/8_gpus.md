# CUDA by Example 笔记(八): 多GPU

*对应书本11章*

# 1 大纲

- 零拷贝内存
- 多个GPU的使用与优化

# 2 内容

## 2.1 零拷贝内存

记得之前说过`cudaHostAlloc()`函数中最后一个参数是`cudaHostAllocDefault`，可以分配一个默认的锁页内存。这里介绍另一个参数：`cudaHostAllocMapped`，它分配的内存仍然是锁页内存。但是，还有一个重要的性质，它分配的内存可以被kernel直接访问。因为这段内存不需要在device与host之间来回拷贝，即零拷贝内存。

### 使用条件

要求device必须支持device映射host的内存，见结构体`cudaDeviceProp`的`canMapHostMemory`字段。

设置device为支持零拷贝内存的状态：

``` cpp
cudaSetDeviceFlags(cudaDeviceMapHost);
```

### 定义

举例如下：

``` cpp
cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
```

注意到，这里有个flag，`cudaHostAllocWriteCombined`，它执行运行时系统分配内存时，对CPU缓存来讲是write-combined。这个flag不会影响任何的功能，但是对只会在GPU上读的内存会带来一个性能提升。然而，如果CPU仍会读这段内存的话，write-combined反而会更加低效。所以，在使用这个flag的时候，必须要考虑具体应用程序的访问模式。

### 在GPU上访问

虽然零拷贝内存是可以在GPU上直接访问的，但是GPU和CPU有不同的虚拟内存空间，所以GPU、CPU上对统一内存访问起来，地址是不同的。`cudaHostAlloc`函数返回指向内存的CPU指针，所以，我们需要使用`cudaHostGetDevicePointer`来得到指向这个内存的GPU指针。将这个指针传入到kernel，就可以在GPU上读写host上分配的内存了。

### CPU、GPU之间同步：`cudaThreadSynchronize`

举例：

``` cpp
// kernel<<<...>>>(...);
cudaThreadSynchronize();
```

*<font color=red>问题9：这个thread同步deprecated了？用啥代替呢？</font>*

*<font color=red>问题10：为啥需要CPU、GPU同步呢？</font>*

因为通过上面的函数分配的内存是不需要再拷贝回到host上的。注意到使用了`cudaThreadSynchronize`进行了CPU、GPU之间进行同步。其实，在kernel执行之间，零拷贝内存的内容是未定义的。经过同步之后，我们确定kernel完成了，并且内存中的内容已经是计算之后的结果了。

### 零拷贝内存的性能

离散式GPU有自己独立专用的DRAM，一般和CPU在不同的电路板上。集成式GPU集成到系统的芯片组上，并且一般和CPU共享普通系统内存。对于集成式GPU，使用零拷贝内存一定会有性能收益，因为内存物理上和host共享内存。注意到，零拷贝内存也是一种锁页内存，所以使用过多的话，会最终导致系统性能的下降。

当输入和输出只使用一次的情况，在离散式GPU使用零拷贝内存也会有性能提升。因为GPU的设计就是精于隐藏和内存访问相关的延迟，通过这种机制进行的经由PCIE进行的读写会得到一定程度得缓解，并得到一定的性能提升。但是，由于零拷贝内存不会被GPU缓存，如果需要多次读写的话，性能反而不如现将内存拷贝到GPU上。

结构体`cudaDeviceProp`中的`integrated`字段表示了GPU是离散式的还是集成式的。

由于不同的计算与带宽的比率，也由于芯片组不同的有效PCIE带宽，不同的GPU展示了不用的性能特质。

## 2.2 使用多个GPU

*用户可以将多个GPU放到不同的PCIE槽位，通过NVIDIA的Scalable Like Interface(SLI)技术将它们相连。*

使用多个GPU的技巧之一就是每个GPU由不同的CPU线程来控制。

### 设置device

``` cpp
cudaSetDevice(data->deviceID);
```

可以在每一个线程里面执行设置device的命令，然后这一个线程内的代码都在此device上执行。

如果你的GPU是不同的，特别是计算能力不同的话，那么需要来讲计算分片，使得每个GPU都使用差不多的时间。

## 2.3 便携式锁页内存

存在这么一种内存，只对单个CPU来说是锁页的。也就是说，如果某个线程分配的是锁页内存，那对这个线程就是锁页的，但是只对分配的那个线程来说是锁页的。就算和其他线程共享了这个指针，对其他线程来说也是标准的、可换页的内存。

当一个非分配内存的线程执行对这块内存进行`cudaMemcpy`拷贝时，这个拷贝就会像标准的、可换页的内存一样。如果想使用`cudaMemcpyAsync`并入队stream后，这个操作就会失败，因为这个函数只接受锁页内存作为参数。

但是可以通过将锁页内存设置为便携式的来解决这个问题，然后它就可以在不同host线程之间迁移。仍然使用`cudaHostAlloc`来分配内存，但是使用flag：`cudaHostAllocPortable`，可以`cudaHostAllocWriteCombined`、`cudaHostAllocMapped`参数同时使用。如：

``` cpp
cudaHostAlloc((void**)&a, N * sizeof(float), 
              cudaHostAllocWriteCombined |
              cudaHostAllocPortable |
              cudaHostAllocMapped);
```

*<font color=red>问题11：write-combined是啥意思呢？</font>*

因为`cudaSetDevice(0)`之后才分配的内存，所以如果我们没有设置其为portable的，只有device0才会看到这些内存是锁页内存。

一旦你在某个线程上设置device，就不能再调用`cudaSetDevice()`了，就算传入的参数是一样的。

# 3 总结

本节主要介绍了零拷贝内存的使用，以及多个GPU的使用。

# 4 英语学习

- whopping：巨大的，很大的
- farfetched：牵强的
- piddling：不重要的，微不足道的
- grisly：恐怖的
- in concert with：合作
- gram：写（或画）的东西

