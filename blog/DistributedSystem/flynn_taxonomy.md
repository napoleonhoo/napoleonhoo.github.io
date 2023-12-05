# 费林分类法（Flynn's Taxonomy）

Reference: [Flynn's taxonomy - Wikipedia](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy)

- Single data stream
  - SISD, Single Instruction Single Data
  - MISD, Multiple Instruction Single Data
- Multiple data streams
  - SIMD, Single Instruction Multiple Data
  - MIMD, Multiple Instructions Multiple Data
- SIMD Subcategories
  - Array processing(SIMT, Single Instruction Multiple Threads)
  - Pipelined processing(packed SIMD)
  - Associative processing(predicted/masked SIMD)
- SPMD, Single Programme Multiple Data
- MPMD, Multiple Programme Multiple Data

## SISD, Single Instruction Single Data

Reference: [Single instruction, single data - Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_single_data)

计算机架构，单个处理器执行单个指令流，对单个内存上的数据进行操作。对应于冯·诺依曼架构（von Neumann architecture）。

## MISD, Multiple Instruction Single Data

Reference: [Multiple instruction, single data - Wikipedia](https://en.wikipedia.org/wiki/Multiple_instruction,_single_data)

一种并行计算机构，多个功能单位在同样的数据上执行不同的操作。“流水线”就是这样的类型。这种架构的应用比起MIMD和SIMD较为少见，因为它们对于一般数据并行技术更为合适。特别地，它们更灵活使用计算资源。*但是，一个很有名的使用MISD计算的例子是太空飞船的飞行计算机。*

## SIMD, Single Instruction Multiple Data

Reference: [Single instruction, multiple data - Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)

一种并行处理的类型。SIMD可以是内部的（硬件设计的一部分），可以直接通过指令集架构（Instruction Set Architecture，ISA）来访问，但是不应该和ISA混淆。SIMD描述了有多个处理单元在不同的数据上同时进行同样的操作。这种机器利用数据级别并行，但不是“并发”：这是同时（并行）计算，但是每个单元在任意时刻执行同样的指令，只是不同的数据。

SIMD特别适合于一般的任务，如调整数字图像中的对比度，调整数字音频的音量。多数现代CPU设计包括了SIMD指令来增强多媒体使用的性能。

## MIMD, Multiple Instructions Multiple Data

Reference: [Multiple instruction, multiple data - Wikipedia](https://en.wikipedia.org/wiki/Multiple_instruction,_multiple_data)

实现并行的一种技术。使用MIMD的机器，有多个的处理器，异步地、独立地工作。在任意时刻，不同的处理器可能对不同的数据执行不同的指令。

MIMD架构可以在多种应用领域中，如计算机辅助设计、计算机辅助制造、仿真、建模、通信开关。MIMD机器可以属于共享内存（shared memory）或分布式内存（distributed memory）分类。这些分类基于MIMD处理器是如何访问内存的。共享内存机器可能是基于总线的、扩展的、分级的类型。分布式内存机器可能有网格网络（Grid network，hypercube）或mesh连接构图。

## Array processing(SIMT, Single Instruction Multiple Threads)

特征是并行子元素有自己的独立的寄存器和内存（缓存和数据内存）。Nvidia

## Pipelined processing(packed SIMD)

系统使用主内存作为流水线读和写的资源。当所有流水线从中读和写的资源是寄存器而非主内存。

## Associative processing(predicted/masked SIMD)

许多现代的设计（特别是GPU）使用下列的几个特征：今天的GPU是SIMT，但也是相关的，即每个SIMT阵列中的每个处理元素也是可预测的。

## SPMD, Single Programme Multiple Data

Reference: [Single Program, Multiple Data - Wikipedia](https://en.wikipedia.org/wiki/Single_program,_multiple_data)

两种不同的计算模型：
- “fork-and-join”，并行任务（“单个程序”）分开且同时运行。在多个SIMD处理器上，不同的输入，数据并行方法
- （更普遍的方法）所有的处理器运行同样的程序，通过同步指令，自我调度来执行不同的指令且在不同的数据上。对一个程序使用MIMD并行，是一个数据并行更基础的方法。比起fork-and-join更高效。

### Distributed Memory

在分布式内存计算机架构上，SPMD的实现通常使用了消息传递编程。一个分布式内存计算机包含了互相链接的、独立的计算机，称为节点（nodes）。对于并行执行，每个节点启动自己的程序，通过发送和接收消息来和其他节点通信。其他的并行指令，如barrier同步，也可以使用消息来实现。对于分布式内存环境，程序中串行部分可以在所有机器上同样地执行，而不是在一个节点上执行然后将结果发送到其他节点上。如今，使用PVM和MPI可以使得程序员原理消息传递的细节。

### Shared Memory

在共享内存机器（一个计算机有多个互相连接的CPU，访问同一块内存地址）上，共享可以通过物理上的共享内存或逻辑上的共享内存（物理上是分布式的）来实现。在共享内存之外，这个系统中的CPU仍然可以拥有本地或私有的内存。对上述这两种情况来说，同步可以通过硬件支持的原语（如compare-and-swap、fetch-and-add）来实现。对于没有这种硬件支持的机器来说，可以使用锁，通过将共享的数据放在共享的内存区域，数据可以在处理器间交换（或更普遍的说法是进程或线程）。当硬件不支持共享内存时，一般将数据打包成“消息”是常见的高效的方法。

在共享内存机器上的SPMD可以通过进程或线程来实现。目前在对共享内存多处理器的标准接口上，使用SPMD，OpenMP使用多线程。

### Symmetric multiprocessing, shared-memory multiprocessing, SMP

Referece: [Symmetric multiprocessing - Wikipedia](https://en.wikipedia.org/wiki/Symmetric_multiprocessing)

包括一个多处理器计算机软件和硬件架构，其中多个相同的处理器连接到一个共享的内存，对所有的输入输出设备有完全的访问权限，由一个OS实例控制，平等看待所有的处理器，不为special purposes预留。大多数多处理器系统使用SMP架构。在多核处理器中，SMP架构应用在核上，将核看成独立的处理器。

### Non-uniform memory access

Reference: [Non-uniform memory access - Wikipedia](https://en.wikipedia.org/wiki/Non-uniform_memory_access)

NUMA。是为多处理器设计的计算机内存，内存访问时间取决于相对处理器的内存位置。在NUMA下，一个处理器可以更快地访问它自己的本地内存，比起来非本地的内存（其他处理器的本地内存或多处理器间共享的内存）。NUMA的收益局限于特定的工作负荷（workloads），特别是在服务器上，数据经常和特定的任务或用户强相关。

NUMA架构逻辑上随着SMP架构扩展。

## MPMD, Multiple Programme Multiple Data

多个自主的处理器同时运行至少2个独立的程序。一般地，这种系统选择一个节点作为host（host/node编程模型）或manager（manager/worker策略），它运行一个程序将数据分发到其他运行另一个程序的节点。这些节点直接将结果返回manager。

一个例子是索尼PS3游戏主机，使用SPU/PPU处理器（又叫Cell，是一个64位多核微架构，结合了通用的普通性能的PowerPC核，流水线协处理元素，很大地加速多媒体和向量处理应用。）。