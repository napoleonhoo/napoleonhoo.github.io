学《Performance Hints》：Google Jeff Dean 大神的性能优化方法论

本文主要来自我对 [Jeffrey Dean & Sanjay Ghemawat, *Performance Hints*, 2025](https://abseil.io/fast/hints.html) 的学习和整理。

这不是原文的逐段翻译，而是我从中摘出了一些更适合日常工程实践、尤其是 C++ 服务端开发中可复用的性能优化思路，并结合自己的理解做了结构化梳理。

*如果你需要《Performance Hints》全文 AI 翻译版，可以私信我获取。*

# 1. 估算

在动手优化之前，先判断性能问题是否重要，以及大概重要到什么程度。

## 1.1 不同代码的重要性不同

- **这是测试代码吗？**  
  如果是，主要关注算法和数据结构的渐近复杂度。同时也要注意开发周期时间，避免写出运行时间过长的测试。

- **这是某个应用自己的代码吗？**  
  如果是，先判断它是否在热路径上。初始化代码和“每个请求都要执行”的代码，对性能的敏感程度完全不同。

- **这是会被很多应用使用的库代码吗？**  
  如果是，就更需要谨慎。库代码一旦被广泛依赖，性能成本会被放大；而且真正遇到性能问题的调用方，往往没有能力轻易修改库内部实现。

## 1.2 粗略估算的基本方法

估算通常可以按下面几步来做：

1. 估算会发生多少低层操作，例如磁盘寻道次数、网络往返次数、内存访问量、传输字节数等。
2. 将每类昂贵操作的次数乘以粗略成本。
3. 把这些成本相加，得到资源使用量层面的粗略估计。
4. 如果关注的是延迟，还需要考虑并发：有些成本可能会重叠，并不一定简单相加。

下面这张表列出了一些典型操作的大致成本，可以用来做粗略估算：

```text
L1 cache reference                             0.5 ns
L2 cache reference                             3 ns
Branch mispredict                              5 ns
Mutex lock/unlock (uncontended)               15 ns
Main memory reference                         50 ns
Compress 1K bytes with Snappy              1,000 ns
Read 4KB from SSD                         20,000 ns
Round trip within same datacenter         50,000 ns
Read 1MB sequentially from memory         64,000 ns
Read 1MB over 100 Gbps network           100,000 ns
Read 1MB from SSD                      1,000,000 ns
Disk seek                              5,000,000 ns
Read 1MB sequentially from disk       10,000,000 ns
Send packet CA->Netherlands->CA      150,000,000 ns
```

## 1.3 例子：估算十亿个 4 字节整数的快速排序时间

这个例子很适合训练性能直觉。

粗略地说，一个好的 quicksort 会对大小为 `N` 的数组进行 `log(N)` 趟扫描。每一趟中，数组内容会从内存流式进入 CPU cache，分区逻辑会把每个元素与 pivot 比较一次。

假设：

- 数组大小：`1B * 4 bytes = 4GB`
- 单核内存带宽：约 `16GB/s`
- `N ≈ 2^30`，所以 `log(N) ≈ 30`

那么：

- **内存带宽成本**：每趟扫描 `4GB`，约 `0.25s`；30 趟约 `7.5s`。
- **分支预测失败成本**：总比较次数约 `N * log(N) ≈ 300 亿次`。如果一半预测失败，即 `150 亿次`，每次 `5ns`，则成本约 `75s`。
- **粗略总成本**：约 `82.5s`。

如果进一步考虑 L3 cache，后面若干趟排序的数据可能已经驻留在缓存中，内存传输成本会降低。但在这个估算中，主导成本是分支预测失败，所以是否细化 cache 模型并不会改变主要结论。

这个例子的重点不是得出精确时间，而是建立直觉：

> 有时候看起来只是一次排序，但主导成本可能不是比较次数本身，而是内存带宽、cache 行为和分支预测。

# 2. 测量

估算能帮助我们判断方向，但真正做优化时，测量是第一工具。

另外，一个很实用的观点是：

> 对不熟悉的代码做性能剖析，也是理解代码结构的一种方式。

通过观察动态调用图、火焰图、热点函数和源码，可以快速建立“程序运行时到底发生了什么”的全局感。

## 2.1 性能剖析工具和技巧

常见建议包括：

- 首先考虑使用 **pprof**。它能提供良好的高层性能信息，本地和线上都比较容易使用。
- 如果需要更底层的信息，可以使用 **perf** 观察 CPU 事件、cache miss、分支预测失败等。
- 构建生产二进制时，要使用合适的调试信息和优化参数。
- 如果可以，为要优化的代码编写 **microbenchmark**。它能缩短反馈周期，验证优化效果，也能防止未来性能回退。
- 使用 benchmark 库输出硬件性能计数器，帮助判断瓶颈来自哪里。
- 锁竞争会人为降低 CPU 使用率；如果怀疑锁竞争，要看 mutex contention 相关指标。
- 机器学习场景要使用专门的 ML profiler。

## 2.2 当 profile 很平坦（flat）时怎么办？

有时 CPU profile 很平坦，没有一个特别突出的热点。这通常发生在明显问题都已经被修过之后。

这种情况下可以考虑：

- **不要低估小优化的累积价值**。在一个子系统中做二十个 1% 的优化，最终可能就是很可观的收益。
- **寻找调用栈上更靠近顶部的循环**。火焰图通常有帮助。高层循环的结构性调整，可能比底层函数微优化更有效。
- **后退一步看结构性问题**，而不是只盯着微优化。
- **寻找过于通用的代码**。如果简单前缀匹配足够，就不一定要用正则表达式。
- **减少分配次数**。分配不仅消耗 allocator 时间，也会增加 cache miss，因为分散分配的数据通常落在不同 cache line 上。
- 收集其他类型的剖析结果，特别是**收集硬件计数器类 profile**。例如 cache miss、branch miss、stall cycle 等，能帮助定位普通 CPU profile 看不出来的问题。

# 3. API 考量

有些性能问题不是实现细节造成的，而是 API 设计本身迫使调用方或被调用方做了不必要的工作。

文章强调：

> 尽量让性能改进发生在封装边界内部，而不是频繁破坏公共接口。

如果模块足够深、接口足够窄，内部就更容易替换数据结构、调整算法或优化内存布局，而不影响调用方。

## 3.1 批量 API

提供批量操作可以减少昂贵的 API 边界跨越，也可能打开算法优化空间。

例如原来每次查一个对象：

```cpp
Lookup(key);
```

如果调用方经常一次查很多对象，就可以提供：

```cpp
LookupMany(keys, results);
```

这样可以：

- 减少函数调用和边界检查次数；
- 批量处理锁、内存分配、缓存查找；
- 利用更高效的批处理算法。

## 3.2 使用视图类型

函数参数优先使用视图类型，例如：

```cpp
std::string_view
absl::Span<T>
absl::FunctionRef<R(Args...)>
```

除非确实需要转移所有权，否则视图类型通常更好。它们可以减少复制，并允许调用方选择自己的容器实现。

比如同一个函数可以接受来自：

- `std::vector<T>`
- `absl::InlinedVector<T, N>`
- 原始数组
- `std::array<T, N>`

只要暴露成 `absl::Span<const T>` 即可。

## 3.3 让调用方传入已知信息或临时存储

对于频繁调用的例程，如果调用方已经拥有某些信息或临时结构，可以允许调用方传入，避免低层例程重复分配或重复计算。

例子：让调用方传入已经获取过的 `WallTime`。

```cpp
// old
static void RecordRPC(const Name& name, const RPC_Stats_Measurement& m);

// new
static void RecordRPC(const Name& name, const RPC_Stats_Measurement& m,
                      WallTime now);
```

调用方原本已经调用过：

```cpp
const WallTime now = WallTime_Now();
```

那就没必要让 `RecordRPC` 内部再调用一次。

## 3.4 线程兼容类型与线程安全类型

一个类型可以是：

- **线程兼容的**：调用方负责外部同步，函数外负责线程安全；
- **线程安全的**：类型内部自己做同步，函数内负责线程安全。

大多数通用类型应该优先设计成线程兼容。这样，不需要线程安全的调用方就不会为内部锁、原子操作、内存屏障付出额外成本。

但如果某个类型的典型使用天然需要同步，则可以把同步放进类型内部。这样实现方可以优化同步机制，比如分片锁、无锁结构等，而不影响调用方。

# 4. 算法改进

性能提升最关键的机会往往来自算法改进，例如：

- 从 `O(N^2)` 变成 `O(N log N)`；
- 从 `O(N log N)` 变成 `O(N)`；
- 避免潜在指数级行为；
- 用更适合访问模式的数据结构替代原结构。

文章中列了很多真实代码变更，我摘几个最有代表性的：

- **按反向后序将节点添加到环检测结构中**：原来逐个添加图节点和边，每条边都要做昂贵检查；改为一次性按顺序初始化整个图，显著降低环检测成本。
- **替换 mutex 内置死锁检测算法**：新算法速度约快 50 倍，并能扩展到数百万个 mutex；旧算法依赖 2K 限制来避免性能断崖。
- **将 `IntervalMap` 替换为哈希表**：如果相邻块可以通过哈希查找找到，就不一定需要 `O(log N)` 的区间结构。

这一节的核心启发是：

> 算法和数据结构选择，比局部微优化更重要。稳定代码中这种机会少，但一旦发现，收益通常很大。

# 5. 更好的内存表示

重要数据结构的内存占用和 cache 行为，往往直接决定性能。

好的内存表示可以：

- 触碰更少 cache line；
- 减少 cache miss；
- 减少内存总线流量；
- 降低 allocator 开销；
- 改善同机其他程序的表现。

## 5.1 紧凑数据结构

对于频繁访问、或占应用内存比例较高的数据，应优先考虑紧凑表示。

紧凑表示能通过触碰更少 cache line 提升性能，但也要注意 cache line 争用和 false sharing。

## 5.2 内存布局

设计结构体时，字段顺序也很重要：

- 重新排序字段，减少不同对齐要求导致的 padding。
- 如果数据能放入更小类型，就不要默认使用大类型。
- 枚举值可能默认占一个完整字；必要时可以使用 `enum class OpType : uint8_t`。
- 经常一起访问的字段放近一些，减少常见操作触碰的 cache line 数量。
- 热的只读字段和热的可变字段要分开，避免写入可变字段时驱逐只读字段所在 cache line。
- 冷数据不要紧挨热数据，可以放到结构体末尾、间接层之后，或者单独数组中。
- 位级、字节级压缩可以显著节省内存，但要权衡访问成本和复杂度，并用 benchmark 验证。

## 5.3 用索引代替指针

在 64 位机器上，指针通常占 64 位。如果一个数据结构里有很多 `T*`，内存开销会很大。

如果对象存储在数组中，可以考虑用整数索引代替指针：

```cpp
// pointer
T* next;

// index
uint32_t next_index;
```

这样不仅引用更小，数组元素也更连续，通常 cache locality 更好。

## 5.4 批量存储

避免使用每个元素都单独分配对象的数据结构，例如：

```cpp
std::map
std::unordered_map
std::list
```

可以考虑使用分块或扁平表示：

```cpp
std::vector
absl::flat_hash_map
absl::flat_hash_set
```

这类结构通常有更好的 cache 行为，也能减少 allocator 开销。

## 5.5 内联存储

某些容器针对少量元素进行了优化，会在对象内部预留一小块空间。当元素数量较少时，不需要堆分配。

典型例子是：

```cpp
absl::InlinedVector<T, N>
```

适合“通常很小，偶尔变大”的动态数组。

但要注意：如果 `sizeof(T)` 很大，内联存储会显著增大对象本身，不一定划算。

## 5.6 不必要的嵌套 map

有时嵌套 map 可以改成复合 key 的单层 map：

```cpp
btree<a, btree<b, c>>
```

改成：

```cpp
btree<pair<a, b>, c>
```

这样可以减少分配并改善 cache 占用。

但如果第一层 key 很大，保留嵌套 map 可能更好，因为可以避免重复存储大 key。

## 5.7 Arenas

Arena 可以降低内存分配成本，也可以把原本独立分配的对象紧密放在一起，从而减少 cache line 数量，并降低析构成本。

它尤其适合包含大量子对象的复杂数据结构。

但 arena 也容易误用。例如把太多短生命周期对象放进长生命周期 arena，会导致内存占用不必要地膨胀。

## 5.8 用数组代替 map

如果 map 的定义域可以表示为小整数、枚举，或者元素数量很少，可以考虑用数组或 vector 替代 map。

例如：

```cpp
std::array<Value, kMaxEnumValue>
```

比哈希表或树结构更直接、更 cache-friendly。

## 5.9 用 bit vector 代替 set

如果 set 的定义域可以表示为小整数，可以用 bit vector 替代 set。

例如：

```cpp
bit[x] = 1;  // 表示 x 在集合中
```

这样集合操作可以转化为按位操作：

- OR 表示并集；
- AND 表示交集；
- AND NOT 表示差集。

`InlinedBitVector` 通常适合短 bit vector 场景。

# 6. 减少分配

内存分配的成本包括：

- allocator 本身的时间；
- 新对象的初始化成本；
- 对象生命周期结束时的析构与释放成本；
- 数据分散在多个 cache line 上导致的 cache miss。

GC 语言有时可以通过连续分配缓解部分问题，但 C++ 中大量小分配依然常常是性能问题来源。

## 6.1 避免不必要的分配

原文有两个非常实用的例子。

### 例子一：共享默认对象

旧代码中，如果 `dinfo` 为空，就每次创建一个默认 `DeviceInfo`：

```cpp
// old
LiveTensor::LiveTensor(tf::Tensor t, std::shared_ptr<const DeviceInfo> dinfo,
                       bool is_batched)
    : tensor(std::move(t)),
      device_info(dinfo ? std::move(dinfo) : std::make_shared<DeviceInfo>()),
      is_batched(is_batched) {}
```

优化后，将默认对象变成共享单例：

```cpp
// new
static const std::shared_ptr<DeviceInfo>& empty_device_info() {
  static std::shared_ptr<DeviceInfo>* result =
      new std::shared_ptr<DeviceInfo>(new DeviceInfo);
  return *result;
}

LiveTensor::LiveTensor(tf::Tensor t, std::shared_ptr<const DeviceInfo> dinfo,
                       bool is_batched)
    : tensor(std::move(t)), is_batched(is_batched) {
  if (dinfo) {
    device_info = std::move(dinfo);
  } else {
    device_info = empty_device_info();
  }
}
```

核心思路：

> 如果默认对象不可变、无身份语义，就不要每次都重新分配；共享一个默认对象即可。

### 例子二：复用静态零向量

旧代码每次都分配一段零数组并清零：

```cpp
// old
template <bool Mean, bool SymmetricInputRange>
static tensorflow::Status EmbeddingLookUpT(...) {
  ...
  std::unique_ptr<tensorflow::quint8[]> zero_data(
      new tensorflow::quint8[max_embedding_width]);
  memset(zero_data.get(), 0,
         sizeof(tensorflow::quint8) * max_embedding_width);
}
```

优化后，对常见 embedding 宽度使用静态全零数组：

```cpp
// new
static const int kTypicalMaxEmbedding = 256;
static tensorflow::quint8 static_zero_data[kTypicalMaxEmbedding];  // 全部为零

template <bool Mean, bool SymmetricInputRange>
static tensorflow::Status EmbeddingLookUpT(...) {
  ...
  std::unique_ptr<tensorflow::quint8[]> zero_data_backing(nullptr);
  tensorflow::quint8* zero_data;

  if (max_embedding_width <= ARRAYSIZE(static_zero_data)) {
    zero_data = &static_zero_data[0];
  } else {
    zero_data_backing =
        absl::make_unique<tensorflow::quint8[]>(max_embedding_width);
    memset(zero_data_backing.get(), 0,
           sizeof(tensorflow::quint8) * max_embedding_width);
    zero_data = zero_data_backing.get();
  }
}
```

这个优化把常见路径上的 `new[] + memset + delete[]` 变成了一次条件判断和一次指针赋值。

## 6.2 调整容器大小或预留容量

如果知道 vector 的最大大小或预期大小，应提前设置后备存储：

```cpp
v.reserve(n);
```

或者：

```cpp
v.resize(n);
```

但要注意：

- 不要每次只 `reserve(size + 1)` 或 `resize(size + 1)`，这可能导致二次方行为。
- 如果元素构造成本高，通常优先 `reserve` 后多次 `push_back` / `emplace_back`，而不是先 `resize`，因为 `resize` 会调用构造函数。

## 6.3 尽可能避免复制

常见策略包括：

- 能移动就不要复制；
- 临时结构中存指针或索引，而不是复制大对象；
- 排序大对象时，可以排序索引 vector，而不是直接排序对象 vector；
- 使用 `std::sort` 替代不必要的 `std::stable_sort`，避免 stable sort 内部额外复制；
- 通过 gRPC 接收 tensor 时避免额外复制；
- 对大型 options 结构使用 `std::move`。

## 6.4 复用临时对象

循环内部声明的容器或对象，会在每次迭代时重新构造和析构。对于 vector、string、protobuf 等对象，这可能带来明显成本。

文中举的例子有：

- 将变量定义提升到循环迭代之外。
- 在循环外定义 protobuf 变量，以便其已分配的存储可以在循环迭代之间复用。
- 序列化到同一个 `std::string`。

但也要注意：这类对象会增长到曾经容纳过的最大容量。必要时可以每使用 N 次后重建一次，避免长期持有过大的 backing storage。

# 7. 避免不必要的工作

性能优化中最有效的一类方法，就是**不要做不必做的工作**。

这包括：

- 为常见情况创建快速路径；
- 一次性预计算昂贵信息；
- 将昂贵计算移出循环；
- 推迟计算，直到真正需要；
- 编写专门化代码替代过于通用的代码；
- 使用缓存避免重复工作；
- 帮助编译器生成更好的代码；
- 降低统计和日志的成本。

## 7.1 常见情况的快速路径

代码通常覆盖所有情况，但其中某些情况更简单、更常见。

例如 `vector::push_back` 的常见路径是 capacity 足够；只有少数情况下需要扩容。

合理组织代码，可以让常见路径更短、更容易内联、更利于分支预测，同时不显著损害慢路径。

## 7.2 一次性预计算昂贵信息

一个常见原则：

> 在模块边界检查格式错误的输入，而不是在内部反复检查。

如果某些信息可以在进入核心逻辑前统一验证或预计算，就不要在内层循环中重复做。

## 7.3 将昂贵计算移出循环

如果某个计算与循环变量无关，就应该移出循环。

这类优化有时编译器能做，但只要涉及函数调用、别名、虚函数、外部状态等，编译器就可能必须保守处理。

## 7.4 推迟昂贵计算

如果某个结果只有在少数情况下才需要，就不要提前计算。

## 7.5 专门化代码

性能敏感调用点未必需要通用库的全部能力。

文章举到的例子包括：

- 为 `VLOG(1)`、`VLOG(2)` 等常量日志级别添加特化，减少代码大小和参数传递成本；
- 在可行时用简单前缀匹配替换 RE2；
- 使用 `StrCat` 而不是 `StringPrintf` 来格式化 IP 地址。

## 7.6 使用缓存避免重复工作

如果输入不变、计算昂贵，可以缓存结果。

比如基于大型序列化 proto 的预计算 fingerprint 做缓存，避免重复解析和计算。

## 7.7 让编译器的工作更轻松

有时多层抽象会让编译器难以优化。应用程序员更了解具体语义，可以适当把关键路径改写成更低层级的形式。

一些可能有用的技巧：

- 避免在热函数中调用函数，让编译器避免栈帧设置成本；
- 将慢路径移到单独的尾调用函数；
- 在大量使用前，把少量数据复制到局部变量，减少别名干扰，改善寄存器分配和自动向量化；
- 手动展开非常热的循环。

例如 CRC 计算中，将一次处理 4 字节改为一次处理 16 字节：

```cpp
#define STEP {                                  \
  uint32 c = l ^ WORD(p);                       \
  p += 4;                                       \
  l = this->table3_[c & 0xff] ^                 \
      this->table2_[(c >> 8) & 0xff] ^          \
      this->table1_[(c >> 16) & 0xff] ^         \
      this->table0_[c >> 24];                   \
}

while ((e - p) >= 16) {
  STEP;
  STEP;
  STEP;
  STEP;
}
#undef STEP
```

此外，文章还提到通过将 `ABSL_LOG(FATAL)` 转换为 `ABSL_DCHECK(false)`，避免在热函数中引入 fatal logging 带来的栈帧设置成本。

## 7.8 降低统计信息收集成本

统计信息很有用，但维护统计也可能很贵。

常见方法是采样：

- 只对部分 RPC 请求记录详细统计；
- 降低采样率；
- 对高频事件避免每次都更新复杂 histogram；
- 删除没人使用的统计项。

## 7.9 避免在热路径上日志记录

日志语句可能代价很高，即使最终没有真正写日志。

例如 `ABSL_VLOG` 至少可能涉及一次加载和一次比较，还可能阻碍编译器优化。

对于极热路径，如果日志没有实际价值，应考虑移除。

# 8. 代码大小方面的考量

性能不只等于运行速度。代码大小也会影响性能。

代码体积过大会带来：

- 编译和链接变慢；
- 二进制膨胀；
- 内存占用增加；
- I-cache 压力增加；*(I-cache：Instruction Cache，指令 Cache；D-cache：Data Cache，数据 Cache)*
- 分支预测器等微架构结构受到负面影响。

这对低层库、模板代码和大量内联代码尤其重要。

## 8.1 裁剪常见的内联代码

被大量调用的函数一旦内联，会在许多调用点复制代码，显著增加二进制大小。

因此，fast path 要尽量小；慢路径最好拆到非内联函数中。

## 8.2 谨慎使用内联

内联通常可以减少函数调用成本，但也会增加代码大小。

当代码变大到影响 I-cache 时，内联反而可能降低性能。

文章举到的例子包括：

- 在不使用 protobuf arena 时，不要内联字符串分配和释放；
- 创建接受 `const char*` 的非内联变体，避免每个调用点都生成 `std::string` 构造代码。

示意：

```cpp
class OpDefBuilderWrapper {
 public:
  explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}

  OpDefBuilderWrapper& Attr(std::string spec) {
    builder_.Attr(std::move(spec));
    return *this;
  }

  OpDefBuilderWrapper& Attr(const char* spec) TF_ATTRIBUTE_NOINLINE {
    return Attr(std::string(spec));
  }
};
```

## 8.3 减少模板实例化

模板代码会针对模板参数组合生成多份代码。

如果模板体积大、实例化组合多，代码大小会迅速膨胀。

常见做法：

- 用普通函数参数替换不必要的模板参数；
- 将大段逻辑从模板构造函数移到非模板共享基类中；
- 慢路径放到非模板非内联函数里。

一个例子：

``` cpp
// sharding_util_ops.cc

// 修改前
template <typename Device, typename T>
class XlaSplitNDBaseOp : public OpKernel {
public:
explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(
    ctx, GetAndValidateAttributes(/*split=*/true, ctx, num_splits_,
                                  num_slices_, paddings_, has_paddings_));
}

// 修改后
// 用于节省代码空间的共享基类
class XlaSplitNDShared : public OpKernel {
public:
explicit XlaSplitNDShared(OpKernelConstruction* ctx) TF_ATTRIBUTE_NOINLINE
  : OpKernel(ctx),
    num_slices_(1),
    has_paddings_(false) {
  GetAndValidateAttributes(/*split=*/true, ctx, num_splits_, num_slices_,
                           paddings_, has_paddings_);
}
```

## 8.4 减少容器操作生成的代码

map 和其他容器操作可能生成大量代码。

文章举到的例子：

- 将一连串初始化 emoji 字符哈希表的 map 插入调用，改成一次批量插入，某个库中的 text 从 188KB 降到 360 字节；
- 停止内联一个大量使用 `InlinedVector` 操作的函数。

# 9. 并行化与同步

> 本文编者注：
> 并行化要思考两个方面：
> 1. 找到可并行的点，如果没有明显的点，可能需要通过额外开发来找到这个点；
> 2. 并行化涉及到的线程切换等 overhead。

## 9.1 利用并行性

现代机器有很多核心，但核心经常未被充分利用。可以通过并行处理不同项来加速昂贵工作，并在完成后合并结果。

但要注意：如果系统没有空闲 CPU，或者已经受限于内存带宽，并行化可能没有帮助，甚至有害。

## 9.2 摊销锁获取成本

避免过细粒度加锁。一次锁保护更多工作，可以减少热路径中的 mutex 操作次数。

但前提是：这样做不会显著增加锁竞争。

## 9.3 保持临界区短小

不要在临界区里做昂贵工作。

尤其要注意那些看起来无害、但内部可能执行 RPC、访问文件或触发复杂逻辑的代码。

## 9.4 通过分片减少竞争

如果一个由 mutex 保护的数据结构竞争很高，可以考虑拆成多个 shard，每个 shard 一个 mutex。

前提是不同 shard 之间没有必须同时维护的跨分片不变量。

如果数据结构是 map，也可以考虑并发哈希表。

## 9.5 SIMD 指令

现代 CPU 支持用 SIMD 指令一次处理多个条目。

典型例子是 `absl::flat_hash_map` 的 Swiss Table：用 SIMD 一次比较多个控制字节，从而快速定位候选槽位。

## 9.6 减少伪共享

如果不同线程访问不同的可变数据，应考虑将这些数据放到不同 cache line 上。

C++ 中可以使用：

```cpp
alignas(ABSL_CACHELINE_SIZE)
```

但它容易被误用，因为会增加对象大小。只有在测量证明 false sharing 确实是问题时才应使用。

示例：将频繁变更的 histogram 字段与其他字段隔离到不同 cache line：

```cpp
#ifndef SWIG
alignas(ABSL_CACHELINE_SIZE)
#endif
std::vector<double> buckets_;
double min_;
double max_;
double count_;
double sum_;
double sum_of_squares_;
```

## 9.7 减少上下文切换频率

如果工作项很小，把它提交到线程池的调度成本可能超过工作本身。

这种情况下，直接内联执行可能更快。

## 9.8 使用带缓冲的 channel 进行流水线处理

无缓冲 channel 会让写入者阻塞，直到读取者准备好接收。

如果 channel 的目的是同步，这很好；但如果目的是提升流水线并行度，就应该考虑使用带缓冲的 channel，让生产者和消费者短暂解耦。

> 本文编者注：
> 如果 channel 是为了做 pipeline，让不同阶段并行推进，就应该给 channel 一点 buffer；否则无缓冲 channel 会让上下游强同步，降低并行度。

## 9.9 考虑无锁方法

无锁数据结构有时比 mutex 保护的数据结构更快。

但直接操作原子变量很危险，应优先使用更高层次、经过验证的抽象。

# 10. Protocol Buffer 建议

Protobuf 是一种方便的数据表示形式，尤其适合需要跨网络传输或持久化存储的数据。

但它不一定适合作为热路径内存表示。文章中有一个例子：填充 1000 个点并对 Y 坐标求和，从 protobuf 改成结构体的 `std::vector` 后，速度提升了 20 倍。

此外，protobuf 还会增加二进制代码和数据大小。在包含大量 protobuf 类型的系统中，这些开销会迅速累积，带来 I-cache 和 D-cache 压力。

一些建议：

- **不要过度地使用 protobuf**。如果数据从不会被序列化或解析，就不一定应该放进 protobuf。
- **避免不必要的消息层级**。额外消息层级会带来分配、函数调用、cache miss 和更大的序列化消息。
- **频繁字段使用较小字段编号**。字段编号 1 到 15 编码为 1 字节；16 到 2047 编码为 2 字节。
- **为性能敏感的未来扩展预留小字段编号**。
- **谨慎选择 int32、sint32、fixed32、uint32 等类型**。小整数适合 varint，大值或哈希值可能适合 fixed 类型，经常为负的值适合 sint。
- **打包 repeated 数值字段**。proto2 中可使用 `[packed=true]`，proto3 默认打包。
- **二进制数据和大值使用 bytes，而不是 string**。
- **考虑 `string_type = VIEW` 避免复制**，但要保证 backing string 生命周期足够长。
- **大字段可考虑 Cord**，降低复制和追加成本。
- **C++ 中使用 protobuf arena**，节省分配和释放成本，改善局部性。
- **保持 `.proto` 文件小巧**，避免单个文件中放太多消息类型。
- **长期持有大量 proto 时，可以考虑以内存中的序列化形式保存**，而不是展开成大量 protobuf 对象。
- **避免 protobuf map 字段**。优先使用由 protobuf 内容初始化的非 protobuf map。
- **如果只访问大消息中的少数字段，可以定义一个只包含所需字段的新 message**。
- **尽可能复用 protobuf 对象**，例如在循环外声明，复用已分配存储。

# 11. C++ 专项建议

## 11.1 `absl::flat_hash_map` / `absl::flat_hash_set`

Absl 哈希表通常优于 C++ 标准库容器，例如 `std::map` 和 `std::unordered_map`。

它们基于 Swiss Table 设计，有更好的 cache locality 和批量探测能力。

> 本文编者注：
> absl 库和 std 库容器的粗略对应关系：
> `absl::btree_map/set` -> `std::map/set`
> `absl::flat_hash_map/set` -> `std::unordered_map/set`

## 11.2 `absl::btree_map` / `absl::btree_set`

`absl::btree_map` 和 `absl::btree_set` 在每个树节点中存储多个条目。

相比 `std::map` / `std::set`，它们通常有：

- 更少指针开销；
- 更少独立分配；
- 更好的 cache locality；
- 更快的有序遍历。

适合需要 ordered map/set 语义，但不依赖节点地址稳定性的场景。

## 11.3 `util::bitmap::InlinedBitVector`

`InlinedBitVector` 可以内联存储较短的位向量，通常比 `std::vector<bool>` 或其他位图类型更适合小整数集合。

## 11.4 `absl::InlinedVector`

`absl::InlinedVector<T, N>` 会在对象内部内联存储少量元素。

当元素数量不超过 `N` 时，它可以完全避免堆分配。适合“通常很小，偶尔变大”的 vector 场景。

## 11.5 `gtl::vector32`

`gtl::vector32` 是一种自定义 vector，使用 32 位大小字段来节省对象头部空间。

如果系统中有大量 vector 实例，且每个 vector 的大小不可能超过 32 位范围，这类优化可能带来非常大的内存收益。

## 11.6 `gtl::small_map`

`gtl::small_map` 使用内联数组存储少量键值对，空间用尽后再升级为指定 map 类型。

适合通常元素很少、偶尔变多的 map 场景。

## 11.7 `gtl::small_ordered_set`

`gtl::small_ordered_set` 针对通常较小的有序集合优化。

它先用固定数组存储少量元素，空间不足时再 fallback 到 `set` 或 `multiset`。相比直接使用树结构，能减少 cache footprint 和临界区长度。

## 11.8 `gtl::intrusive_list`

`gtl::intrusive_list<T>` 把链表指针嵌入到元素 `T` 内部。

相比 `std::list<T*>`，它可以为每个元素节省一次独立链表节点分配、一个 cache line，以及一次间接访问。

## 11.9 限制 `absl::Status` 和 `absl::StatusOr` 的使用

`absl::Status` 和 `absl::StatusOr` 已经比较高效，但成功路径上仍然不是零成本。

对于不会失败，或者不需要返回有意义错误细节的热路径函数，可以考虑使用更轻量的返回形式。

# 12. 批量操作

如果可能，一次处理很多项，而不是一次只处理一个项。

批量操作可以摊销：

- 函数调用成本；
- 边界检查成本；
- 锁成本；
- 分配成本；
- SIMD 加速机会。

例如 `absl::flat_hash_map` 可以使用单条 SIMD 指令，在一组槽位中同时比较多个哈希控制字节：

```cpp
BitMask<uint32_t> Match(h2_t hash) const {
  auto ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
  auto match = _mm_set1_epi8(hash);
  return BitMask<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(match, ctrl)));
}
```

这就是批量处理在底层数据结构中的典型体现。

# 13. 展示多种技术的 CL

原文最后专门列了一些同时使用多种技术的 CL（Change List，代码变更案例）。

真实工程里的性能优化，往往不是单点技巧，而是组合拳：

- 换数据结构；
- 减少分配；
- 减少复制；
- 改 API；
- 拆 hot path / cold path；
- 降低统计和日志成本；
- 调整内存布局；
- 批量化处理。

观察这些真实变更，有助于培养性能优化的思维方式：

> 当一个系统部分被识别为瓶颈后，如何从结构上、接口上、数据表示上同时下手。

# 14. 下期预告

看完 Jeff Dean 和 Sanjay Ghemawat 这篇文章后，还有很多点值得继续深挖。后续我会围绕这些问题继续展开，包括但不限于：

- Floyd's heap construction 算法；
- `std::string_view`、`const std::string&`、`const char*` 的区别；
- Abseil 容器的设计：`flat_hash_map`、`InlinedVector`、`btree_map`、`intrusive_list`、`small_map`；
- `alignas`机制

# 15. 进一步阅读

原文推荐了一些性能相关资料，我也按原链接保留在这里：

- Agner Fog 的 [《Optimizing software in C++》](https://www.agner.org/optimize/optimizing_cpp.pdf)：大量底层性能优化技术。
- Richard L. Sites 的 [《Understanding Software Dynamics》](https://www.oreilly.com/library/view/understanding-software-dynamics/9780137589692/)：诊断和修复性能问题的专家方法。
- [Performance tips of the week](https://abseil.io/fast/)：Abseil 的性能技巧合集。
- [Performance Matters](https://travisdowns.github.io/)：Travis Downs 的性能文章。
- [Daniel Lemire 的博客](https://lemire.me/blog/)：高性能算法实现与底层优化。
- [Building Software Systems at Google and Lessons Learned](https://www.youtube.com/watch?v=modXC5IWTJI)：Google 多年系统性能问题经验。
- Jon Bentley 的 [《Programming Pearls》](https://books.google.com/books/about/Programming_Pearls.html?id=kse_7qbWbjsC) 和 [《More Programming Pearls: Confessions of a Coder》](https://books.google.com/books/about/More_Programming_Pearls.html?id=a2AZAQAAIAAJ)：从算法到简单高效实现的经典文章。
- Henry S. Warren 的 [《Hacker’s Delight》](https://en.wikipedia.org/wiki/Hacker%27s_Delight)：位运算和算术技巧。
- John L. Hennessy 和 David A. Patterson 的 [《Computer Architecture: A Quantitative Approach》](https://books.google.com/books/about/Computer_Architecture.html?id=cM8mDwAAQBAJ)：体系结构经典书，涵盖 cache、分支预测、TLB 等。
