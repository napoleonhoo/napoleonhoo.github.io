性能提示

Jeff Dean、Sanjay Ghemawat

原始版本：2023/07/27，最后更新：2025/12/16

多年来，我们（Jeff 和 Sanjay）对各种代码片段的性能调优做了相当多的深入研究；从 Google 最早期开始，提升软件性能就一直很重要，因为它让我们能为更多用户做更多事情。我们撰写本文档，是为了总结在做这类工作时使用的一些通用原则和具体技术，并尽量挑选有说明性的源码变更（变更列表，或 CL），为各种方法和技术提供示例。下面的大多数具体建议都引用 C++ 类型和 CL，但这些通用原则也适用于其他语言。本文档聚焦于单个二进制文件上下文中的通用性能调优，不涵盖分布式系统或机器学习（ML）硬件性能调优（这些本身就是庞大的领域）。我们希望其他人会觉得它有用。

本文档中的许多示例都包含演示这些技术的代码片段（点击小三角！）。请注意，其中一些代码片段提到了 Google 内部代码库中的各种抽象。如果我们认为这些示例足够自包含，能让不熟悉这些抽象细节的人理解，我们仍然将它们包含在内。

思考性能的重要性

Knuth 常被断章取义地引用为说过：过早优化是万恶之源。完整引文是：“我们应该忘掉小的效率问题，大约 97% 的时间都应如此：过早优化是万恶之源。然而，我们不应错过那关键 3% 中的机会。”本文档讲的正是那关键的 3%；另一段同样来自 Knuth、也更有说服力的引文如下：

从示例 2 到示例 2a 的速度提升只有大约 12%，很多人会称其微不足道。当今许多软件工程师共享的传统观点主张忽略小处的效率；但我认为这只是对那些目光短浅的程序员滥用优化的过度反应，这些程序员无法调试或维护他们“优化”过的程序。在成熟的工程学科中，一个容易获得的 12% 改进从不会被认为是边际的；我认为同样的观点也应适用于软件工程。当然，我不会在一次性任务上费心做这种优化，但当问题是准备高质量程序时，我不想把自己限制在那些让我无法获得此类效率的工具上。

很多人会说：“我们先用尽可能简单的方式写代码，等之后能进行性能剖析时再处理性能。”然而，这种做法通常是错误的：

- 如果在开发大型系统时忽略所有性能问题，最终会得到一个平坦的剖析结果，其中没有明显热点，因为性能损失散布在各处。要弄清楚从哪里开始做性能改进会很困难。

- 如果你在开发一个会被其他人使用的库，遇到性能问题的人很可能是那些不容易进行性能改进的人（他们必须理解其他人/团队编写的代码细节，并且必须和他们协商性能优化的重要性）。

- 当系统处于大量使用状态时，要对其进行重大更改会更困难。

- 另外，也很难判断是否存在可以轻松解决的性能问题，因此我们最终可能会采用代价较高的解决方案，例如过度复制，或为了处理负载问题而对服务进行严重的过度配置。

相反，我们建议在编写代码时，如果更快的替代方案不会显著影响代码的可读性/复杂度，就尽量选择它。

估算

如果你能培养出一种直觉，知道性能在你正在编写的代码中可能有多重要，你就能做出更明智的决定（例如，为了性能而引入多少额外复杂度是合理的）。在编写代码时估算性能的一些提示：

- 这是测试代码吗？如果是，你主要需要关注算法和数据结构的渐近复杂度。（顺带一提：开发周期时间很重要，因此应避免编写需要很长时间运行的测试。）

- 这是特定于某个应用的代码吗？如果是，试着弄清楚性能对这段代码有多重要。这通常并不困难：只要判断代码是初始化/设置代码，还是最终会出现在热路径上的代码（例如，处理服务中的每个请求），通常就足够了

- 这是会被许多应用使用的库代码吗？在这种情况下，很难判断它将来会变得多么敏感。这时，遵循本文档中描述的一些简单技术就显得尤为重要。例如，如果你需要存储一个通常只有少量元素的 vector，请使用 absl::InlinedVector，而不是 std::vector。这类技术并不难遵循，也不会给系统增加任何非局部复杂度。而且，如果事实证明你正在编写的代码最终确实会使用大量资源，它从一开始就会具有更高性能。并且在查看剖析结果时，也会更容易找到下一个需要关注的点。

在可能具有不同性能特征的选项之间做选择时，你可以依赖粗略估算来做稍微更深入的分析。这类计算可以快速给出不同替代方案性能的非常粗略估计，其结果可用于排除其中一些替代方案，而无需先实现它们。

下面是这种估算可能如何进行：

- 估算需要多少各种类型的低级操作，例如磁盘寻道次数、网络往返次数、传输的字节数等。

- 将每种昂贵操作的数量乘以其粗略成本，并把结果相加。

- 前面的结果给出了系统在资源使用方面的成本。如果你关注的是延迟，并且系统存在任何并发性，那么部分成本可能会重叠，你可能需要做稍微更复杂的分析来估算延迟。

下表是 Stanford University 2007 年一次演讲中表格的更新版本（2007 年演讲的视频已不存在，但有一个相关的 2011 年 Stanford 演讲视频，涵盖了一些相同内容）。它列出了应考虑的操作类型及其粗略成本，因此可能很有用：

L1 缓存引用                                0.5 ns

L2 缓存引用                                3 ns

分支预测失败                               5 ns

Mutex lock/unlock（无竞争）               15 ns

主内存引用                                 50 ns

使用 Snappy 压缩 1K 字节                   1,000 ns

从 SSD 读取 4KB                            20,000 ns

同一数据中心内往返                         50,000 ns

从内存顺序读取 1MB                         64,000 ns

通过 100 Gbps 网络读取 1MB                 100,000 ns

从 SSD 读取 1MB                            1,000,000 ns

磁盘寻道                                   5,000,000 ns

从磁盘顺序读取 1MB                         10,000,000 ns

发送数据包 CA->Netherlands->CA            150,000,000 ns

前面的表格包含一些基本低级操作的粗略成本。你可能还会发现，跟踪与你的系统相关的更高级操作的估算成本也很有用。例如，你可能想知道从 SQL 数据库进行一次点读取的粗略成本、与某个 Cloud 服务交互的延迟，或渲染一个简单 HTML 页面所需的时间。如果你不知道不同操作的相关成本，就无法进行像样的粗略估算！

示例：对十亿个 4 字节数字进行快速排序所需的时间

粗略地说，一个好的快速排序算法会对大小为 N 的数组进行 log(N) 趟扫描。每一趟中，数组内容会从内存流式进入处理器缓存，分区代码会将每个元素与一个枢轴元素比较一次。让我们把主要成本加总：

- 内存带宽：该数组占用 4 GB（每个数字 4 字节乘以十亿个数字）。假设每个核心的内存带宽约为 ~16GB/s。这意味着每一趟需要 ~0.25s。N 约为 ~2^30，因此我们会进行 ~30 趟，所以内存传输的总成本约为 ~7.5 秒。

- 分支预测失败：我们总共会进行 N*log(N) 次比较，也就是约 ~300 亿次比较。假设其中一半（即 150 亿次）预测失败。按每次预测失败 5 ns 相乘，可得到 75 秒的预测失败成本。在此分析中，我们假设正确预测的分支是免费的。

- 将前面的数字相加，我们得到约 ~82.5 秒的估算值。

如有必要，我们可以细化分析以考虑处理器缓存。根据上面的分析，分支预测失败是主导成本，因此这种细化可能并不需要，但我们仍将其作为另一个示例包含在这里。假设我们有一个 32MB L3 缓存，并且从 L3 缓存向处理器传输数据的成本可以忽略不计。L3 缓存可以容纳 2^23 个数字，因此最后 22 趟可以在驻留于 L3 缓存中的数据上运行（倒数第 23 趟将数据带入 L3 缓存，剩余趟则在该数据上运行）。这将内存传输成本从 7.5 秒（30 次 4GB 内存传输，速度为 16GB/s）降低到 2.5 秒（10 次 4GB 内存传输，速度为 16GB/s）。

示例：生成包含 30 个图像缩略图的网页所需的时间

我们来比较两种潜在设计，其中原始图像存储在磁盘上，每张图像大小约为 1MB。

- 串行读取 30 张图像的内容，并为每张图像生成一个缩略图。每次读取需要一次寻道 + 一次传输，总计为寻道 5ms 加上传输 10ms，即 30 张图像乘以每张图像 15ms，也就是 450ms。

- 并行读取，假设图像均匀分布在 K 块磁盘上。前面的资源使用估算仍然成立，但延迟会大约降低 K 倍，这里忽略方差（例如，我们有时会运气不好，某块磁盘上拥有超过所读取图像 1/K 的数量）。因此，如果我们运行在拥有数百块磁盘的分布式文件系统上，预期延迟会降至 ~15ms。

- 让我们考虑一个变体：所有图像都在单个 SSD 上。这会把顺序读取性能变为每张图像 20µs + 1ms，总体加起来约为 ~30 ms。

测量

前一节给出了一些提示，说明在编写代码时如何思考性能，而不必过于担心如何测量你的选择带来的性能影响。然而，在你真正开始改进之前，或者遇到涉及性能、简单性等各种因素的权衡时，你会希望测量或估算潜在的性能收益。在进行性能相关工作时，能够有效地测量事物是你最需要掌握的头号工具。

顺带一提，值得指出的是，对你不熟悉的代码进行性能剖析，也可以很好地帮助你大致了解代码库的结构及其运作方式。检查程序动态调用图中参与较深的例程源码，可以让你从较高层次了解运行代码时“发生了什么”，从而增强你在稍微不熟悉的代码中进行性能改进变更的信心。

性能剖析工具和提示

有许多有用的性能剖析工具可用。首先可以使用的一个有用工具是 pprof，因为它能提供良好的高层次性能信息，并且无论在本地还是在生产环境中运行的代码上都易于使用。如果你想更深入地洞察性能，也可以尝试 perf。

性能剖析的一些提示：

- 使用适当的调试信息和优化标志构建生产二进制文件。

- 如果可以，编写一个覆盖你正在改进的代码的 microbenchmark。microbenchmark 可以在进行性能改进时缩短周转时间，帮助验证性能改进的影响，并有助于防止未来的性能回退。不过，microbenchmark 可能存在一些陷阱，使其无法代表完整系统性能。用于编写 microbenchmark 的有用库：C++ Go Java。

- 使用基准测试库来发出性能计数器读数，既能获得更高精度，也能更深入了解程序行为。

- 锁竞争通常会人为降低 CPU 使用率。一些 mutex 实现支持对锁竞争进行性能剖析。

- 对机器学习性能工作使用 ML profilers。

当剖析结果很平坦时该怎么做

你经常会遇到 CPU 剖析结果很平坦的情况（没有明显导致变慢的主要因素）。这通常会在所有容易摘取的果实都已经被摘掉时发生。如果你发现自己处于这种情况，可以考虑以下提示：

- 不要低估许多小优化的价值！在某个子系统中分别做出二十个 1% 的改进通常完全可行，并且合在一起意味着相当可观的改进（这种类型的工作通常依赖稳定且高质量的 microbenchmark）。这类变更的一些示例见“演示多种技术的变更”一节。

- 找到更靠近调用栈顶部的循环（CPU 剖析结果的火焰图视图在这里可能有帮助）。潜在地，这个循环或它调用的代码可以被重构得更高效。某些代码最初通过遍历输入的节点和边来增量构建一个复杂的图结构，后来改为通过传入整个输入一次性构建图结构。这移除了初始代码中按每条边发生的一堆内部检查。

- 后退一步，寻找调用栈更高层的结构性更改，而不是专注于微优化。在这样做时，“算法改进”下列出的技术可能会有用。

- 寻找过于通用的代码。用定制化或更低层级的实现替换它。例如，如果某个应用反复使用正则表达式匹配，而简单的前缀匹配就足够了，可以考虑放弃使用正则表达式。

- 尝试减少分配次数：获取分配剖析结果，并逐步削减分配次数的最大贡献者。这会产生两个效果：(1) 它会直接减少在分配器（以及对使用 GC 的语言而言的垃圾收集器）中花费的时间 (2) 通常还会减少缓存未命中，因为在使用 tcmalloc 的长时间运行程序中，每次分配往往都会落到不同的缓存行上。

- 收集其他类型的剖析结果，特别是基于硬件性能计数器的剖析结果。这类剖析结果可能指出哪些函数遇到了较高的缓存未命中率。“性能剖析工具和提示”一节中描述的技术可能会有帮助。

API 考量

下面建议的一些技术需要更改数据结构和函数签名，这可能会对调用方造成干扰。尽量组织代码，使建议的性能改进能够在封装边界内完成，而不影响公共接口。如果你的模块足够深（通过窄接口访问大量功能），这会更容易。

广泛使用的 API 会承受很大的新增功能压力。添加新功能时要谨慎，因为这些功能会约束未来的实现，并且会给不需要这些新功能的用户不必要地增加成本。例如，许多 C++ 标准库容器承诺迭代器稳定性，而在典型实现中，这会显著增加分配次数，即使许多用户并不需要指针稳定性。

下面列出了一些具体技术。请仔细权衡性能收益，以及此类变更引入的任何 API 可用性问题。

批量 API

提供批量操作，以减少昂贵的 API 边界跨越，或利用算法改进。

添加批量 MemoryManager::LookupMany 接口。

除了添加批量接口之外，这还简化了新的批量变体的签名：事实证明，客户端只需要知道是否找到了所有键，因此我们可以返回 bool，而不是 Status 对象。

memory_manager.h

class MemoryManager {

public:

...

util::StatusOr<LiveTensor> Lookup(const TensorIdProto& id);

class MemoryManager {

public:

...

util::StatusOr<LiveTensor> Lookup(const TensorIdProto& id);

// 查找标识出的张量

struct LookupKey {

ClientHandle client;

uint64 local_id;

};

bool LookupMany(absl::Span<const LookupKey> keys,

absl::Span<tensorflow::Tensor> tensors);

添加批量 ObjectStore::DeleteRefs API 以分摊锁开销。

object_store.h

template <typename T>

class ObjectStore {

public:

...

absl::Status DeleteRef(Ref);

template <typename T>

class ObjectStore {

public:

...

absl::Status DeleteRef(Ref);

// 删除许多引用。对于每个 ref，如果没有其他 Refs 指向同一个

// object，则该 object 将被删除。任何错误都会返回非 OK。

absl::Status DeleteRefs(absl::Span<const Ref> refs);

...

template <typename T>

absl::Status ObjectStore<T>::DeleteRefs(absl::Span<const Ref> refs) {

util::Status result;

absl::MutexLock l(&mu_);

for (auto ref : refs) {

result.Update(DeleteRefLocked(ref));

}

return result;

}

memory_tracking.cc

void HandleBatch(int, const plaque::Batch& input) override {

for (const auto& t : input) {

auto in = In(t);

PLAQUE_OP_ASSIGN_OR_RETURN(const auto& handles, in.handles());

for (const auto handle : handles.value->handles()) {

PLAQUE_OP_RETURN_IF_ERROR(in_buffer_store_

? bstore_->DeleteRef(handle)

: tstore_->DeleteRef(handle));

}

}

}

void HandleBatch(int, const plaque::Batch& input) override {

for (const auto& t : input) {

auto in = In(t);

PLAQUE_OP_ASSIGN_OR_RETURN(const auto& handles, in.handles());

if (in_buffer_store_) {

PLAQUE_OP_RETURN_IF_ERROR(

bstore_->DeleteRefs(handles.value->handles()));

} else {

PLAQUE_OP_RETURN_IF_ERROR(

tstore_->DeleteRefs(handles.value->handles()));

}

}

}

使用 Floyd 的堆构造进行高效初始化。

堆的批量初始化可以在 O(N) 时间内完成，而每次添加一个元素并在每次添加后更新堆属性，需要 O(N lg(N)) 时间。

有时很难让调用方直接使用新的批量 API。在这种情况下，内部使用批量 API 并缓存结果，以供未来的非批量 API 调用使用，可能会有好处：

缓存块解码结果，以供未来调用使用。

每次查找都需要解码包含 K 个条目的整个块。将解码后的条目存入缓存，并在后续查找中查询缓存。

lexicon.cc

void GetTokenString(int pos, std::string* out) const {

...

absl::FixedArray<LexiconEntry, 32> entries(pos + 1);

// 解码直到并包括 pos 的所有词典条目。

for (int i = 0; i <= pos; ++i) {

p = util::coding::TwoValuesVarint::Decode32(p, &entries[i].remaining,

&entries[i].shared);

entries[i].remaining_str = p;

p += entries[i].remaining;  // remaining 字节跟在每个条目之后。

}

mutable std::vector<absl::InlinedVector<std::string, 16>> cache_;

...

void GetTokenString(int pos, std::string* out) const {

...

DCHECK_LT(skentry, cache_.size());

if (!cache_[skentry].empty()) {

*out = cache_[skentry][pos];

return;

}

...

// 初始化缓存。

...

const char* prev = p;

for (int i = 0; i < block_sz; ++i) {

uint32 shared, remaining;

p = TwoValuesVarint::Decode32(p, &remaining, &shared);

auto& cur = cache_[skentry].emplace_back();

gtl::STLStringResizeUninitialized(&cur, remaining + shared);

std::memcpy(cur.data(), prev, shared);

std::memcpy(cur.data() + shared, p, remaining);

prev = cur.data();

p += remaining;

}

*out = cache_[skentry][pos];

视图类型

函数参数优先使用视图类型（例如 std::string_view、std::Span<T>、absl::FunctionRef<R(Args...)>）（除非正在转移数据所有权）。这些类型减少复制，并允许调用方选择自己的容器类型（例如，一个调用方可能使用 std::vector，而另一个调用方使用 absl::InlinedVector）。

预分配/预计算参数

对于频繁调用的例程，有时允许更高层调用方传入其拥有的数据结构，或被调用例程需要且客户端已经拥有的信息，会很有用。这可以避免低层例程被迫分配自己的临时数据结构，或重新计算已经可用的信息。

添加 RPC_Stats::RecordRPC 变体，允许客户端传入已经可用的 WallTime 值。

rpc-stats.h

static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m);

static void RecordRPC(const Name &name, const RPC_Stats_Measurement& m,

WallTime now);

clientchannel.cc

const WallTime now = WallTime_Now();

...

RPC_Stats::RecordRPC(stats_name, m);

const WallTime now = WallTime_Now();

...

RPC_Stats::RecordRPC(stats_name, m, now);

线程兼容类型与线程安全类型

一个类型可以是线程兼容的（外部同步），也可以是线程安全的（内部同步）。大多数通用类型应当是线程兼容的。这样，不需要线程安全的调用方就不会为其付出代价。

使一个类线程兼容，因为调用方已经进行了同步。

hitless-transfer-phase.cc

TransferPhase HitlessTransferPhase::get() const {

static CallsiteMetrics cm("HitlessTransferPhase::get");

MonitoredMutexLock l(&cm, &mutex_);

return phase_;

}

TransferPhase HitlessTransferPhase::get() const { return phase_; }

hitless-transfer-phase.cc

bool HitlessTransferPhase::AllowAllocate() const {

static CallsiteMetrics cm("HitlessTransferPhase::AllowAllocate");

MonitoredMutexLock l(&cm, &mutex_);

return phase_ == TransferPhase::kNormal || phase_ == TransferPhase::kBrownout;

}

bool HitlessTransferPhase::AllowAllocate() const {

return phase_ == TransferPhase::kNormal || phase_ == TransferPhase::kBrownout;

}

不过，如果某个类型的典型使用需要同步，则优先将同步移入类型内部。这样可以按需调整同步机制以提升性能（例如，通过分片减少竞争），而不影响调用方。

算法改进

性能提升最关键的机会来自算法改进，例如，将 O(N²) 算法转为 O(N lg(N)) 或 O(N)，避免潜在的指数级行为等。这些机会在稳定代码中很少见，但在编写新代码时值得关注。下面是几个展示对既有代码进行此类改进的示例：

按反向后序将节点添加到环检测结构中。

我们之前是一次一个地向环检测数据结构添加图节点和边，这需要对每条边执行昂贵工作。现在我们按反向后序添加整个图，这使环检测变得非常简单。

graphcycles.h

class GraphCycles : public util_graph::Graph {

public:

GraphCycles();

~GraphCycles() override;

using Node = util_graph::Node;

class GraphCycles : public util_graph::Graph {

public:

GraphCycles();

~GraphCycles() override;

using Node = util_graph::Node;

// InitFrom 会添加来自 src 的所有节点和边；如果成功则返回 true，

// 如果遇到环则返回 false。

// REQUIRES: 尚未向 GraphCycles 添加任何节点和边。

bool InitFrom(const util_graph::Graph& src);

graphcycles.cc

bool GraphCycles::InitFrom(const util_graph::Graph& src) {

...

// 按拓扑顺序分配 rank，这样初始化期间就不需要任何重新排序。

// 对于无环图，DFS 会使节点处于反向拓扑顺序，因此我们在离开节点时

// 为节点分配递减的 rank。

Rank last_rank = n;

auto leave = [&](util_graph::Node node) {

DCHECK(r->rank[node] == kMissingNodeRank);

NodeInfo* nn = &r->nodes[node];

nn->in = kNil;

nn->out = kNil;

r->rank[node] = --last_rank;

};

util_graph::DFSAll(src, std::nullopt, leave);

// 添加所有边（同时检测环）。

bool have_cycle = false;

util_graph::PerEdge(src, [&](util_graph::Edge e) {

DCHECK_NE(r->rank[e.src], kMissingNodeRank);

DCHECK_NE(r->rank[e.dst], kMissingNodeRank);

if (r->rank[e.src] >= r->rank[e.dst]) {

have_cycle = true;

} else if (!HasEdge(e.src, e.dst)) {

EdgeListAddNode(r, &r->nodes[e.src].out, e.dst);

EdgeListAddNode(r, &r->nodes[e.dst].in, e.src);

}

});

if (have_cycle) {

return false;

} else {

DCHECK(CheckInvariants());

return true;

}

}

graph_partitioner.cc

absl::Status MergeGraph::Init() {

const Graph& graph = *compiler_->graph();

clusters_.resize(graph.NodeLimit());

graph.PerNode([&](Node node) {

graph_->AddNode(node);

NodeList* n = new NodeList;

n->push_back(node);

clusters_[node] = n;

});

absl::Status s;

PerEdge(graph, [&](Edge e) {

if (!s.ok()) return;

if (graph_->HasEdge(e.src, e.dst)) return;  // 已添加

if (!graph_->InsertEdge(e.src, e.dst)) {

s = absl::InvalidArgumentError("cycle in the original graph");

}

});

return s;

}

absl::Status MergeGraph::Init() {

const Graph& graph = *compiler_->graph();

if (!graph_->InitFrom(graph)) {

return absl::InvalidArgumentError("cycle in the original graph");

}

clusters_.resize(graph.NodeLimit());

graph.PerNode([&](Node node) {

NodeList* n = new NodeList;

n->push_back(node);

clusters_[node] = n;

});

return absl::OkStatus();

}

用更好的算法替换内置于 mutex 实现中的死锁检测系统。

将死锁检测算法替换为速度约快 50 倍、并且可以毫无问题地扩展到数百万个 mutex 的算法（旧算法依赖 2K 限制来避免性能断崖）。新代码基于以下论文：A dynamic topological sort algorithm for directed acyclic graphs David J. Pearce, Paul H. J. Kelly Journal of Experimental Algorithmics (JEA) JEA Homepage archive Volume 11, 2006, Article No. 1.7

新算法占用 O(|V|+|E|) 空间（而不是旧算法所需的 O(|V|^2) 位）。锁获取顺序图非常稀疏，因此这会少得多。该算法也相当简单：其核心大约是 100 行 C++。由于代码现在可以扩展到数量大得多的 Mutexes，我们能够放宽一个人为的 2K 限制，而这暴露了真实程序中的若干潜在死锁。

基准测试结果：这些是在 DEBUG 模式下运行的，因为死锁检测主要在调试模式下启用。基准测试参数（/2k 等）是跟踪节点的数量。在旧算法默认的 2k 限制下，新算法每次 InsertEdge 只需 0.5 微秒，而旧算法需要 22 微秒。新算法还可以轻松扩展到大得多的图而没有问题，而旧算法会很快崩溃。

DEBUG: 基准测试            时间(ns)    CPU(ns) 迭代次数

----------------------------------------------------------

DEBUG: BM_StressTest/2k        23553      23566      29086

DEBUG: BM_StressTest/4k        45879      45909      15287

DEBUG: BM_StressTest/16k      776938     777472        817

DEBUG: BM_StressTest/2k          392        393   10485760

DEBUG: BM_StressTest/4k          392        393   10485760

DEBUG: BM_StressTest/32k         407        407   10485760

DEBUG: BM_StressTest/256k        456        456   10485760

DEBUG: BM_StressTest/1M          534        534   10485760

将 IntervalMap（O(lg N) 查找）替换为哈希表（O(1) 查找）。

初始代码使用 IntervalMap，因为它看起来像是支持相邻块合并的正确数据结构；但哈希表已经足够，因为相邻块可以通过哈希表查找找到。这一点（加上该变更列表（CL）中的其他变更）将 tpu::BestFitAllocator 的性能提升了约 4X。

best_fit_allocator.h

using Block = gtl::IntervalMap<int64, BlockState>::Entry;

...

// (address range, BlockState) 对的映射，每个 allocation 一个条目，

// 覆盖范围 [0, allocatable_range_end_)。相邻的 kFree 和

// kReserved 块会合并。相邻的 kAllocated 块不会

// 合并。

gtl::IntervalMap<int64, BlockState> block_list_;

// 按 allocation 策略排序的所有空闲块集合。相邻的

// 空闲块会合并。

std::set<Block, BlockSelector> free_list_;

// BlockTable 中 offsets 的更快哈希函数

struct OffsetHash {

ABSL_ATTRIBUTE_ALWAYS_INLINE size_t operator()(int64 value) const {

uint64 m = value;

m *= uint64_t{0x9ddfea08eb382d69};

return static_cast<uint64_t>(m ^ (m >> 32));

}

};

// 哈希表从块起始地址映射到块信息。

// 我们在此信息中包含前一个块的长度，这样就可以

// 找到前面的块以进行合并。

struct HashTableEntry {

BlockState state;

int64 my_length;

int64 prev_length;  // 如果没有前一个块，则为零。

};

using BlockTable = absl::flat_hash_map<int64, HashTableEntry, OffsetHash>;

将排序列表求交（O(N log N)）替换为哈希表查找（O(N)）。

用于检测两个节点是否共享公共源的旧代码，会按排序顺序获取每个节点的源，然后执行排序求交。新代码将一个节点的源放入哈希表，然后遍历另一个节点的源并检查哈希表。

name             old time/op  new time/op  delta

BM_CompileLarge   28.5s ± 2%   22.4s ± 2%  -21.61%  (p=0.008 n=5+5)

实现良好的哈希函数，使操作为 O(1) 而不是 O(N)。

location.h

// Location 对象的 Hasher。

struct LocationHash {

size_t operator()(const Location* key) const {

return key != nullptr ? util_hash::Hash(key->address()) : 0;

}

};

size_t HashLocation(const Location& loc);

...

struct LocationHash {

size_t operator()(const Location* key) const {

return key != nullptr ? HashLocation(*key) : 0;

}

};

location.cc

size_t HashLocation(const Location& loc) {

util_hash::MurmurCat m;

// 将一些更简单的特征编码到单个值中。

m.AppendAligned((loc.dynamic() ? 1 : 0)                    //

| (loc.append_shard_to_address() ? 2 : 0)  //

| (loc.is_any() ? 4 : 0)                   //

| (!loc.any_of().empty() ? 8 : 0)          //

| (loc.has_shardmap() ? 16 : 0)            //

| (loc.has_sharding() ? 32 : 0));

if (loc.has_shardmap()) {

m.AppendAligned(loc.shardmap().output() |

static_cast<uint64_t>(loc.shardmap().stmt()) << 20);

}

if (loc.has_sharding()) {

uint64_t num = 0;

switch (loc.sharding().type_case()) {

case Sharding::kModShard:

num = loc.sharding().mod_shard();

break;

case Sharding::kRangeSplit:

num = loc.sharding().range_split();

break;

case Sharding::kNumShards:

num = loc.sharding().num_shards();

break;

default:

num = 0;

break;

}

m.AppendAligned(static_cast<uint64_t>(loc.sharding().type_case()) |

(num << 3));

}

auto add_string = [&m](absl::string_view s) {

if (!s.empty()) {

m.Append(s.data(), s.size());

}

};

add_string(loc.address());

add_string(loc.lb_policy());

// 我们不包含 any_of，因为计算一个对顺序和重复

// 不敏感的哈希值很复杂。

return m.GetHash();

}

更好的内存表示

仔细考虑重要数据结构的内存占用和缓存占用，通常可以带来很大的节省。下面的数据结构侧重于通过触及更少的缓存行来支持常见操作。这里的谨慎处理可以 (a) 避免代价高昂的缓存未命中，(b) 减少内存总线流量，从而加快相关程序以及同一台机器上运行的其他任何程序的速度。它们依赖一些常见技术，你在设计自己的数据结构时可能会觉得有用。

紧凑数据结构

对于会被频繁访问，或占应用内存使用很大一部分的数据，使用紧凑表示。紧凑表示可以通过触及更少的缓存行并减少内存总线带宽使用，显著降低内存使用并提升性能。不过，要注意缓存行争用。

内存布局

仔细考虑具有较大内存或缓存占用的类型的内存布局。

- 重新排序字段，以减少具有不同对齐要求的字段之间的填充（参见 class layout discussion）。

- 在存储的数据适合较小类型时，使用更小的数值类型。

- 如果不谨慎，枚举值有时会占用一个完整字。考虑使用更小的表示形式（例如，使用 enum class OpType : uint8_t { ... } 而不是 enum class OpType { ... }）。

- 对字段排序，使经常一起访问的字段彼此更靠近——这会减少常见操作触及的缓存行数量。

- 将热的只读字段放在远离热的可变字段的位置，这样对可变字段的写入就不会导致只读字段从附近缓存中被驱逐。

- 移动冷数据，使其不要紧挨着热数据存放，可以将冷数据放在 struct 的末尾，或放在一层间接访问之后，或放在单独的数组中。

- 考虑使用位级和字节级编码，将内容打包到更少的字节中。这可能会很复杂，因此只有当相关数据被封装在经过充分测试的模块内，并且总体内存使用减少显著时，才这样做。此外，要注意副作用，例如频繁使用的数据对齐不足，或访问打包表示的代码代价更高。使用基准测试验证此类变更。

用索引代替指针

在现代 64 位机器上，指针占用 64 位。如果你的数据结构富含指针，很容易因为 T* 的间接引用消耗大量内存。相反，可以考虑使用指向数组 T[] 或其他数据结构的整数索引。这样不仅引用会更小（如果索引数量少到可以放入 32 位或更少位），而且所有 T[] 元素的存储将是连续的，通常会带来更好的缓存局部性。

批量存储

避免使用每个存储元素都分配一个单独对象的数据结构（例如 C++ 中的 std::map、std::unordered_map）。相反，考虑使用分块或扁平表示、将多个元素存储在内存中相近位置的类型（例如 C++ 中的 std::vector、absl::flat_hash_{map,set}）。这类类型往往具有好得多的缓存行为。此外，它们遇到的分配器开销也更少。

一种有用的技术是将元素划分为块，每个块可以容纳固定数量的元素。该技术可以在保持良好渐近行为的同时，显著降低数据结构的缓存占用。

对于某些数据结构，单个块就足以容纳所有元素（例如字符串和 vector）。其他类型（例如 absl::flat_hash_map）也使用这种技术。

内联存储

某些容器类型针对存储少量元素进行了优化。这些类型在顶层为少量元素提供空间，并在元素数量较小时完全避免分配。当这类类型的实例经常被构造（例如作为频繁执行代码中的栈变量）时，或当许多实例同时存活时，这会非常有帮助。如果一个容器通常会包含少量元素，请考虑使用一种内联存储类型，例如 InlinedVector。

注意：如果 sizeof(T) 很大，内联存储容器可能不是最佳选择，因为内联后备存储会很大。

不必要的嵌套 map

有时，嵌套 map 数据结构可以替换为带有复合键的单层 map。这可以显著降低查找和插入成本。

通过将 btree<a,btree<b,c>> 转换为 btree<pair<a,b>,c>，减少分配并改善缓存占用。

graph_splitter.cc

absl::btree_map<std::string, absl::btree_map<std::string, OpDef>> ops;

// btree 从 {package_name, op_name} 映射到其 const Opdef*。

absl::btree_map<std::pair<absl::string_view, absl::string_view>,

const OpDef*>

ops;

注意：如果第一个 map 键很大，保留嵌套 map 可能更好：

切换到嵌套 map 后，微基准测试中的性能提升了 76%。

我们之前有一个单层哈希表，其键由一个（字符串）路径和一些其他数值子键组成。每个路径平均大约出现在 1000 个键中。我们将哈希表拆分为两层，第一层以路径为键，每个第二层哈希表只保留特定路径的子键到数据的映射。这使存储路径所需的内存使用减少了 1000 倍，并且在一起访问同一路径的许多子键时也加快了访问速度。

Arenas

Arena 可以帮助降低内存分配成本，但它们还有一个好处：将彼此独立分配的项紧密打包在一起，通常放入更少的缓存行，并消除大多数析构成本。对于具有许多子对象的复杂数据结构，它们很可能最有效。考虑为 arena 提供合适的初始大小，因为这有助于减少分配。

注意：arena 很容易被误用，例如将太多短生命周期对象放入长生命周期 arena 中，这会不必要地膨胀内存占用。

用数组代替 map

如果 map 的定义域可以用小整数表示，或是 enum，或者该 map 只有很少的元素，那么有时可以用数组或某种形式的 vector 替代 map。

使用数组代替 flat_map。

rtp_controller.h

const gtl::flat_map<int, int> payload_type_to_clock_frequency_;

// 一个映射（实现为简单数组），由 payload_type 索引到时钟频率

// 用于该 payload type（或 0）

struct PayloadTypeToClockRateMap {

int map[128];

};

...

const PayloadTypeToClockRateMap payload_type_to_clock_frequency_;

用 bit vector 代替 set

如果 set 的定义域可以用小整数表示，则可以用 bit vector（InlinedBitVector 通常是不错的选择）替代该 set。在这些表示上，set 操作也可以使用按位布尔运算（OR 表示并集，AND 表示交集，等等）非常高效地完成。

Spanner 放置系统。将 dense_hash_set<ZoneId> 替换为每个 zone 一个 bit 的 bit-vector。

zone_set.h

class ZoneSet: public dense_hash_set<ZoneId> {

public:

...

bool Contains(ZoneId zone) const {

return count(zone) > 0;

}

class ZoneSet {

...

// 当且仅当 "zone" 包含在集合中时返回 true

bool ContainsZone(ZoneId zone) const {

return zone < b_.size() && b_.get_bit(zone);

}

...

private:

int size_;          // 已插入的 zone 数量

util::bitmap::InlinedBitVector<256> b_;

基准测试结果：

CPU: AMD Opteron (4 cores) dL1:64KB dL2:1024KB

Benchmark                          Base (ns)  New (ns) Improvement

------------------------------------------------------------------

BM_Evaluate/1                            960       676    +29.6%

BM_Evaluate/2                           1661      1138    +31.5%

BM_Evaluate/3                           2305      1640    +28.9%

BM_Evaluate/4                           3053      2135    +30.1%

BM_Evaluate/5                           3780      2665    +29.5%

BM_Evaluate/10                          7819      5739    +26.6%

BM_Evaluate/20                         17922     12338    +31.2%

BM_Evaluate/40                         36836     26430    +28.2%

使用 bit matrix 跟踪操作数之间的可达性属性，而不是使用 hash table。

hlo_computation.h

using TransitiveOperandMap =

std::unordered_map<const HloInstruction*,

std::unordered_set<const HloInstruction*>>;

class HloComputation::ReachabilityMap {

...

// 从 HloInstruction* 到编号的密集 id 分配

tensorflow::gtl::FlatMap<const HloInstruction*, int> ids_;

// 当且仅当 b 可从 a 到达时，matrix_(a,b) 为 true

tensorflow::core::Bitmap matrix_;

};

减少分配

内存分配会增加成本：

- 它会增加分配器中花费的时间。

- 新分配的对象可能需要代价高昂的初始化，并且在不再需要时有时还需要相应的高成本析构。

- 每次分配往往位于新的缓存行上，因此分散在许多独立分配中的数据，会比分散在较少分配中的数据具有更大的缓存占用。

垃圾回收运行时有时会通过将连续分配按顺序放在内存中来消除问题 #3。

避免不必要的分配

减少分配使基准测试吞吐量提高了 21%。

memory_manager.cc

LiveTensor::LiveTensor(tf::Tensor t, std::shared_ptr<const DeviceInfo> dinfo,

bool is_batched)

: tensor(std::move(t)),

device_info(dinfo ? std::move(dinfo) : std::make_shared<DeviceInfo>()),

is_batched(is_batched) {

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

在可能时使用静态分配的零 vector，而不是分配一个 vector 并用零填充它。

embedding_executor_8bit.cc

// EmbeddingLookUpT 的实际实现，使用模板参数

// 而不是对象成员来提升性能。

template <bool Mean, bool SymmetricInputRange>

static tensorflow::Status EmbeddingLookUpT(...) {

...

std::unique_ptr<tensorflow::quint8[]> zero_data(

new tensorflow::quint8[max_embedding_width]);

memset(zero_data.get(), 0, sizeof(tensorflow::quint8) * max_embedding_width);

// 足够大以处理大多数 embedding 宽度的大小

static const int kTypicalMaxEmbedding = 256;

static tensorflow::quint8 static_zero_data[kTypicalMaxEmbedding];  // 全部为零

...

// EmbeddingLookUpT 的实际实现，使用模板参数

// 而不是对象成员来提升性能。

template <bool Mean, bool SymmetricInputRange>

static tensorflow::Status EmbeddingLookUpT(...) {

...

std::unique_ptr<tensorflow::quint8[]> zero_data_backing(nullptr);

// 获取指向内存区域的指针，该区域至少包含

// "max_embedding_width" 个 quint8 零值。

tensorflow::quint8* zero_data;

if (max_embedding_width <= ARRAYSIZE(static_zero_data)) {

// static_zero_data 足够大，因此不需要分配 zero data

zero_data = &static_zero_data[0];

} else {

// static_zero_data 不够大：我们需要分配 zero data

zero_data_backing =

absl::make_unique<tensorflow::quint8[]>(max_embedding_width);

memset(zero_data_backing.get(), 0,

sizeof(tensorflow::quint8) * max_embedding_width);

zero_data = zero_data_backing.get();

}

此外，当对象生命周期受作用域限制时，优先使用栈分配而不是堆分配（不过要注意大型对象的栈帧大小）。

调整容器大小或预留容量

当 vector（或其他某些容器类型）的最大大小或预期最大大小已提前知道时，预先设置容器的后备存储大小（例如在 C++ 中使用 resize 或 reserve）。

预先设置 vector 的大小并填充它，而不是执行 N 次 push_back 操作。

indexblockdecoder.cc

for (int i = 0; i < ndocs-1; i++) {

uint32 delta;

ERRORCHECK(b->GetRice(rice_base, &delta));

docs_.push_back(DocId(my_shard_ + (base + delta) * num_shards_));

base = base + delta + 1;

}

docs_.push_back(last_docid_);

docs_.resize(ndocs);

DocId* docptr = &docs_[0];

for (int i = 0; i < ndocs-1; i++) {

uint32 delta;

ERRORCHECK(b.GetRice(rice_base, &delta));

*docptr = DocId(my_shard_ + (base + delta) * num_shards_);

docptr++;

base = base + delta + 1;

}

*docptr = last_docid_;

注意：不要使用 resize 或 reserve 每次只增长一个元素，因为这可能导致二次方行为。另外，如果元素构造代价高，优先先调用一次 reserve，然后进行多次 push_back 或 emplace_back 调用，而不是先调用一次 resize，因为后者会使构造函数调用次数翻倍。

尽可能避免复制

- 尽可能优先移动数据结构，而不是复制数据结构。

- 如果生命周期不是问题，在临时数据结构中存储指针或索引，而不是对象副本。例如，如果使用本地 map 从传入的 proto 列表中选择一组 proto，我们可以让该 map 只存储指向传入 proto 的指针，而不是复制可能深度嵌套的数据。另一个常见例子是对索引 vector 进行排序，而不是直接对大型对象 vector 排序，因为后者会产生显著的复制/移动成本。

通过 gRPC 接收 tensor 时避免额外复制。

发送约 400KB tensor 的基准测试加速了约 10-15%：

Benchmark              Time(ns)    CPU(ns) Iterations

-----------------------------------------------------

BM_RPC/30/98k_mean    148764691 1369998944       1000

Benchmark              Time(ns)    CPU(ns) Iterations

-----------------------------------------------------

BM_RPC/30/98k_mean    131595940 1216998084       1000

移动大型 options 结构，而不是复制它。

index.cc

return search_iterators::DocPLIteratorFactory::Create(opts);

return search_iterators::DocPLIteratorFactory::Create(std::move(opts));

使用 std::sort 而不是 std::stable_sort，这可以避免 stable sort 实现中的内部复制。

encoded-vector-hits.h

std::stable_sort(hits_.begin(), hits_.end(),

gtl::OrderByField(&HitWithPayloadOffset::docid));

struct HitWithPayloadOffset {

search_iterators::LocalDocId64 docid;

int first_payload_offset;  // payload vector 中的偏移量。

int num_payloads;

bool operator<(const HitWithPayloadOffset& other) const {

return (docid < other.docid) ||

(docid == other.docid &&

first_payload_offset < other.first_payload_offset);

}

};

...

std::sort(hits_.begin(), hits_.end());

复用临时对象

在循环内部声明的容器或对象会在每次循环迭代时重新创建。这可能导致代价高昂的构造、析构和调整大小。将声明提升到循环之外可以复用，并能带来显著的性能提升。（由于语言语义或无法确保程序等价性，编译器通常无法自行进行这种提升。）

将变量定义提升到循环迭代之外。

autofdo_profile_utils.h

auto iterator = absl::WrapUnique(sstable->GetIterator());

while (!iterator->done()) {

T profile;

if (!profile.ParseFromString(iterator->value_view())) {

return absl::InternalError(

"Failed to parse mem_block to specified profile type.");

}

...

iterator->Next();

}

auto iterator = absl::WrapUnique(sstable->GetIterator());

T profile;

while (!iterator->done()) {

if (!profile.ParseFromString(iterator->value_view())) {

return absl::InternalError(

"Failed to parse mem_block to specified profile type.");

}

...

iterator->Next();

}

在循环外定义 protobuf 变量，以便其已分配的存储可以在循环迭代之间复用。

stats-router.cc

for (auto& r : routers_to_update) {

...

ResourceRecord record;

{

MutexLock agg_lock(r.agg->mutex());

r.agg->AddResourceRecordUsages(measure_indices, &record);

}

...

}

ResourceRecord record;

for (auto& r : routers_to_update) {

...

record.Clear();

{

MutexLock agg_lock(r.agg->mutex());

r.agg->AddResourceRecordUsages(measure_indices, &record);

}

...

}

反复序列化到同一个 std::string。

program_rep.cc

std::string DeterministicSerialization(const proto2::Message& m) {

std::string result;

proto2::io::StringOutputStream sink(&result);

proto2::io::CodedOutputStream out(&sink);

out.SetSerializationDeterministic(true);

m.SerializePartialToCodedStream(&out);

return result;

}

absl::string_view DeterministicSerializationTo(const proto2::Message& m,

std::string* scratch) {

scratch->clear();

proto2::io::StringOutputStream sink(scratch);

proto2::io::CodedOutputStream out(&sink);

out.SetSerializationDeterministic(true);

m.SerializePartialToCodedStream(&out);

return absl::string_view(*scratch);

}

注意事项：protobuf、string、vector、容器等往往会增长到其中曾存储过的最大值大小。因此，定期重建它们（例如每使用 N 次后）有助于降低内存需求和重新初始化成本。

避免不必要的工作

提升性能最有效的类别之一，或许就是避免做不必做的工作。这可以有多种形式，包括为常见情况创建专门的代码路径，以避免更通用但昂贵的计算；预计算；将工作推迟到确实需要时再做；将工作提升到执行频率较低的代码片段中；以及其他类似方法。下面按几个有代表性的类别给出这种通用方法的许多示例。

常见情况的快速路径

代码通常会写成覆盖所有情况，但其中某些子集比其他情况简单得多，也常见得多。例如，vector::push_back 通常有足够的空间容纳新元素，但也包含在空间不足时调整底层存储大小的代码。适当关注代码结构，可以让常见的简单情况更快，同时不会显著损害非常见情况的性能。

让快速路径覆盖更多常见情况。

增加对末尾单个 ASCII 字节的处理，而不是只用此例程处理四字节倍数。这避免了对全 ASCII 字符串（例如 5 字节）调用较慢的通用例程。

utf8statetable.cc

// 基于状态表扫描 UTF-8 stringpiece。

// 始终扫描完整的 UTF-8 字符

// 设置已扫描的字节数。返回退出原因

// 针对 7-bit ASCII 0000..007f 全部有效的情况进行优化

int UTF8GenericScanFastAscii(const UTF8ScanObj* st, absl::string_view str,

int* bytes_consumed) {

...

int exit_reason;

do {

// 一次跳过 8 字节 ASCII；没有字节序问题

while ((src_limit - src >= 8) &&

(((UNALIGNED_LOAD32(src + 0) | UNALIGNED_LOAD32(src + 4)) &

0x80808080) == 0)) {

src += 8;

}

// 对剩余部分运行状态表

int rest_consumed;

exit_reason = UTF8GenericScan(

st, absl::ClippedSubstr(str, src - initial_src), &rest_consumed);

src += rest_consumed;

} while (exit_reason == kExitDoAgain);

*bytes_consumed = src - initial_src;

return exit_reason;

}

// 基于状态表扫描 UTF-8 stringpiece。

// 始终扫描完整的 UTF-8 字符

// 设置已扫描的字节数。返回退出原因

// 针对 7-bit ASCII 0000..007f 全部有效的情况进行优化

int UTF8GenericScanFastAscii(const UTF8ScanObj* st, absl::string_view str,

int* bytes_consumed) {

...

int exit_reason = kExitOK;

do {

// 一次跳过 8 字节 ASCII；没有字节序问题

while ((src_limit - src >= 8) &&

(((UNALIGNED_LOAD32(src + 0) | UNALIGNED_LOAD32(src + 4)) &

0x80808080) == 0)) {

src += 8;

}

while (src < src_limit && Is7BitAscii(*src)) { // 跳过 ASCII 字节

src++;

}

if (src < src_limit) {

// 对剩余部分运行状态表

int rest_consumed;

exit_reason = UTF8GenericScan(

st, absl::ClippedSubstr(str, src - initial_src), &rest_consumed);

src += rest_consumed;

}

} while (exit_reason == kExitDoAgain);

*bytes_consumed = src - initial_src;

return exit_reason;

}

用于 InlinedVector 的更简单快速路径。

inlined_vector.h

auto Storage<T, N, A>::Resize(ValueAdapter values, size_type new_size) -> void {

StorageView storage_view = MakeStorageView();

IteratorValueAdapter<MoveIterator> move_values(

MoveIterator(storage_view.data));

AllocationTransaction allocation_tx(GetAllocPtr());

ConstructionTransaction construction_tx(GetAllocPtr());

absl::Span<value_type> construct_loop;

absl::Span<value_type> move_construct_loop;

absl::Span<value_type> destroy_loop;

if (new_size > storage_view.capacity) {

...

} else if (new_size > storage_view.size) {

construct_loop = {storage_view.data + storage_view.size,

new_size - storage_view.size};

} else {

destroy_loop = {storage_view.data + new_size, storage_view.size - new_size};

}

auto Storage<T, N, A>::Resize(ValueAdapter values, size_type new_size) -> void {

StorageView storage_view = MakeStorageView();

auto* const base = storage_view.data;

const size_type size = storage_view.size;

auto* alloc = GetAllocPtr();

if (new_size <= size) {

// 销毁多余的旧元素。

inlined_vector_internal::DestroyElements(alloc, base + new_size,

size - new_size);

} else if (new_size <= storage_view.capacity) {

// 就地构造新元素。

inlined_vector_internal::ConstructElements(alloc, base + size, &values,

new_size - size);

} else {

...

}

初始化 1-D 到 4-D tensors 常见情况的快速路径。

tensor_shape.cc

template <class Shape>

TensorShapeBase<Shape>::TensorShapeBase(gtl::ArraySlice<int64> dim_sizes) {

set_tag(REP16);

set_data_type(DT_INVALID);

set_ndims_byte(0);

set_num_elements(1);

for (int64 s : dim_sizes) {

AddDim(internal::SubtleMustCopy(s));

}

}

template <class Shape>

void TensorShapeBase<Shape>::InitDims(gtl::ArraySlice<int64> dim_sizes) {

DCHECK_EQ(tag(), REP16);

// 允许小于 kint64max^0.25 的大小，以便下面的 4 路乘法

// 不会溢出。

static const uint64 kMaxSmall = 0xd744;

static_assert(kMaxSmall * kMaxSmall * kMaxSmall * kMaxSmall <= kint64max,

"bad overflow check");

bool large_size = false;

for (auto s : dim_sizes) {

if (s > kMaxSmall) {

large_size = true;

break;

}

}

if (!large_size) {

// 每个大小都适合 16 位；对 {1,2,3,4} 中的 dims 使用快速路径。

uint16* dst = as16()->dims_;

switch (dim_sizes.size()) {

case 1: {

set_ndims_byte(1);

const int64 size = dim_sizes[0];

const bool neg = Set16(kIsPartial, dst, 0, size);

set_num_elements(neg ? -1 : size);

return;

}

case 2: {

set_ndims_byte(2);

const int64 size0 = dim_sizes[0];

const int64 size1 = dim_sizes[1];

bool neg = Set16(kIsPartial, dst, 0, size0);

neg |= Set16(kIsPartial, dst, 1, size1);

set_num_elements(neg ? -1 : (size0 * size1));

return;

}

case 3: {

...

}

case 4: {

...

}

}

}

set_ndims_byte(0);

set_num_elements(1);

for (int64 s : dim_sizes) {

AddDim(internal::SubtleMustCopy(s));

}

}

让 varint 解析器快速路径只覆盖 1 字节情况，而不是同时覆盖 1 字节和 2 字节情况。

减小（内联）快速路径的大小会减少代码大小和 icache 压力，从而提升性能。

parse_context.h

template <typename T>

PROTOBUF_NODISCARD const char* VarintParse(const char* p, T* out) {

auto ptr = reinterpret_cast<const uint8_t*>(p);

uint32_t res = ptr[0];

if (!(res & 0x80)) {

*out = res;

return p + 1;

}

uint32_t byte = ptr[1];

res += (byte - 1) << 7;

if (!(byte & 0x80)) {

*out = res;

return p + 2;

}

return VarintParseSlow(p, res, out);

}

template <typename T>

PROTOBUF_NODISCARD const char* VarintParse(const char* p, T* out) {

auto ptr = reinterpret_cast<const uint8_t*>(p);

uint32_t res = ptr[0];

if (!(res & 0x80)) {

*out = res;

return p + 1;

}

return VarintParseSlow(p, res, out);

}

parse_context.cc

std::pair<const char*, uint32_t> VarintParseSlow32(const char* p,

uint32_t res) {

for (std::uint32_t i = 2; i < 5; i++) {

...

}

...

std::pair<const char*, uint64_t> VarintParseSlow64(const char* p,

uint32_t res32) {

uint64_t res = res32;

for (std::uint32_t i = 2; i < 10; i++) {

...

}

std::pair<const char*, uint32_t> VarintParseSlow32(const char* p,

uint32_t res) {

for (std::uint32_t i = 1; i < 5; i++) {

...

}

...

std::pair<const char*, uint64_t> VarintParseSlow64(const char* p,

uint32_t res32) {

uint64_t res = res32;

for (std::uint32_t i = 1; i < 10; i++) {

...

}

如果没有发生错误，则在 RPC_Stats_Measurement 加法中跳过大量工作。

rpc-stats.h

struct RPC_Stats_Measurement {

...

double errors[RPC::NUM_ERRORS];

struct RPC_Stats_Measurement {

...

double get_errors(int index) const { return errors[index]; }

void set_errors(int index, double value) {

errors[index] = value;

any_errors_set = true;

}

private:

...

// 将其设为 private，以便我们能够跟踪这些值中是否有任何一个

// 被设置为非零值。

double errors[RPC::NUM_ERRORS];

bool any_errors_set;  // 当且仅当任意 errors[i] 值非零时为 true

rpc-stats.cc

void RPC_Stats_Measurement::operator+=(const RPC_Stats_Measurement& x) {

...

for (int i = 0; i < RPC::NUM_ERRORS; ++i) {

errors[i] += x.errors[i];

}

}

void RPC_Stats_Measurement::operator+=(const RPC_Stats_Measurement& x) {

...

if (x.any_errors_set) {

for (int i = 0; i < RPC::NUM_ERRORS; ++i) {

errors[i] += x.errors[i];

}

any_errors_set = true;

}

}

对字符串首字节进行数组查找，通常可避免对完整字符串做指纹计算。

soft-tokens-helper.cc

bool SoftTokensHelper::IsSoftToken(const StringPiece& token) const {

return soft_tokens_.find(Fingerprint(token.data(), token.size())) !=

soft_tokens_.end();

}

soft-tokens-helper.h

class SoftTokensHelper {

...

private:

...

// 由于 soft tokens 大多与标点相关，出于性能

// 目的，我们维护一个数组 filter_。当且仅当有任意

// soft tokens 以字节值 'i' 开头时，filter_[i] 为 true。这避免了

// 在常见情况下对 term 做指纹计算，因为我们只需基于第一个字节

// 做一次数组查找；如果 filter_[b] 为 false，那么

// 可以立即返回 false。

bool          filter_[256];

...

};

inline bool SoftTokensHelper::IsSoftToken(const StringPiece& token) const {

if (token.size() >= 1) {

char first_char = token.data()[0];

if (!filter_[first_char]) {

return false;

}

}

return IsSoftTokenFallback(token);

}

soft-tokens-helper.cc

bool SoftTokensHelper::IsSoftTokenFallback(const StringPiece& token) const {

return soft_tokens_.find(Fingerprint(token.data(), token.size())) !=

soft_tokens_.end();

}

一次性预计算昂贵信息

预计算一个 TensorFlow 图执行节点属性，使我们能够快速排除某些不常见情况。

executor.cc

struct NodeItem {

...

bool kernel_is_expensive = false;  // 当且仅当 kernel->IsExpensive() 时为 true

bool kernel_is_async = false;      // 当且仅当 kernel->AsAsync() != nullptr 时为 true

bool is_merge = false;             // 当且仅当 IsMerge(node) 时为 true

...

if (IsEnter(node)) {

...

} else if (IsExit(node)) {

...

} else if (IsNextIteration(node)) {

...

} else {

// 大多数 nodes 的正常路径

...

}

struct NodeItem {

...

bool kernel_is_expensive : 1;  // 当且仅当 kernel->IsExpensive() 时为 true

bool kernel_is_async : 1;      // 当且仅当 kernel->AsAsync() != nullptr 时为 true

bool is_merge : 1;             // 当且仅当 IsMerge(node) 时为 true

bool is_enter : 1;             // 当且仅当 IsEnter(node) 时为 true

bool is_exit : 1;              // 当且仅当 IsExit(node) 时为 true

bool is_control_trigger : 1;   // 当且仅当 IsControlTrigger(node) 时为 true

bool is_sink : 1;              // 当且仅当 IsSink(node) 时为 true

// 当且仅当 IsEnter(node) || IsExit(node) || IsNextIteration(node) 时为 true

bool is_enter_exit_or_next_iter : 1;

...

if (!item->is_enter_exit_or_next_iter) {

// 对不需要特殊处理的 nodes 类型使用快速路径

DCHECK_EQ(input_frame, output_frame);

...

} else if (item->is_enter) {

...

} else if (item->is_exit) {

...

} else {

DCHECK(IsNextIteration(node));

...

}

预计算 256 元素数组，并在 trigram 初始化期间使用。

byte_trigram_classifier.cc

void ByteTrigramClassifier::VerifyModel(void) const {

ProbT class_sums[num_classes_];

for (int cls = 0; cls < num_classes_; cls++) {

class_sums[cls] = 0;

}

for (ByteNgramId id = 0; id < trigrams_.num_trigrams(); id++) {

for (int cls = 0; cls < num_classes_; ++cls) {

class_sums[cls] += Prob(trigram_probs_[id].log_probs[cls]);

}

}

...

}

void ByteTrigramClassifier::VerifyModel(void) const {

CHECK_EQ(sizeof(ByteLogProbT), 1);

ProbT fast_prob[256];

for (int b = 0; b < 256; b++) {

fast_prob[b] = Prob(static_cast<ByteLogProbT>(b));

}

ProbT class_sums[num_classes_];

for (int cls = 0; cls < num_classes_; cls++) {

class_sums[cls] = 0;

}

for (ByteNgramId id = 0; id < trigrams_.num_trigrams(); id++) {

for (int cls = 0; cls < num_classes_; ++cls) {

class_sums[cls] += fast_prob[trigram_probs_[id].log_probs[cls]];

}

}

...

}

一般建议：在模块边界检查格式错误的输入，而不是在内部反复检查。

将昂贵计算移出循环

将边界计算移出循环。

literal_linearizer.cc

for (int64 i = 0; i < src_shape.dimensions(dimension_numbers.front());

++i) {

int64 dim_front = src_shape.dimensions(dimension_numbers.front());

const uint8* src_buffer_data = src_buffer.data();

uint8* dst_buffer_data = dst_buffer.data();

for (int64 i = 0; i < dim_front; ++i) {

推迟昂贵计算

将 GetSubSharding 调用推迟到需要时再执行，这将 CPU 时间从 43 秒减少到 2 秒。

sharding_propagation.cc

HloSharding alternative_sub_sharding =

user.sharding().GetSubSharding(user.shape(), {i});

if (user.operand(i) == &instruction &&

hlo_sharding_util::IsShardingMoreSpecific(alternative_sub_sharding,

sub_sharding)) {

sub_sharding = alternative_sub_sharding;

}

if (user.operand(i) == &instruction) {

// 只有当此 operand 是关注对象时才评估 GetSubSharding，

// 因为它相对昂贵。

HloSharding alternative_sub_sharding =

user.sharding().GetSubSharding(user.shape(), {i});

if (hlo_sharding_util::IsShardingMoreSpecific(

alternative_sub_sharding, sub_sharding)) {

sub_sharding = alternative_sub_sharding;

}

}

不要急切更新 stats；按需计算。

不要在非常频繁的 allocation/deallocation 调用中更新 stats。相反，在调用频率低得多的 Stats() 方法被调用时按需计算 stats。

在 Google 的 web server 中，为 query 处理预分配 10 个 nodes，而不是 200 个。

这个简单变更将 web server 的 CPU 使用率降低了 7.5%。

querytree.h

static const int kInitParseTreeSize = 200;   // querynode pool 的初始大小

static const int kInitParseTreeSize = 10;   // querynode pool 的初始大小

更改搜索顺序以提升 19% 的吞吐量。

一个旧的搜索系统（约 2000 年）有两层：一层包含全文索引，另一层只包含 title 和 anchor terms 的索引。我们过去会先搜索较小的 title/anchor 层。与直觉相反，我们发现先搜索较大的全文索引层成本更低，因为如果到达全文层末尾，就可以完全跳过搜索 title/anchor 层（全文层的一个子集）。这种情况相当常见，使我们能够减少处理 query 所需的平均磁盘寻道次数。

背景信息请参阅 The Anatomy of a Large-Scale Hypertextual Web Search Engine 中关于 title 和 anchor text 处理的讨论。

专门化代码

某个特定的性能敏感调用点，可能不需要通用库提供的全部通用性。在这种情况下，如果能提升性能，请考虑编写专门化代码，而不是调用通用代码。

Histogram 类的自定义打印代码比 sprintf 快 4 倍。

这段代码对性能敏感，因为它会在监控系统从各种服务器收集统计信息时被调用。

histogram_export.cc

void Histogram::PopulateBuckets(const string &prefix,

expvar::MapProto *const var) const {

...

for (int i = min_bucket; i <= max_bucket; ++i) {

const double count = BucketCount(i);

if (!export_empty_buckets && count == 0.0) continue;

acc += count;

// 离散直方图导出的存储桶标签格式

// 指定了一个包含式上界，这与原始

// Histogram 实现中的格式相同。此格式

// 不适用于非离散直方图，因此对它们使用半开区间，

// 并使用 "_" 而不是 "-" 作为分隔符，

// 以便区分这些格式。

string key =

options_.export_cumulative_counts() ?

StringPrintf("%.12g", boundaries_->BucketLimit(i)) :

options_.discrete() ?

StringPrintf("%.0f-%.0f",

ceil(boundaries_->BucketStart(i)),

ceil(boundaries_->BucketLimit(i)) - 1.0) :

StringPrintf("%.12g_%.12g",

boundaries_->BucketStart(i),

boundaries_->BucketLimit(i));

EscapeMapKey(&key);

const double value = options_.export_cumulative_counts() ? acc : count;

expvar::AddMapFloat(StrCat(prefix,

options_.export_bucket_key_prefix(),

key),

value * count_mult,

var);

}

// 根据 format 格式化 "val"。如果 "need_escape" 为 true，则

// format 可能生成包含 '.' 的输出，并且结果将被转义。

// 如果 "need_escape" 为 false，则调用方保证 format

// 会使得到的数字不包含任何 '.' 字符，

// 因此我们可以避免调用 EscapeKey。

// 如有必要，该函数可以使用 "*scratch" 作为临时工作空间，

// 并且得到的 StringPiece 可能指向 "*scratch" 内部。

static StringPiece FormatNumber(const char* format,

bool need_escape,

double val, string* scratch) {

// 此例程专门用于仅处理有限数量的格式

DCHECK(StringPiece(format) == "%.0f" || StringPiece(format) == "%.12g");

scratch->clear();

if (val == trunc(val) && val >= kint32min && val <= kint32max) {

// 可以直接使用 StrAppend 的整数

StrAppend(scratch, static_cast<int32>(val));

return StringPiece(*scratch);

} else if (isinf(val)) {

// 无穷大，表示为 'inf'。

return StringPiece("inf", 3);

} else {

// 根据 "format" 格式化，并可能进行转义。

StringAppendF(scratch, format, val);

if (need_escape) {

EscapeMapKey(scratch);

} else {

DCHECK(!StringPiece(*scratch).contains("."));

}

return StringPiece(*scratch);

}

}

...

void Histogram::PopulateBuckets(const string &prefix,

expvar::MapProto *const var) const {

...

const string full_key_prefix = StrCat(prefix,

options_.export_bucket_key_prefix());

string key = full_key_prefix;  // 键将以 "full_key_prefix" 开头。

string start_scratch;

string limit_scratch;

const bool cumul_counts = options_.export_cumulative_counts();

const bool discrete = options_.discrete();

for (int i = min_bucket; i <= max_bucket; ++i) {

const double count = BucketCount(i);

if (!export_empty_buckets && count == 0.0) continue;

acc += count;

// 离散直方图导出的存储桶标签格式

// 指定了一个包含式上界，这与原始

// Histogram 实现中的格式相同。此格式

// 不适用于非离散直方图，因此对它们使用半开区间，

// 并使用 "_" 而不是 "-" 作为分隔符，

// 以便区分这些格式。

key.resize(full_key_prefix.size());  // 从 full_key_prefix 开始。

DCHECK_EQ(key, full_key_prefix);

const double limit = boundaries_->BucketLimit(i);

if (cumul_counts) {

StrAppend(&key, FormatNumber("%.12g", true, limit, &limit_scratch));

} else {

const double start = boundaries_->BucketStart(i);

if (discrete) {

StrAppend(&key,

FormatNumber("%.0f", false, ceil(start), &start_scratch),

"-",

FormatNumber("%.0f", false, ceil(limit) - 1.0,

&limit_scratch));

} else {

StrAppend(&key,

FormatNumber("%.12g", true, start, &start_scratch),

"_",

FormatNumber("%.12g", true, limit, &limit_scratch));

}

}

const double value = cumul_counts ? acc : count;

// 添加到 map var

expvar::AddMapFloat(key, value * count_mult, var);

}

}

为 VLOG(1)、VLOG(2)、… 添加特化，以提升速度并减小代码大小。

VLOG 是整个代码库中大量使用的宏。此更改避免在几乎每个调用点传递额外的整数常量（如果日志级别在调用点是常量，几乎总是如此，例如 VLOG(1) << ...），从而节省代码空间。

vlog_is_on.h

class VLogSite final {

public:

...

bool IsEnabled(int level) {

int stale_v = v_.load(std::memory_order_relaxed);

if (ABSL_PREDICT_TRUE(level > stale_v)) {

return false;

}

// 我们将快速路径以外的所有内容，即 vlogging 已初始化

// 但未开启的情况，放到一个非内联函数之后，以减小代码大小。

return SlowIsEnabled(stale_v, level);

}

...

private:

...

ABSL_ATTRIBUTE_NOINLINE

bool SlowIsEnabled(int stale_v, int level);

...

};

class VLogSite final {

public:

...

bool IsEnabled(int level) {

int stale_v = v_.load(std::memory_order_relaxed);

if (ABSL_PREDICT_TRUE(level > stale_v)) {

return false;

}

// 我们将快速路径以外的所有内容，即 vlogging 已初始化

// 但未开启的情况，放到一个非内联函数之后，以减小代码大小。

// "level" 几乎总是调用点常量，因此我们可以通过

// 对级别 1、2 和 3 进行特殊处理来节省一些代码空间。

#if defined(__has_builtin) && __has_builtin(__builtin_constant_p)

if (__builtin_constant_p(level)) {

if (level == 0) return SlowIsEnabled0(stale_v);

if (level == 1) return SlowIsEnabled1(stale_v);

if (level == 2) return SlowIsEnabled2(stale_v);

if (level == 3) return SlowIsEnabled3(stale_v);

if (level == 4) return SlowIsEnabled4(stale_v);

if (level == 5) return SlowIsEnabled5(stale_v);

}

#endif

return SlowIsEnabled(stale_v, level);

...

private:

...

ABSL_ATTRIBUTE_NOINLINE

bool SlowIsEnabled(int stale_v, int level);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled0(int stale_v);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled1(int stale_v);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled2(int stale_v);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled3(int stale_v);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled4(int stale_v);

ABSL_ATTRIBUTE_NOINLINE bool SlowIsEnabled5(int stale_v);

...

};

vlog_is_on.cc

bool VLogSite::SlowIsEnabled0(int stale_v) { return SlowIsEnabled(stale_v, 0); }

bool VLogSite::SlowIsEnabled1(int stale_v) { return SlowIsEnabled(stale_v, 1); }

bool VLogSite::SlowIsEnabled2(int stale_v) { return SlowIsEnabled(stale_v, 2); }

bool VLogSite::SlowIsEnabled3(int stale_v) { return SlowIsEnabled(stale_v, 3); }

bool VLogSite::SlowIsEnabled4(int stale_v) { return SlowIsEnabled(stale_v, 4); }

bool VLogSite::SlowIsEnabled5(int stale_v) { return SlowIsEnabled(stale_v, 5); }

在可能的情况下，用简单的前缀匹配替换 RE2 调用。

read_matcher.cc

enum MatchItemType {

MATCH_TYPE_INVALID,

MATCH_TYPE_RANGE,

MATCH_TYPE_EXACT,

MATCH_TYPE_REGEXP,

};

enum MatchItemType {

MATCH_TYPE_INVALID,

MATCH_TYPE_RANGE,

MATCH_TYPE_EXACT,

MATCH_TYPE_REGEXP,

MATCH_TYPE_PREFIX,   // regexp ".*" 的特殊类型

};

read_matcher.cc

p->type = MATCH_TYPE_REGEXP;

term.NonMetaPrefix().CopyToString(&p->prefix);

if (term.RegexpSuffix() == ".*") {

// 匹配任何内容的 regexp 的特殊情况，因此我们可以

// 绕过 RE2::FullMatch

p->type = MATCH_TYPE_PREFIX;

} else {

p->type = MATCH_TYPE_REGEXP;

使用 StrCat 而不是 StringPrintf 来格式化 IP 地址。

ipaddress.cc

string IPAddress::ToString() const {

char buf[INET6_ADDRSTRLEN];

switch (address_family_) {

case AF_INET:

CHECK(inet_ntop(AF_INET, &addr_.addr4, buf, INET6_ADDRSTRLEN) != NULL);

return buf;

case AF_INET6:

CHECK(inet_ntop(AF_INET6, &addr_.addr6, buf, INET6_ADDRSTRLEN) != NULL);

return buf;

case AF_UNSPEC:

LOG(DFATAL) << "Calling ToString() on an empty IPAddress";

return "";

default:

LOG(FATAL) << "Unknown address family " << address_family_;

}

}

...

string IPAddressToURIString(const IPAddress& ip) {

switch (ip.address_family()) {

case AF_INET6:

return StringPrintf("[%s]", ip.ToString().c_str());

default:

return ip.ToString();

}

}

...

string SocketAddress::ToString() const {

return IPAddressToURIString(host_) + StringPrintf(":%u", port_);

}

string IPAddress::ToString() const {

char buf[INET6_ADDRSTRLEN];

switch (address_family_) {

case AF_INET: {

uint32 addr = gntohl(addr_.addr4.s_addr);

int a1 = static_cast<int>((addr >> 24) & 0xff);

int a2 = static_cast<int>((addr >> 16) & 0xff);

int a3 = static_cast<int>((addr >> 8) & 0xff);

int a4 = static_cast<int>(addr & 0xff);

return StrCat(a1, ".", a2, ".", a3, ".", a4);

}

case AF_INET6:

CHECK(inet_ntop(AF_INET6, &addr_.addr6, buf, INET6_ADDRSTRLEN) != NULL);

return buf;

case AF_UNSPEC:

LOG(DFATAL) << "Calling ToString() on an empty IPAddress";

return "";

default:

LOG(FATAL) << "Unknown address family " << address_family_;

}

}

...

string IPAddressToURIString(const IPAddress& ip) {

switch (ip.address_family()) {

case AF_INET6:

return StrCat("[", ip.ToString(), "]");

default:

return ip.ToString();

}

}

...

string SocketAddress::ToString() const {

return StrCat(IPAddressToURIString(host_), ":", port_);

}

使用缓存避免重复工作

基于大型序列化 proto 的预计算指纹进行缓存。

dp_ops.cc

InputOutputMappingProto mapping_proto;

PLAQUE_OP_REQUIRES(

mapping_proto.ParseFromStringPiece(GetAttrMappingProto(state)),

absl::InternalError("Failed to parse InputOutputMappingProto"));

ParseMapping(mapping_proto);

uint64 mapping_proto_fp = GetAttrMappingProtoFp(state);

{

absl::MutexLock l(&fp_to_iometa_mu);

if (fp_to_iometa == nullptr) {

fp_to_iometa =

new absl::flat_hash_map<uint64, std::unique_ptr<ProgramIOMetadata>>;

}

auto it = fp_to_iometa->find(mapping_proto_fp);

if (it != fp_to_iometa->end()) {

io_metadata_ = it->second.get();

} else {

auto serial_proto = GetAttrMappingProto(state);

DCHECK_EQ(mapping_proto_fp, Fingerprint(serial_proto));

InputOutputMappingProto mapping_proto;

PLAQUE_OP_REQUIRES(

mapping_proto.ParseFromStringPiece(GetAttrMappingProto(state)),

absl::InternalError("Failed to parse InputOutputMappingProto"));

auto io_meta = ParseMapping(mapping_proto);

io_metadata_ = io_meta.get();

(*fp_to_iometa)[mapping_proto_fp] = std::move(io_meta);

}

}

让编译器的工作更轻松

编译器可能难以跨越多层抽象进行优化，因为它必须对代码的整体行为做出保守假设，或者可能无法在速度与大小之间做出正确权衡。应用程序员通常会更了解系统行为，并且可以通过将代码重写为在更低层级上运行来帮助编译器。不过，只有在剖析结果显示存在问题时才应这样做，因为编译器通常会自行正确处理。查看性能关键例程生成的汇编代码，可以帮助你理解编译器是否“做对了”。Pprof 提供了非常有用的源代码与反汇编交错显示，并标注性能数据。

一些可能有用的技术：

- 避免在热函数中调用函数（使编译器能够避免栈帧设置成本）。

- 将慢路径代码移动到单独的尾调用函数中。

- 在大量使用之前，将少量数据复制到局部变量中。这可以让编译器假设它不会与其他数据发生别名，从而可能改进自动向量化和寄存器分配。

- 手动展开非常热的循环。

通过将 absl::Span 替换为指向底层数组的原始指针，加速 ShapeUtil::ForEachState。

shape_util.h

struct ForEachState {

ForEachState(const Shape& s, absl::Span<const int64_t> b,

absl::Span<const int64_t> c, absl::Span<const int64_t> i);

~ForEachState();

const Shape& shape;

const absl::Span<const int64_t> base;

const absl::Span<const int64_t> count;

const absl::Span<const int64_t> incr;

struct ForEachState {

ForEachState(const Shape& s, absl::Span<const int64_t> b,

absl::Span<const int64_t> c, absl::Span<const int64_t> i);

inline ~ForEachState() = default;

const Shape& shape;

// 指向传入 spans 的数组的指针

const int64_t* const base;

const int64_t* const count;

const int64_t* const incr;

手动展开循环冗余校验（CRC）计算循环。

crc.cc

void CRC32::Extend(uint64 *lo, uint64 *hi, const void *bytes, size_t length)

const {

...

// 一次处理 4 个字节

while ((p + 4) <= e) {

uint32 c = l ^ WORD(p);

p += 4;

l = this->table3_[c & 0xff] ^

this->table2_[(c >> 8) & 0xff] ^

this->table1_[(c >> 16) & 0xff] ^

this->table0_[c >> 24];

}

// 处理最后几个字节

while (p != e) {

int c = (l & 0xff) ^ *p++;

l = this->table0_[c] ^ (l >> 8);

}

*lo = l;

}

void CRC32::Extend(uint64 *lo, uint64 *hi, const void *bytes, size_t length)

const {

...

#define STEP {                                  \

uint32 c = l ^ WORD(p);                     \

p += 4;                                     \

l = this->table3_[c & 0xff] ^               \

this->table2_[(c >> 8) & 0xff] ^        \

this->table1_[(c >> 16) & 0xff] ^       \

this->table0_[c >> 24];                 \

}

// 一次处理 16 个字节

while ((e-p) >= 16) {

STEP;

STEP;

STEP;

STEP;

}

// 一次处理 4 个字节

while ((p + 4) <= e) {

STEP;

}

#undef STEP

// 处理最后几个字节

while (p != e) {

int c = (l & 0xff) ^ *p++;

l = this->table0_[c] ^ (l >> 8);

}

*lo = l;

}

解析 Spanner 键时一次处理四个字符。

- 手动展开循环，以便一次处理四个字符，而不是使用 memchr

- 手动展开循环，用于查找名称中分隔的部分

- 向后查找带有 '#' 分隔符的名称分隔部分（而不是向前），因为名称中的第一部分很可能最长。

key.cc

void Key::InitSeps(const char* start) {

const char* base = &rep_[0];

const char* limit = base + rep_.size();

const char* s = start;

DCHECK_GE(s, base);

DCHECK_LT(s, limit);

for (int i = 0; i < 3; i++) {

s = (const char*)memchr(s, '#', limit - s);

DCHECK(s != NULL);

seps_[i] = s - base;

s++;

}

}

inline const char* ScanBackwardsForSep(const char* base, const char* p) {

while (p >= base + 4) {

if (p[0] == '#') return p;

if (p[-1] == '#') return p-1;

if (p[-2] == '#') return p-2;

if (p[-3] == '#') return p-3;

p -= 4;

}

while (p >= base && *p != '#') p--;

return p;

}

void Key::InitSeps(const char* start) {

const char* base = &rep_[0];

const char* limit = base + rep_.size();

const char* s = start;

DCHECK_GE(s, base);

DCHECK_LT(s, limit);

// 我们从字符串末尾向后遍历，而不是向前，

// 因为目录名可能很长，并且肯定不包含

// 任何 '#' 字符。

const char* p = ScanBackwardsForSep(s, limit - 1);

DCHECK(*p == '#');

seps_[2] = p - base;

p--;

p = ScanBackwardsForSep(s, p);

DCHECK(*p == '#');

seps_[1] = p - base;

p--;

p = ScanBackwardsForSep(s, p);

DCHECK(*p == '#');

seps_[0] = p - base;

}

通过将 ABSL_LOG(FATAL) 转换为 ABSL_DCHECK(false) 来避免栈帧设置成本。

arena_cleanup.h

inline ABSL_ATTRIBUTE_ALWAYS_INLINE size_t Size(Tag tag) {

if (!EnableSpecializedTags()) return sizeof(DynamicNode);

switch (tag) {

case Tag::kDynamic:

return sizeof(DynamicNode);

case Tag::kString:

return sizeof(TaggedNode);

case Tag::kCord:

return sizeof(TaggedNode);

default:

ABSL_LOG(FATAL) << "Corrupted cleanup tag: " << static_cast<int>(tag);

return sizeof(DynamicNode);

}

}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE size_t Size(Tag tag) {

if (!EnableSpecializedTags()) return sizeof(DynamicNode);

switch (tag) {

case Tag::kDynamic:

return sizeof(DynamicNode);

case Tag::kString:

return sizeof(TaggedNode);

case Tag::kCord:

return sizeof(TaggedNode);

default:

ABSL_DCHECK(false) << "Corrupted cleanup tag: " << static_cast<int>(tag);

return sizeof(DynamicNode);

}

}

降低统计信息收集成本

需要在统计信息以及关于系统的其他行为信息的实用性，与维护这些信息的成本之间取得平衡。额外信息通常可以帮助人们理解并改进高层行为，但维护它们也可能代价高昂。

没有用的统计信息可以完全删除。

停止维护 SelectServer 中关于 alarm 和 closure 数量的高成本统计信息。

这是一组变更的一部分，将设置 alarm 的时间从 771 ns 降低到 271 ns。

selectserver.h

class SelectServer {

public:

...

protected:

...

scoped_ptr<MinuteTenMinuteHourStat> num_alarms_stat_;

...

scoped_ptr<MinuteTenMinuteHourStat> num_closures_stat_;

...

};

// Selectserver 类

class SelectServer {

...

protected:

...

};

/selectserver.cc

void SelectServer::AddAlarmInternal(Alarmer* alarmer,

int offset_in_ms,

int id,

bool is_periodic) {

...

alarms_->insert(alarm);

num_alarms_stat_->IncBy(1);

...

}

void SelectServer::AddAlarmInternal(Alarmer* alarmer,

int offset_in_ms,

int id,

bool is_periodic) {

...

alarms_->Add(alarm);

...

}

/selectserver.cc

void SelectServer::RemoveAlarm(Alarmer* alarmer, int id) {

...

alarms_->erase(alarm);

num_alarms_stat_->IncBy(-1);

...

}

void SelectServer::RemoveAlarm(Alarmer* alarmer, int id) {

...

alarms_->Remove(alarm);

...

}

通常，统计信息或其他属性可以只针对系统处理的元素样本来维护（例如 RPC 请求、输入记录、用户）。许多子系统都采用这种方法（tcmalloc 分配跟踪、/requestz 状态页、Dapper 样本）。

采样时，在适当情况下考虑降低采样率。

只为 doc info 请求的一个样本维护统计信息。

采样让我们能够避免在大多数请求上触碰 39 个直方图和 MinuteTenMinuteHour 统计信息。

generic-leaf-stats.cc

... code that touches 39 histograms to update various stats on every request ...

// 定期添加到直方图

if (TryLockToUpdateHistogramsDocInfo(docinfo_stats, bucket)) {

// 仅当应对此请求采样以维护统计信息时，才返回 true

// 并获取 bucket->lock

... code that touches 39 histograms to update various stats ...

bucket->lock.Unlock();

}

降低采样率并加快采样决策。

此变更将采样率从 1/10 降低到 1/32。此外，我们现在只为被采样的事件保留执行时间统计信息，并通过使用 2 的幂取模来加快采样决策。Google Meet 视频会议系统中的每个 packet 都会调用这段代码；在 COVID 暴发初期，随着用户快速转向更多在线会议，需要进行性能工作以满足容量需求。

packet_executor.cc

class ScopedPerformanceMeasurement {

public:

explicit ScopedPerformanceMeasurement(PacketExecutor* packet_executor)

: packet_executor_(packet_executor),

tracer_(packet_executor->packet_executor_trace_threshold_,

kClosureTraceName) {

// ThreadCPUUsage 是一次高成本调用。撰写本文时，

// 它耗时超过 400ns，约比 absl::Now 慢 30 倍，

// 因此我们只对 10% 的 closure 采样，以降低成本。

if (packet_executor->closures_executed_ % 10 == 0) {

thread_cpu_usage_start_ = base::ThreadCPUUsage();

}

// 在可能执行上述高成本调用之后再采样开始时间，

// 以免污染 wall time 测量。

run_start_time_ = absl::Now();

}

~ScopedPerformanceMeasurement() {

ScopedPerformanceMeasurement::ScopedPerformanceMeasurement(

PacketExecutor* packet_executor)

: packet_executor_(packet_executor),

tracer_(packet_executor->packet_executor_trace_threshold_,

kClosureTraceName) {

// ThreadCPUUsage 是一次高成本调用。撰写本文时，

// 它耗时超过 400ns，约比 absl::Now 慢 30 倍，

// 因此我们只对每 32 个 closure 中的 1 个采样，以降低成本。

if (packet_executor->closures_executed_ % 32 == 0) {

thread_cpu_usage_start_ = base::ThreadCPUUsage();

}

// 在可能执行上述高成本调用之后再采样开始时间，

// 以免污染 wall time 测量。

run_start_time_ = absl::Now();

}

packet_executor.cc

~ScopedPerformanceMeasurement() {

auto run_end_time = absl::Now();

auto run_duration = run_end_time - run_start_time_;

if (thread_cpu_usage_start_.has_value()) {

...

}

closure_execution_time->Record(absl::ToInt64Microseconds(run_duration));

ScopedPerformanceMeasurement::~ScopedPerformanceMeasurement() {

auto run_end_time = absl::Now();

auto run_duration = run_end_time - run_start_time_;

if (thread_cpu_usage_start_.has_value()) {

...

closure_execution_time->Record(absl::ToInt64Microseconds(run_duration));

}

基准测试结果:

Run on (40 X 2793 MHz CPUs); 2020-03-24T20:08:19.991412535-07:00

CPU: Intel Ivybridge with HyperThreading (20 cores) dL1:32KB dL2:256KB dL3:25MB

Benchmark                                      Base (ns)    New (ns) Improvement

----------------------------------------------------------------------------

BM_PacketOverhead_mean                               224          85    +62.0%

避免在热代码路径上进行日志记录

日志记录语句可能代价高昂，即使该语句的日志级别实际上不会记录任何内容。例如，ABSL_VLOG 的实现至少需要一次加载和一次比较，这在热代码路径中可能会成为问题。此外，日志代码的存在可能会阻碍编译器优化。考虑从热代码路径中完全删除日志记录。

从内存分配器的核心部分移除日志记录。

这只是更大变更中的一小部分。

gpu_bfc_allocator.cc

void GPUBFCAllocator::SplitChunk(...) {

...

VLOG(6) << "Adding to chunk map: " << new_chunk->ptr;

...

}

...

void GPUBFCAllocator::DeallocateRawInternal(void* ptr) {

...

VLOG(6) << "Chunk at " << c->ptr << " no longer in use";

...

}

void GPUBFCAllocator::SplitChunk(...) {

...

}

...

void GPUBFCAllocator::DeallocateRawInternal(void* ptr) {

...

}

在嵌套循环之外预先计算是否启用了日志记录。

image_similarity.cc

for (int j = 0; j < output_subimage_size_y; j++) {

int j1 = j - rad + output_to_integral_subimage_y;

int j2 = j1 + 2 * rad + 1;

// 为此行的输出创建一个指针，同时考虑到相对于完整图像的偏移

// to the full image.

double *image_diff_ptr = &(*image_diff)(j + min_j, min_i);

for (int i = 0; i < output_subimage_size_x; i++) {

...

if (VLOG_IS_ON(3)) {

...

}

...

}

}

const bool vlog_3 = DEBUG_MODE ? VLOG_IS_ON(3) : false;

for (int j = 0; j < output_subimage_size_y; j++) {

int j1 = j - rad + output_to_integral_subimage_y;

int j2 = j1 + 2 * rad + 1;

// 为此行的输出创建一个指针，同时考虑到相对于完整图像的偏移

// to the full image.

double *image_diff_ptr = &(*image_diff)(j + min_j, min_i);

for (int i = 0; i < output_subimage_size_x; i++) {

...

if (vlog_3) {

...

}

}

}

Run on (40 X 2801 MHz CPUs); 2016-05-16T15:55:32.250633072-07:00

CPU: Intel Ivybridge with HyperThreading (20 cores) dL1:32KB dL2:256KB dL3:25MB

Benchmark                          Base (ns)  New (ns) Improvement

------------------------------------------------------------------

BM_NCCPerformance/16                   29104     26372     +9.4%

BM_NCCPerformance/64                  473235    425281    +10.1%

BM_NCCPerformance/512               30246238  27622009     +8.7%

BM_NCCPerformance/1k              125651445  113361991     +9.8%

BM_NCCLimitedBoundsPerformance/16       8314      7498     +9.8%

BM_NCCLimitedBoundsPerformance/64     143508    132202     +7.9%

BM_NCCLimitedBoundsPerformance/512   9335684   8477567     +9.2%

BM_NCCLimitedBoundsPerformance/1k   37223897  34201739     +8.1%

预先计算是否启用了日志记录，并在 helper routine 中使用结果。

periodic_call.cc

VLOG(1) << Logid()

<< "MaybeScheduleAlarmAtNextTick. Time until next real time: "

<< time_until_next_real_time;

...

uint64 next_virtual_time_ms =

next_virtual_time_ms_ - num_ticks * kResolutionMs;

CHECK_GE(next_virtual_time_ms, 0);

ScheduleAlarm(now, delay, next_virtual_time_ms);

}

void ScheduleNextAlarm(uint64 current_virtual_time_ms)

ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {

if (calls_.empty()) {

VLOG(1) << Logid() << "No calls left, entering idle mode";

next_real_time_ = absl::InfiniteFuture();

return;

}

uint64 next_virtual_time_ms = FindNextVirtualTime(current_virtual_time_ms);

auto delay =

absl::Milliseconds(next_virtual_time_ms - current_virtual_time_ms);

ScheduleAlarm(GetClock().TimeNow(), delay, next_virtual_time_ms);

}

// 此函数调度的 alarm 会取代所有先前调度的

// alarm。这通过 `scheduling_sequence_number_` 来确保。

void ScheduleAlarm(absl::Time now, absl::Duration delay,

uint64 virtual_time_ms)

ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {

next_real_time_ = now + delay;

next_virtual_time_ms_ = virtual_time_ms;

++ref_count_;  // Alarm 持有一个引用。

++scheduling_sequence_number_;

VLOG(1) << Logid() << "ScheduleAlarm. Time : "

<< absl::FormatTime("%M:%S.%E3f", now, absl::UTCTimeZone())

<< ", delay: " << delay << ", virtual time: " << virtual_time_ms

<< ", refs: " << ref_count_

<< ", seq: " << scheduling_sequence_number_

<< ", executor: " << executor_;

executor_->AddAfter(

delay, new Alarm(this, virtual_time_ms, scheduling_sequence_number_));

}

const bool vlog_1 = VLOG_IS_ON(1);

if (vlog_1) {

VLOG(1) << Logid()

<< "MaybeScheduleAlarmAtNextTick. Time until next real time: "

<< time_until_next_real_time;

}

...

uint64 next_virtual_time_ms =

next_virtual_time_ms_ - num_ticks * kResolutionMs;

CHECK_GE(next_virtual_time_ms, 0);

ScheduleAlarm(now, delay, next_virtual_time_ms, vlog_1);

}

void ScheduleNextAlarm(uint64 current_virtual_time_ms, bool vlog_1)

ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {

if (calls_.empty()) {

if (vlog_1) {

VLOG(1) << Logid() << "No calls left, entering idle mode";

}

next_real_time_ = absl::InfiniteFuture();

return;

}

uint64 next_virtual_time_ms = FindNextVirtualTime(current_virtual_time_ms);

auto delay =

absl::Milliseconds(next_virtual_time_ms - current_virtual_time_ms);

ScheduleAlarm(GetClock().TimeNow(), delay, next_virtual_time_ms, vlog_1);

}

// 此函数调度的 alarm 会取代所有先前调度的

// alarm。这通过 `scheduling_sequence_number_` 来确保。

void ScheduleAlarm(absl::Time now, absl::Duration delay,

uint64 virtual_time_ms,

bool vlog_1)

ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {

next_real_time_ = now + delay;

next_virtual_time_ms_ = virtual_time_ms;

++ref_count_;  // Alarm 持有一个引用。

++scheduling_sequence_number_;

if (vlog_1) {

VLOG(1) << Logid() << "ScheduleAlarm. Time : "

<< absl::FormatTime("%M:%S.%E3f", now, absl::UTCTimeZone())

<< ", delay: " << delay << ", virtual time: " << virtual_time_ms

<< ", refs: " << ref_count_

<< ", seq: " << scheduling_sequence_number_

<< ", executor: " << executor_;

}

executor_->AddAfter(

delay, new Alarm(this, virtual_time_ms, scheduling_sequence_number_));

}

代码大小方面的考量

性能不仅仅包括运行时速度。有时值得考虑软件选择对生成代码大小的影响。代码体积大意味着编译和链接时间更长、二进制文件臃肿、内存使用更多、icache 压力更大，并且有时会对分支预测器等微架构结构产生其他负面影响。在编写会被许多地方使用的低层库代码时，或在编写你预期会针对许多不同类型实例化的模板代码时，思考这些问题尤其重要。

用于减少代码大小的技术在不同编程语言之间差异很大。以下是我们发现对 C++ 代码有用的一些技术（C++ 代码可能会受到模板和内联过度使用的影响）。

裁剪常见的内联代码

被广泛调用的函数与内联相结合，会对代码大小产生显著影响。

加速 TF_CHECK_OK。

避免创建 Ok object，并通过将致命错误消息的复杂格式化放到 out of line 代码中、而不是每个调用点都执行，来节省代码空间。

status.h

#define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))

#define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))

extern tensorflow::string* TfCheckOpHelperOutOfLine(

const ::tensorflow::Status& v, const char* msg);

inline tensorflow::string* TfCheckOpHelper(::tensorflow::Status v,

const char* msg) {

if (v.ok()) return nullptr;

return TfCheckOpHelperOutOfLine(v, msg);

}

#define TF_CHECK_OK(val)                                           \

while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \

LOG(FATAL) << *(_result)

#define TF_QCHECK_OK(val)                                          \

while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \

LOG(QFATAL) << *(_result)

status.cc

string* TfCheckOpHelperOutOfLine(const ::tensorflow::Status& v,

const char* msg) {

string r("Non-OK-status: ");

r += msg;

r += " status: ";

r += v.ToString();

// 会泄漏 string，但这只用于致命错误消息

return new string(r);

}

将每个 RETURN_IF_ERROR 调用点缩小 79 字节代码。

- 添加了专门只供 RETURN_IF_ERROR 使用的特殊 adaptor class。

- 不要在 RETURN_IF_ERROR 的 fast path 上构造/析构 StatusBuilder。

- 不要内联某些 StatusBuilder 方法，因为现在 fast path 上不再需要它们。

- 避免不必要的 ~Status 调用。

将 CHECK_GE 的性能提高 4.5 倍，并将代码大小从 125 字节缩小到 77 字节。

logging.h

struct CheckOpString {

CheckOpString(string* str) : str_(str) { }

~CheckOpString() { delete str_; }

operator bool() const { return str_ == NULL; }

string* str_;

};

...

#define DEFINE_CHECK_OP_IMPL(name, op) \

template <class t1, class t2> \

inline string* Check##name##Impl(const t1& v1, const t2& v2, \

const char* names) { \

if (v1 op v2) return NULL; \

else return MakeCheckOpString(v1, v2, names); \

} \

string* Check##name##Impl(int v1, int v2, const char* names);

DEFINE_CHECK_OP_IMPL(EQ, ==)

DEFINE_CHECK_OP_IMPL(NE, !=)

DEFINE_CHECK_OP_IMPL(LE, <=)

DEFINE_CHECK_OP_IMPL(LT, < )

DEFINE_CHECK_OP_IMPL(GE, >=)

DEFINE_CHECK_OP_IMPL(GT, > )

#undef DEFINE_CHECK_OP_IMPL

struct CheckOpString {

CheckOpString(string* str) : str_(str) { }

// 没有析构函数：如果 str_ 非 NULL，我们即将 LOG(FATAL)，

// 因此没有必要清理 str_。

operator bool() const { return str_ == NULL; }

string* str_;

};

...

extern string* MakeCheckOpStringIntInt(int v1, int v2, const char* names);

template<int, int>

string* MakeCheckOpString(const int& v1, const int& v2, const char* names) {

return MakeCheckOpStringIntInt(v1, v2, names);

}

...

#define DEFINE_CHECK_OP_IMPL(name, op) \

template <class t1, class t2> \

inline string* Check##name##Impl(const t1& v1, const t2& v2, \

const char* names) { \

if (v1 op v2) return NULL; \

else return MakeCheckOpString(v1, v2, names); \

} \

inline string* Check##name##Impl(int v1, int v2, const char* names) { \

if (v1 op v2) return NULL; \

else return MakeCheckOpString(v1, v2, names); \

}

DEFINE_CHECK_OP_IMPL(EQ, ==)

DEFINE_CHECK_OP_IMPL(NE, !=)

DEFINE_CHECK_OP_IMPL(LE, <=)

DEFINE_CHECK_OP_IMPL(LT, < )

DEFINE_CHECK_OP_IMPL(GE, >=)

DEFINE_CHECK_OP_IMPL(GT, > )

#undef DEFINE_CHECK_OP_IMPL

logging.cc

string* MakeCheckOpStringIntInt(int v1, int v2, const char* names) {

strstream ss;

ss << names << " (" << v1 << " vs. " << v2 << ")";

return new string(ss.str(), ss.pcount());

}

谨慎使用内联

内联通常可以提升性能，但有时它会增加代码大小，却没有相应的性能收益（某些情况下甚至会因指令缓存压力增加而造成性能损失）。

减少 TensorFlow 中的内联。

该变更停止内联许多对性能不敏感的函数（例如错误路径和 op 注册代码）。此外，一些对性能敏感函数的慢路径被移入非内联函数中。

这些变更将典型二进制文件中 tensorflow 符号的大小减少了 12.2%（从 8814545 字节降至 7740233 字节）。

Protocol buffer 库变更。避免为 ≥ 128 字节消息的消息长度编码生成昂贵的内联代码空间，而是改为过程调用一个共享的非内联例程。

不仅使重要的大型二进制文件更小，也使其更快。

某个大型二进制文件中一个高度内联例程每行生成代码的字节数。第一个数字表示某个特定源代码行生成的总字节数，包括该代码被内联的所有位置。

变更前：

.           0   1825 template <typename MessageType>

.           0   1826 inline uint8* WireFormatLite::InternalWriteMessage(

.           0   1827     int field_number, const MessageType& value, uint8* target,

.           0   1828     io::EpsCopyOutputStream* stream) {

>>>    389246   1829   target = WriteTagToArray(field_number, WIRETYPE_LENGTH_DELIMITED, target);

>>>   5454640   1830   target = io::CodedOutputStream::WriteVarint32ToArray(

>>>    337837   1831       static_cast<uint32>(value.GetCachedSize()), target);

>>>   1285539   1832   return value._InternalSerialize(target, stream);

.           0   1833 }

应用此变更后的新代码大小输出如下：

.           0   1825 template <typename MessageType>

.           0   1826 inline uint8* WireFormatLite::InternalWriteMessage(

.           0   1827     int field_number, const MessageType& value, uint8* target,

.           0   1828     io::EpsCopyOutputStream* stream) {

>>>    450612   1829   target = WriteTagToArray(field_number, WIRETYPE_LENGTH_DELIMITED, target);

>>       9609   1830   target = io::CodedOutputStream::WriteVarint32ToArrayOutOfLine(

>>>    434668   1831       static_cast<uint32>(value.GetCachedSize()), target);

>>>   1597394   1832   return value._InternalSerialize(target, stream);

.           0   1833 }

coded_stream.h

class PROTOBUF_EXPORT CodedOutputStream {

...

// 类似 WriteVarint32()，但直接写入目标数组，并且将

// 较少见场景的路径放在非内联代码中，而不是内联。

static uint8* WriteVarint32ToArrayOutOfLine(uint32 value, uint8* target);

...

};

...

inline uint8* CodedOutputStream::WriteVarint32ToArrayOutOfLine(uint32 value,

uint8* target) {

target[0] = static_cast<uint8>(value);

if (value < 0x80) {

return target + 1;

} else {

return WriteVarint32ToArrayOutOfLineHelper(value, target);

}

}

coded_stream.cc

uint8* CodedOutputStream::WriteVarint32ToArrayOutOfLineHelper(uint32 value,

uint8* target) {

DCHECK_GE(value, 0x80);

target[0] |= static_cast<uint8>(0x80);

value >>= 7;

target[1] = static_cast<uint8>(value);

if (value < 0x80) {

return target + 2;

}

target += 2;

do {

// 打开刚刚写入字节中的 continuation bit。

target[-1] |= static_cast<uint8>(0x80);

value >>= 7;

*target = static_cast<uint8>(value);

++target;

} while (value >= 0x80);

return target;

}

减少 absl::flat_hash_set 和 absl::flat_hash_map 的代码大小。

- 将不依赖具体哈希表类型的代码提取到公共（非内联）函数中。

- 审慎放置 ABSL_ATTRIBUTE_NOINLINE 指令。

- 将一些慢路径移出内联代码。

使一些大型二进制文件的大小减少约 ~0.5%。

在不使用 protobuf arenas 时，不要内联字符串分配和释放。

public/arenastring.h

if (IsDefault(default_value)) {

std::string* new_string = new std::string();

tagged_ptr_.Set(new_string);

return new_string;

} else {

return UnsafeMutablePointer();

}

}

if (IsDefault(default_value)) {

return SetAndReturnNewString();

} else {

return UnsafeMutablePointer();

}

}

internal/arenastring.cc

std::string* ArenaStringPtr::SetAndReturnNewString() {

std::string* new_string = new std::string();

tagged_ptr_.Set(new_string);

return new_string;

}

避免内联某些例程。创建接受 'const char*' 而不是 'const std::string&' 的例程变体，以避免在每个调用点产生 std::string 构造代码。

op.h

class OpDefBuilderWrapper {

public:

explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}

OpDefBuilderWrapper& Attr(std::string spec) {

builder_.Attr(std::move(spec));

return *this;

}

OpDefBuilderWrapper& Input(std::string spec) {

builder_.Input(std::move(spec));

return *this;

}

OpDefBuilderWrapper& Output(std::string spec) {

builder_.Output(std::move(spec));

return *this;

}

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

OpDefBuilderWrapper& Input(std::string spec) {

builder_.Input(std::move(spec));

return *this;

}

OpDefBuilderWrapper& Input(const char* spec) TF_ATTRIBUTE_NOINLINE {

return Input(std::string(spec));

}

OpDefBuilderWrapper& Output(std::string spec) {

builder_.Output(std::move(spec));

return *this;

}

OpDefBuilderWrapper& Output(const char* spec) TF_ATTRIBUTE_NOINLINE {

return Output(std::string(spec));

}

减少模板实例化

模板代码在实例化时，可能会针对模板实参的每一种可能组合被重复生成。

用普通参数替换模板实参。

将一个以 bool 为模板参数的大型例程改为将该 bool 作为额外参数传入。（该 bool 只被使用一次，用于在两个字符串常量中选择一个，因此运行时检查完全可以接受。）这将该大型例程的实例化数量从 287 减少到 143。

sharding_util_ops.cc

template <bool Split>

Status GetAndValidateAttributes(OpKernelConstruction* ctx,

std::vector<int32>& num_partitions,

int& num_slices, std::vector<int32>& paddings,

bool& has_paddings) {

absl::string_view num_partitions_attr_name =

Split ? kNumSplitsAttrName : kNumConcatsAttrName;

...

return OkStatus();

}

Status GetAndValidateAttributes(bool split, OpKernelConstruction* ctx,

std::vector<int32>& num_partitions,

int& num_slices, std::vector<int32>& paddings,

bool& has_paddings) {

absl::string_view num_partitions_attr_name =

split ? kNumSplitsAttrName : kNumConcatsAttrName;

...

return OkStatus();

}

将体积较大的代码从模板化构造函数移到非模板化的共享基类构造函数中。

同时将模板实例化数量从每个 <T, Device, Rank> 组合一个，减少到每个 <T> 一个以及每个 <Rank> 一个。

sharding_util_ops.cc

template <typename Device, typename T>

class XlaSplitNDBaseOp : public OpKernel {

public:

explicit XlaSplitNDBaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

OP_REQUIRES_OK(

ctx, GetAndValidateAttributes(/*split=*/true, ctx, num_splits_,

num_slices_, paddings_, has_paddings_));

}

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

减少 absl::flat_hash_set 和 absl::flat_hash_map 的生成代码大小。

- 将不依赖具体哈希表类型的代码提取到公共（非内联）函数中。

- 审慎放置 ABSL_ATTRIBUTE_NOINLINE 指令。

- 将一些慢路径移出内联代码。

减少容器操作

考虑 map 和其他容器操作的影响，因为每次调用这类操作都可能产生大量生成代码。

将一连串用于初始化 emoji 字符哈希表的 map 插入调用，转换为一次批量插入操作（在链接到许多二进制文件的库中，text 从 188KB 降至 360 字节）。😊

textfallback_init.h

inline void AddEmojiFallbacks(TextFallbackMap *map) {

(*map)[0xFE000] = &kFE000;

(*map)[0xFE001] = &kFE001;

(*map)[0xFE002] = &kFE002;

(*map)[0xFE003] = &kFE003;

(*map)[0xFE004] = &kFE004;

(*map)[0xFE005] = &kFE005;

...

(*map)[0xFEE7D] = &kFEE7D;

(*map)[0xFEEA0] = &kFEEA0;

(*map)[0xFE331] = &kFE331;

};

inline void AddEmojiFallbacks(TextFallbackMap *map) {

#define PAIR(x) {0x##x, &k##x}

// clang-format off

map->insert({

PAIR(FE000),

PAIR(FE001),

PAIR(FE002),

PAIR(FE003),

PAIR(FE004),

PAIR(FE005),

...

PAIR(FEE7D),

PAIR(FEEA0),

PAIR(FE331)});

// clang-format on

#undef PAIR

};

停止内联一个大量使用 InlinedVector 操作的函数。

将一个曾经从 .h 文件内联的很长例程移动到 .cc（内联它并没有真正的性能收益）。

reduction_ops_common.h

Status Simplify(const Tensor& data, const Tensor& axis,

const bool keep_dims) {

... 八十行例程主体 ...

}

Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

并行化与同步

利用并行性

现代机器有许多核心，而这些核心经常未被充分利用。因此，可以通过并行化使昂贵的工作更快完成。最常见的方法是并行处理不同项，并在完成后合并结果。通常会先将这些项划分为批次，以避免为每一项并行运行都付出成本。

四路并行化将编码 token 的速率提高了约 ~3.6x。

blocked-token-coder.cc

MutexLock l(&encoder_threads_lock);

if (encoder_threads == NULL) {

encoder_threads = new ThreadPool(NumCPUs());

encoder_threads->SetStackSize(262144);

encoder_threads->StartWorkers();

}

encoder_threads->Add

(NewCallback(this,

&BlockedTokenEncoder::EncodeRegionInThread,

region_tokens, N, region,

stats,

controller_->GetClosureWithCost

(NewCallback(&DummyCallback), N)));

并行化将解码性能提高了 5x。

coding.cc

for (int c = 0; c < clusters->size(); c++) {

RET_CHECK_OK(DecodeBulkForCluster(...);

}

struct SubTask {

absl::Status result;

absl::Notification done;

};

std::vector<SubTask> tasks(clusters->size());

for (int c = 0; c < clusters->size(); c++) {

options_.executor->Schedule([&, c] {

tasks[c].result = DecodeBulkForCluster(...);

tasks[c].done.Notify();

});

}

for (int c = 0; c < clusters->size(); c++) {

tasks[c].done.WaitForNotification();

}

for (int c = 0; c < clusters->size(); c++) {

RETURN_IF_ERROR(tasks[c].result);

}

应仔细衡量对系统性能的影响——如果没有空闲 CPU，或内存带宽已经饱和，并行化可能没有帮助，甚至可能有害。

摊销锁获取成本

避免细粒度加锁，以降低热路径中 Mutex 操作的成本。注意：只有在变更不会增加锁竞争时才应这样做。

获取一次锁来释放整个查询节点树，而不是对树中的每个节点都重新获取锁。

mustang-query.cc

// 查询节点池

ThreadSafeFreeList<MustangQuery> pool_(256);

...

void MustangQuery::Release(MustangQuery* node) {

if (node == NULL)

return;

for (int i=0; i < node->children_->size(); ++i)

Release((*node->children_)[i]);

node->children_->clear();

pool_.Delete(node);

}

// 查询节点池

Mutex pool_lock_;

FreeList<MustangQuery> pool_(256);

...

void MustangQuery::Release(MustangQuery* node) {

if (node == NULL)

return;

MutexLock l(&pool_lock_);

ReleaseLocked(node);

}

void MustangQuery::ReleaseLocked(MustangQuery* node) {

#ifndef NDEBUG

pool_lock_.AssertHeld();

#endif

if (node == NULL)

return;

for (int i=0; i < node->children_->size(); ++i)

ReleaseLocked((*node->children_)[i]);

node->children_->clear();

pool_.Delete(node);

}

保持临界区短小

避免在临界区内执行昂贵工作。尤其要注意看起来无害、但可能正在执行 RPC 或访问文件的代码。

减少临界区中触及的缓存行数量。

谨慎的数据结构调整显著减少了访问的缓存行数量，并将一次 ML 训练运行的性能提升了 3.3%。

- 将一些每节点类型属性预计算为 NodeItem 数据结构中的 bit，这意味着我们可以避免在临界区中为出边触及 Node* 对象。

- 将 ExecutorState::ActivateNodes 改为对每条出边使用目标节点的 NodeItem，而不是触及 *item->node 对象中的字段。通常这意味着访问所需边数据时总共只触及 1 或 2 个缓存行，而不是 ~2 + O(num_outgoing edges)（并且对于由许多核心执行的大型图，TLB 压力也更小）。

避免在持有 Mutex 时执行 RPC。

trainer.cc

{

// 通知参数服务器我们正在启动。

MutexLock l(&lock_);

model_ = model;

MaybeRecordProgress(last_global_step_);

}

bool should_start_record_progress = false;

int64 step_for_progress = -1;

{

// 通知参数服务器我们正在启动。

MutexLock l(&lock_);

model_ = model;

should_start_record_progress = ShouldStartRecordProgress();

step_for_progress = last_global_step_;

}

if (should_start_record_progress) {

StartRecordProgress(step_for_progress);

}

此外，要警惕在 Mutex 解锁之前运行的昂贵析构函数（当 Mutex 解锁由 ~MutexUnlock 触发时，这种情况经常发生）。在 MutexLock 之前声明带有昂贵析构函数的对象可能会有所帮助（前提是它是线程安全的）。

通过分片减少竞争

有时，一个由 Mutex 保护且表现出高竞争的数据结构，可以安全地拆分为多个分片，每个分片都有自己的 Mutex。（注意：这要求不同分片之间不存在跨分片不变量。）

将一个缓存分为 16 个分片，在多线程负载下将吞吐量提升约 ~2x。

cache.cc

class ShardedLRUCache : public Cache {

private:

LRUCache shard_[kNumShards];

port::Mutex id_mutex_;

uint64_t last_id_;

static inline uint32_t HashSlice(const Slice& s) {

return Hash(s.data(), s.size(), 0);

}

static uint32_t Shard(uint32_t hash) {

return hash >> (32 - kNumShardBits);

}

...

virtual Handle* Lookup(const Slice& key) {

const uint32_t hash = HashSlice(key);

return shard_[Shard(hash)].Lookup(key, hash);

}

对用于跟踪调用的 spanner 数据结构进行分片。

transaction_manager.cc

absl::MutexLock l(&active_calls_in_mu_);

ActiveCallMap::const_iterator iter = active_calls_in_.find(m->tid());

if (iter != active_calls_in_.end()) {

iter->second.ExtractElements(&m->tmp_calls_);

}

ActiveCalls::LockedShard shard(active_calls_in_, m->tid());

const ActiveCallMap& active_calls_map = shard.active_calls_map();

ActiveCallMap::const_iterator iter = active_calls_map.find(m->tid());

if (iter != active_calls_map.end()) {

iter->second.ExtractElements(&m->tmp_calls_);

}

如果所讨论的数据结构是 map，请考虑改用并发哈希 map 实现。

谨慎处理用于分片选择的信息。例如，如果使用哈希值的某些 bit 进行分片选择，而这些相同 bit 后续又被再次使用，那么后续使用的性能可能很差，因为它看到的是偏斜的哈希值分布。

修复用于分片选择的信息，以防止哈希表问题。

netmon_map_impl.h

ConnectionBucket* GetBucket(Index index) {

// 对 hash 再次进行 rehash，以确保我们不是基于原始 hash 对 buckets 进行分区。

// 如果 num_buckets_ 是 2 的幂，那会降低 buckets 的熵。

size_t original_hash = absl::Hash<Index>()(index);

int hash = absl::Hash<size_t>()(original_hash) % num_buckets_;

return &buckets_[hash];

}

ConnectionBucket* GetBucket(Index index) {

absl::Hash<std::pair<Index, size_t>> hasher{};

// 将 hash 与 42 组合，以防止分片选择使用相同的 bit

// 作为底层哈希表。

return &buckets_[hasher({index, 42}) % num_buckets_];

}

对用于跟踪调用的 Spanner 数据结构进行分片。

此变更列表（CL）将 ActiveCallMap 划分为 64 个分片。每个分片由单独的 mutex 保护。给定事务会被映射到恰好一个分片。新增了接口 LockedShard(tid)，用于以线程安全的方式访问某个事务的 ActiveCallMap。示例用法：

transaction_manager.cc

{

absl::MutexLock l(&active_calls_in_mu_);

delayed_locks_timer_ring_.Add(delayed_locks_flush_time_ms, tid);

}

{

ActiveCalls::LockedShard shard(active_calls_in_, tid);

shard.delayed_locks_timer_ring().Add(delayed_locks_flush_time_ms, tid);

}

结果显示，在使用 8192 个 fiber 运行基准测试时，整体挂钟时间减少了 69%。

Benchmark                   Time(ns)        CPU(ns)     Iterations

------------------------------------------------------------------

BM_ActiveCalls/8k        11854633492     98766564676            10

BM_ActiveCalls/16k       26356203552    217325836709            10

Benchmark                   Time(ns)        CPU(ns)     Iterations

------------------------------------------------------------------

BM_ActiveCalls/8k         3696794642     39670670110            10

BM_ActiveCalls/16k        7366284437     79435705713            10

SIMD 指令

探索使用现代 CPU 上可用的 SIMD 指令一次处理多个条目是否能够带来加速（例如，参见下文批量操作（Bulk Operations）一节中关于 absl::flat_hash_map 的讨论）。

减少伪共享

如果不同线程访问不同的可变数据，请考虑将不同数据项放在不同的缓存行上，例如在 C++ 中使用 alignas 指令。不过，这些指令容易被误用，并且可能显著增加对象大小，因此请确保性能测量能够证明其使用是合理的。

将经常变更的字段与其他字段隔离到不同的缓存行中。

histogram.h

HistogramOptions options_;

...

internal::HistogramBoundaries *boundaries_;

...

std::vector<double> buckets_;

double min_;             // 最小值。

double max_;             // 最大值。

double count_;           // 出现次数总计。

double sum_;             // 值的总和。

double sum_of_squares_;  // 值的平方和。

...

RegisterVariableExporter *exporter_;

HistogramOptions options_;

...

internal::HistogramBoundaries *boundaries_;

...

RegisterVariableExporter *exporter_;

...

// 将以下字段放入专用缓存行，因为它们会被频繁

// 变更，这样可以避免潜在的伪共享。

...

#ifndef SWIG

alignas(ABSL_CACHELINE_SIZE)

#endif

std::vector<double> buckets_;

double min_;             // 最小值。

double max_;             // 最大值。

double count_;           // 出现次数总计。

double sum_;             // 值的总和。

double sum_of_squares_;  // 值的平方和。

减少上下文切换频率

将小型工作项内联处理，而不是放到设备线程池中处理。

cast_op.cc

template <typename Device, typename Tout, typename Tin>

void CastMaybeInline(const Device& d, typename TTypes<Tout>::Flat o,

typename TTypes<Tin>::ConstFlat i) {

if (o.size() * (sizeof(Tin) + sizeof(Tout)) < 16384) {

// CPU 上的小型转换：内联执行

o = i.template cast<Tout>();

} else {

o.device(d) = i.template cast<Tout>();

}

}

使用带缓冲的 channel 进行流水线处理

Channel 可以是无缓冲的，这意味着写入者会阻塞，直到读取者准备好取走一个条目。当 channel 用于同步时，无缓冲 channel 可能很有用；但当 channel 用于提升并行度时则不适合。

考虑无锁方法

有时，无锁数据结构相比更传统的 mutex 保护数据结构能带来差异。不过，直接操作原子变量可能很危险。优先使用更高层次的抽象。

使用无锁 map 管理 RPC channel 的缓存。

RPC stub 缓存中的条目每秒会被读取数千次，但很少被修改。切换到合适的无锁 map 可将查找延迟降低 3%-5%。

使用固定词典 + 无锁哈希 map 来加速判断 IsValidTokenId。

dynamic_token_class_manager.h

mutable Mutex mutex_;

// 此哈希 map 的密度由以下事实保证：

// dynamic lexicon 在尝试分配新的 TokenId 之前，

// 会复用先前已分配的 TokenId。

dense_hash_map<TokenId, common::LocalTokenClassId> tid_to_cid_

GUARDED_BY(mutex_);

// 对此 hash-map 的读取访问应使用

// 'epoch_gc_'::(EnterFast / LeaveFast)。写入者应定期

// 对已删除条目执行 GC，只需调用 LockFreeHashMap::CreateGC。

typedef util::gtl::LockFreeHashMap<TokenId, common::LocalTokenClassId>

TokenIdTokenClassIdMap;

TokenIdTokenClassIdMap tid_to_cid_;

Protocol Buffer 建议

Protobuf 是一种方便的数据表示形式，尤其适合需要通过网络发送或持久化存储的数据。不过，它们可能带来显著的性能成本。例如，一段填充 1000 个点的列表然后对 Y 坐标求和的代码，从 protobuf 转换为结构体的 C++ std::vector 后，速度提升了 20 倍！

两个版本的基准测试代码。

name                old time/op  new time/op  delta

BenchmarkIteration  17.4µs ± 5%   0.8µs ± 1%  -95.30%  (p=0.000 n=11+12)

Protobuf 版本：

message PointProto {

int32 x = 1;

int32 y = 2;

}

message PointListProto {

repeated PointProto points = 1;

}

void SumProto(const PointListProto& vec) {

int sum = 0;

for (const PointProto& p : vec.points()) {

sum += p.y();

}

ABSL_VLOG(1) << sum;

}

void BenchmarkIteration() {

PointListProto points;

points.mutable_points()->Reserve(1000);

for (int i = 0; i < 1000; i++) {

PointProto* p = points.add_points();

p->set_x(i);

p->set_y(i * 2);

}

SumProto(points);

}

非 protobuf 版本：

struct PointStruct {

int x;

int y;

};

void SumVector(const std::vector<PointStruct>& vec) {

int sum = 0;

for (const PointStruct& p : vec) {

sum += p.y;

}

ABSL_VLOG(1) << sum;

}

void BenchmarkIteration() {

std::vector<PointStruct> points;

points.reserve(1000);

for (int i = 0; i < 1000; i++) {

points.push_back({i, i * 2});

}

SumVector(points);

}

此外，protobuf 版本会向二进制文件增加几 KB 的代码和数据，这看起来可能不多，但在包含许多 protobuf 类型的系统中会迅速累积。增大的体积会造成 i-cache 和 d-cache 压力，从而产生性能问题。

以下是一些与 protobuf 性能相关的建议：

不要不必要地使用 protobuf。

鉴于上文所述 20 倍的性能差异，如果某些数据从不会被序列化或解析，你可能不应该把它放入 protocol buffer。Protocol buffer 的目的是让数据结构易于序列化和反序列化，但它们可能带来显著的代码大小、内存和 CPU 开销。如果你只是想要 DebugString 和可复制性等其他便利特性，请不要使用它们。

避免不必要的消息层级。

消息层级可用于以更可读的方式组织信息。不过，额外一层消息层级会带来内存分配、函数调用、缓存未命中、更大的序列化消息等开销。

例如，不要使用：

message Foo {

optional Bar bar = 1;

}

message Bar {

optional Baz baz = 1;

}

message Baz {

optional int32 count = 1;

}

优先使用：

message Foo {

optional int32 count = 1;

}

protocol buffer 消息对应于 C++ 生成代码中的一个消息类，并会在线路格式中发出标签和 payload 长度。为了携带一个整数，旧形式需要更多分配（和释放），并会生成更多代码。因此，所有 protocol buffer 操作（解析、序列化、大小计算等）都必须遍历消息树，成本更高。新形式没有这类开销，效率更高。

对频繁出现的字段使用较小的字段编号。

Protobuf 使用可变长度整数表示字段编号和线路格式的组合（参见 protobuf 编码文档）。对于 1 到 15 之间的字段编号，该表示占 1 字节；对于 16 到 2047 之间的字段编号，占 2 字节。（通常应避免使用 2048 或更大的字段编号。）

考虑为性能敏感的 protobuf 未来扩展预留一些较小的字段编号。

在 int32、sint32、fixed32 和 uint32（以及对应的 64 位变体）之间谨慎选择。

通常使用 int32 或 int64，但对于哈希码等大值使用 fixed32 或 fixed64，对于经常为负的值使用 sint32 或 sint64。

varint 在编码小整数时占用更少字节，可以节省空间，但代价是解码更昂贵。不过，对于负值或大值，它可能占用更多空间。在这种情况下，使用 fixed32 或 fixed64（而不是 uint32 或 uint64）可以以更低的编码和解码成本减少大小。对于小的负整数，请使用 sint32 或 sint64，而不是 int32 或 int64.d

对于 proto2，通过使用 [packed=true] 注解来打包 repeated 数值字段。

在 proto2 中，repeated 值默认序列化为一系列 (tag, value) 对。这效率较低，因为必须为每个元素解码标签。

打包的 repeated primitive 会先序列化 payload 的长度，然后是不带标签的值。使用固定宽度值时，我们可以在开始解析的那一刻就知道最终大小，从而避免重新分配；也就是说，没有重新分配成本。我们仍然不知道 payload 中有多少个 varint，可能仍需承担重新分配成本。

在 proto3 中，repeated 字段默认会被打包。

打包最适合 fixed32、fixed64、float、double 等固定宽度值，因为整个编码长度可以通过元素数量乘以固定值大小预先确定，而不必计算每个单独元素的长度。

对二进制数据和大值使用 bytes，而不是 string。

string 类型保存 UTF8 编码文本，有时可能需要验证。bytes 类型可以保存任意字节序列（非文本数据），通常比 string 更合适，也更高效。

考虑使用 string_type = VIEW 以避免复制。

解析期间复制大的 string 或 bytes 字段成本很高。通过用 string_type = VIEW 标记字段，通常可以避免这种成本。

message Image {

...

bytes jpeg_encoding = 4 [features.(pb.cpp).string_type=VIEW];

}

如果没有 VIEW 注解，当解析 protocol buffer 时，可能很大的字段内容会从序列化的 protocol buffer 复制到内存中的 string 对象。根据 string 或 bytes 字段的数量及这些字段的大小，复制开销可能很显著。

ParseFromStringWithAliasing 等例程不会复制大的二进制 blob，而是使用 absl::string_view 引用原始 backing string。请注意，backing string（序列化的 protocol buffer）的生命周期必须长于包含该别名的 protocol buffer 实例。

考虑对大字段使用 Cord，以降低复制成本。

使用 [ctype=CORD] 注解大的 bytes 和 string 字段可能会降低复制成本。此注解会将字段表示从 std::string 改为 absl::Cord。absl::Cord 使用引用计数和基于树的存储来降低复制和追加成本。如果 protocol buffer 被序列化到 cord 中，解析带有 [ctype=CORD] 的 string 或 bytes 字段可以避免复制字段内容。

message Document {

...

bytes html = 4 [ctype = CORD];

}

Cord 字段的性能取决于长度分布和访问模式。请使用基准测试验证此类更改。

在 C++ 代码中使用 protobuf arena。

考虑使用 arena 来节省分配和释放成本，尤其是对于包含 repeated、string 或 message 字段的 protobuf。

message 和 string 字段在堆上分配（即使顶层 protocol buffer 对象是在栈上分配的）。如果一个 protocol buffer 消息包含大量子消息字段和 string 字段，分配和释放成本可能很显著。Arena 会摊销分配成本，并使释放几乎免费。它还通过从连续内存块中分配来改善内存局部性。

保持 .proto 文件小巧

不要在单个 .proto 文件中放入太多消息。一旦你依赖 .proto 文件中的任何内容，整个文件都会被链接器拉入，即使其中大部分未被使用。这会增加构建时间和二进制大小。你可以使用 extensions 和 Any 来避免对包含许多消息类型的大型 .proto 文件创建硬依赖。

考虑以序列化形式存储 protocol buffer，即使在内存中也是如此。

内存中的 protobuf 对象内存占用很大（通常是线路格式大小的 5 倍），并且可能分散在许多缓存行上。因此，如果你的应用需要长时间保持许多 protobuf 对象存活，请考虑以序列化形式存储它们。

避免 protobuf map 字段。

Protobuf map 字段存在性能问题，通常超过它们提供的少量语法便利。优先使用由 protobuf 内容初始化的非 protobuf map：

msg.proto

map<string, bytes> env_variables = 5;

message Var {

string key = 1;

bytes value = 2;

}

repeated Var env_variables = 5;

使用只包含部分字段的 protobuf 消息定义。

如果你只想访问大型消息类型中的少数字段，请考虑定义自己的 protocol buffer 消息类型，它模仿原始类型，但只定义你关心的字段。示例如下：

message FullMessage {

optional int32 field1 = 1;

optional BigMessage field2 = 2;

optional int32 field3 = 3;

repeater AnotherBigMessage field4 = 4;

...

optional int32 field100 = 100;

}

message SubsetMessage {

optional int32 field3 = 3;

optional int32 field88 = 88;

}

通过将序列化的 FullMessage 解析为 SubsetMessage，只会解析一百个字段中的两个，其他字段会被视为未知字段。适当时，请考虑使用会丢弃未知字段的 API，以进一步提升性能。

尽可能复用 protobuf 对象。

在循环外声明 protobuf 对象，以便它们已分配的存储可以在循环迭代之间复用。

C++ 专项建议

absl::flat_hash_map（以及 set）

Absl 哈希表通常优于 C++ 标准库容器，例如 std::map 和 std::unordered_map。

加速 LanguageFromCode（使用 absl::flat_hash_map 代替 __gnu_cxx::hash_map）。

languages.cc

class CodeToLanguage

...

: public __gnu_cxx::hash_map<absl::string_view, i18n::languages::Language,

CodeHash, CodeCompare> {

class CodeToLanguage

...

: public absl::flat_hash_map<absl::string_view, i18n::languages::Language,

CodeHash, CodeCompare> {

基准测试结果：

name               old time/op  new time/op  delta

BM_CodeToLanguage  19.4ns ± 1%  10.2ns ± 3%  -47.47%  (p=0.000 n=8+10)

加速 stats publish/unpublish（这是较早的更改，因此使用 dense_hash_map，而不是当时尚不存在的 absl::flat_hash_map）。

publish.cc

typedef hash_map<uint64, Publication*> PublicationMap;

static PublicationMap* publications = NULL;

typedef dense_hash_map<uint64, Publication*> PublicationMap;;

static PublicationMap* publications GUARDED_BY(mu) = NULL;

使用 dense_hash_map 代替 hash_map 来跟踪 SelectServer alarm（如今会使用 absl::flat_hash_map）。

alarmer.h

typedef hash_map<int, Alarm*> AlarmList;

typedef dense_hash_map<int, Alarm*> AlarmList;

absl::btree_map/absl::btree_set

absl::btree_map 和 absl::btree_set 在每个树节点中存储多个条目。与有序 C++ 标准库容器（如 std::map）相比，这有许多优势。首先，指向子树节点的指针开销通常会显著减少。其次，由于给定 btree 树节点中的条目或键/值在内存中连续存储，缓存效率通常会显著更好。

使用 btree_set 代替 std::set 来表示使用非常频繁的工作队列。

register_allocator.h

using container_type = std::set<WorklistItem>;

using container_type = absl::btree_set<WorklistItem>;

util::bitmap::InlinedBitVector

util::bitmap::InlinedBitvector 可以内联存储较短的位向量，因此通常会比 std::vector<bool> 或其他位图类型更适合。

使用 InlinedBitVector 代替 std::vector<bool>，然后使用 FindNextBitSet 查找下一个感兴趣的项。

block_encoder.cc

vector<bool> live_reads(nreads);

...

for (int offset = 0; offset < b_.block_width(); offset++) {

...

for (int r = 0; r < nreads; r++) {

if (live_reads[r]) {

util::bitmap::InlinedBitVector<4096> live_reads(nreads);

...

for (int offset = 0; offset < b_.block_width(); offset++) {

...

for (size_t r = 0; live_reads.FindNextSetBit(&r); r++) {

DCHECK(live_reads[r]);

absl::InlinedVector

absl::InlinedVector 会内联存储少量元素（可通过第二个模板参数配置）。这使得元素数量不超过该值的小向量通常具有更好的缓存效率，并且在元素数量较少时也完全避免分配后备存储数组。

在多个位置使用 InlinedVector 代替 std::vector。

bundle.h

class Bundle {

public:

...

private:

// （带槽位的指令，无槽位的立即数操作数）的序列。

std::vector<InstructionRecord> instructions_;

...

};

class Bundle {

public:

...

private:

// （带槽位的指令，无槽位的立即数操作数）的序列。

absl::InlinedVector<InstructionRecord, 2> instructions_;

...

};

gtl::vector32

通过使用只支持可放入 32 位大小的自定义 vector 类型来节省空间。

一个简单的类型更改在 Spanner 中节省了约 8TiB 内存。

table_ply.h

class TablePly {

...

// 返回此表的此文件中存储的数据列集合。

const std::vector<FamilyId>& modified_data_columns() const {

return modified_data_columns_;

}

...

private:

...

std::vector<FamilyId> modified_data_columns_;  // 表中的数据列。

#include "util/gtl/vector32.h"

...

// 返回此表的此文件中存储的数据列集合。

absl::Span<const FamilyId> modified_data_columns() const {

return modified_data_columns_;

}

...

...

// 表中的数据列。

gtl::vector32<FamilyId> modified_data_columns_;

gtl::small_map

gtl::small_map 使用内联数组来存储最多一定数量的唯一键值对元素，但在空间用尽时会自动升级为由用户指定的 map 类型作为后备。

在 tflite_model 中使用 gtl::small_map。

tflite_model.cc

using ChoiceIdToContextMap = gtl::flat_hash_map<int, TFLiteContext*>;

using ChoiceIdToContextMap =

gtl::small_map<gtl::flat_hash_map<int, TFLiteContext*>>;

gtl::small_ordered_set

gtl::small_ordered_set 是一种针对关联容器（如 std::set 或 absl::btree_multiset）的优化。它使用固定数组来存储一定数量的元素，然后在空间用尽时退回到使用 set 或 multiset。对于通常较小的集合，这会比直接使用 set 之类的容器快得多，因为 set 针对大型数据集进行了优化。此更改缩小了缓存占用，并减少了临界区长度。

使用 gtl::small_ordered_set 保存监听器集合。

broadcast_stream.h

class BroadcastStream : public ParsedRtpTransport {

...

private:

...

std::set<ParsedRtpTransport*> listeners_ ABSL_GUARDED_BY(listeners_mutex_);

};

class BroadcastStream : public ParsedRtpTransport {

...

private:

...

using ListenersSet =

gtl::small_ordered_set<std::set<ParsedRtpTransport*>, 10>;

ListenersSet listeners_ ABSL_GUARDED_BY(listeners_mutex_);

gtl::intrusive_list

gtl::intrusive_list<T> 是一种双向链表，其中链接指针嵌入在 T 类型的元素中。与 std::list<T*> 相比，它为每个元素节省一个缓存行加一次间接访问。

使用 intrusive_list 跟踪每个索引行更新的正在进行中的请求。

row-update-sender-inflight-set.h

std::set<int64> inflight_requests_ GUARDED_BY(mu_);

class SeqNum : public gtl::intrusive_link<SeqNum> {

...

int64 val_ = -1;

...

};

...

gtl::intrusive_list<SeqNum> inflight_requests_ GUARDED_BY(mu_);

限制 absl::Status 和 absl::StatusOr 的使用

尽管 absl::Status 和 absl::StatusOr 类型相当高效，但即使在成功路径上也有非零开销，因此对于不需要返回任何有意义错误细节（甚至可能永远不会失败！）的热例程，应避免使用它们：

避免 RoundUpToAlignment() 函数使用 StatusOr<int64> 返回类型。

best_fit_allocator.cc

absl::StatusOr<int64> BestFitAllocator::RoundUpToAlignment(int64 bytes) const {

TPU_RET_CHECK_GE(bytes, 0);

const int64 max_aligned = MathUtil::RoundDownTo<int64>(

std::numeric_limits<int64>::max(), alignment_in_bytes_);

if (bytes > max_aligned) {

return util::ResourceExhaustedErrorBuilder(ABSL_LOC)

<< "Attempted to allocate "

<< strings::HumanReadableNumBytes::ToString(bytes)

<< " which after aligning to "

<< strings::HumanReadableNumBytes::ToString(alignment_in_bytes_)

<< " cannot be expressed as an int64.";

}

return MathUtil::RoundUpTo<int64>(bytes, alignment_in_bytes_);

}

best_fit_allocator.h

// 将 bytes 向上取整到 alignment_ 的最近倍数。

// REQUIRES: bytes >= 0.

// REQUIRES: result does not overflow int64.

// REQUIRES: alignment_in_bytes_ is a power of 2 (checked in constructor).

int64 RoundUpToAlignment(int64 bytes) const {

DCHECK_GE(bytes, 0);

DCHECK_LE(bytes, max_aligned_bytes_);

int64 result =

((bytes + (alignment_in_bytes_ - 1)) & ~(alignment_in_bytes_ - 1));

DCHECK_EQ(result, MathUtil::RoundUpTo<int64>(bytes, alignment_in_bytes_));

return result;

}

添加 ShapeUtil::ForEachIndexNoStatus，以避免为张量的每个元素创建 Status 返回对象。

shape_util.h

using ForEachVisitorFunction =

absl::FunctionRef<StatusOr<bool>(absl::Span<const int64_t>)>;

...

static void ForEachIndex(const Shape& shape, absl::Span<const int64_t> base,

absl::Span<const int64_t> count,

absl::Span<const int64_t> incr,

const ForEachVisitorFunction& visitor_function);

using ForEachVisitorFunctionNoStatus =

absl::FunctionRef<bool(absl::Span<const int64_t>)>;

...

static void ForEachIndexNoStatus(

const Shape& shape, absl::Span<const int64_t> base,

absl::Span<const int64_t> count, absl::Span<const int64_t> incr,

const ForEachVisitorFunctionNoStatus& visitor_function);

literal.cc

ShapeUtil::ForEachIndex(

result_shape, [&](absl::Span<const int64_t> output_index) {

for (int64_t i = 0, end = dimensions.size(); i < end; ++i) {

scratch_source_index[i] = output_index[dimensions[i]];

}

int64_t dest_index = IndexUtil::MultidimensionalIndexToLinearIndex(

result_shape, output_index);

int64_t source_index = IndexUtil::MultidimensionalIndexToLinearIndex(

shape(), scratch_source_index);

memcpy(dest_data + primitive_size * dest_index,

source_data + primitive_size * source_index, primitive_size);

return true;

});

ShapeUtil::ForEachIndexNoStatus(

result_shape, [&](absl::Span<const int64_t> output_index) {

// 计算 dest_index

int64_t dest_index = IndexUtil::MultidimensionalIndexToLinearIndex(

result_shape, result_minor_to_major, output_index);

// 计算 source_index

int64_t source_index;

for (int64_t i = 0, end = dimensions.size(); i < end; ++i) {

scratch_source_array[i] = output_index[dimensions[i]];

}

if (src_shape_dims == 1) {

// 此情况的快速路径

source_index = scratch_source_array[0];

DCHECK_EQ(source_index,

IndexUtil::MultidimensionalIndexToLinearIndex(

src_shape, src_minor_to_major, scratch_source_span));

} else {

source_index = IndexUtil::MultidimensionalIndexToLinearIndex(

src_shape, src_minor_to_major, scratch_source_span);

}

// 将一个元素从 source 中的 source_index 移动到 dest 中的 dest_index

memcpy(dest_data + PRIMITIVE_SIZE * dest_index,

source_data + PRIMITIVE_SIZE * source_index, PRIMITIVE_SIZE);

return true;

});

在 TF_CHECK_OK 中，避免为了测试 ok() 而创建 Ok 对象。

status.h

#define TF_CHECK_OK(val) CHECK_EQ(::tensorflow::Status::OK(), (val))

#define TF_QCHECK_OK(val) QCHECK_EQ(::tensorflow::Status::OK(), (val))

extern tensorflow::string* TfCheckOpHelperOutOfLine(

const ::tensorflow::Status& v, const char* msg);

inline tensorflow::string* TfCheckOpHelper(::tensorflow::Status v,

const char* msg) {

if (v.ok()) return nullptr;

return TfCheckOpHelperOutOfLine(v, msg);

}

#define TF_CHECK_OK(val)                                           \

while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \

LOG(FATAL) << *(_result)

#define TF_QCHECK_OK(val)                                          \

while (tensorflow::string* _result = TfCheckOpHelper(val, #val)) \

LOG(QFATAL) << *(_result)

从远程过程调用（RPC）的热路径中移除 StatusOr。

从热路径中移除 StatusOr，消除了早先更改在 RPC 基准测试中导致的 14% CPU 回退。

privacy_context.h

absl::StatusOr<privacy::context::PrivacyContext> GetRawPrivacyContext(

const CensusHandle& h);

privacy_context_statusfree.h

enum class Result {

kSuccess,

kNoRootScopedData,

kNoPrivacyContext,

kNoDDTContext,

kDeclassified,

kNoPrequestContext

};

...

Result GetRawPrivacyContext(const CensusHandle& h,

PrivacyContext* privacy_context);

批量操作

如果可能，一次处理许多项，而不是一次只处理一个项。

absl::flat_hash_map 使用单条 SIMD 指令，从一组键中为每个键比较一个哈希字节。

参见 Swiss Table Design Notes 以及 Matt Kulukundis 相关的 CppCon 2017 和 CppCon 2019 演讲。

raw_hash_set.h

// 返回一个表示与 hash 匹配的槽位位置的位掩码。

BitMask<uint32_t> Match(h2_t hash) const {

auto ctrl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));

auto match = _mm_set1_epi8(hash);

return BitMask<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(match, ctrl)));

}

执行一次性操作来处理许多字节并修正结果，而不是检查每个字节该做什么。

ordered-code.cc

int len = 0;

while (val > 0) {

len++;

buf[9 - len] = (val & 0xff);

val >>= 8;

}

buf[9 - len - 1] = (unsigned char)len;

len++;

FastStringAppend(dest, reinterpret_cast<const char*>(buf + 9 - len), len);

BigEndian::Store(val, buf + 1);  // 可能需要 buf[0] 存放长度

const unsigned int length = OrderedNumLength(val);

char* start = buf + 9 - length - 1;

*start = length;

AppendUpto9(dest, start, length + 1);

通过以分块方式更高效地处理多个交错输入缓冲区，提高 Reed-Solomon 处理速度。

运行于 (12 X 3501 MHz CPUs)；2016-09-27T16:04:55.065995192-04:00

CPU: Intel Haswell with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:15MB

基准测试                          基线 (ns)  新版 (ns) 改进

------------------------------------------------------------------

BM_OneOutput/3/2                      466867    351818    +24.6%

BM_OneOutput/4/2                      563130    474756    +15.7%

BM_OneOutput/5/3                      815393    688820    +15.5%

BM_OneOutput/6/3                      897246    780539    +13.0%

BM_OneOutput/8/4                     1270489   1137149    +10.5%

BM_AllOutputs/3/2                     848772    642942    +24.3%

BM_AllOutputs/4/2                    1067647    638139    +40.2%

BM_AllOutputs/5/3                    1739135   1151369    +33.8%

BM_AllOutputs/6/3                    2045817   1456744    +28.8%

BM_AllOutputs/8/4                    3012958   2484937    +17.5%

BM_AllOutputsSetUpOnce/3/2            717310    493371    +31.2%

BM_AllOutputsSetUpOnce/4/2            833866    600060    +28.0%

BM_AllOutputsSetUpOnce/5/3           1537870   1137357    +26.0%

BM_AllOutputsSetUpOnce/6/3           1802353   1398600    +22.4%

BM_AllOutputsSetUpOnce/8/4           3166930   2455973    +22.4%

一次解码四个整数（大约 2004 年）。

引入了一种 GroupVarInt 格式，它一次以 5-17 字节编码/解码 4 个变长整数，而不是一次处理一个整数。使用新格式解码一组 4 个整数所需时间约为单独解码 4 个 varint 编码整数的 1/3。

groupvarint.cc

const char* DecodeGroupVar(const char* p, int N, uint32* dest) {

assert(groupvar_initialized);

assert(N % 4 == 0);

while (N) {

uint8 tag = *p;

p++;

uint8* lenptr = &groupvar_table[tag].length[0];

#define GET_NEXT                                        \

do {                                                \

uint8 len = *lenptr;                              \

*dest = UNALIGNED_LOAD32(p) & groupvar_mask[len]; \

dest++;                                           \

p += len;                                         \

lenptr++;                                         \

} while (0)

GET_NEXT;

GET_NEXT;

GET_NEXT;

GET_NEXT;

#undef GET_NEXT

N -= 4;

}

return p;

}

一次编码 4 个 k 位数字组成的组。

添加了 KBitStreamEncoder 和 KBitStreamDecoder 类，用于一次将 4 个 k 位数字编码/解码到位流中。由于 K 在编译时已知，编码和解码可以非常高效。例如，由于一次编码四个数字，代码可以假定流始终是字节对齐的（对于偶数 k），或半字节对齐的（对于奇数 k）。

展示多种技术的变更列表（CL）

有时，一个变更列表（CL）会包含多个提升性能的更改，并使用前面许多技术。观察这些变更列表（CL）中的更改类型，有时是培养思维方式的一种好方法：当系统某部分已被识别为瓶颈后，如何进行通用更改以加速其性能。

将 GPU 内存分配器提速约 40%。

GPUBFCAllocator 的分配/释放速度提升 36-48%：

- 通过句柄编号而不是指向 Chunk 的指针来识别块。Chunk 数据结构现在分配在 vector<Chunk> 中，句柄是该 vector 中的索引，用于引用特定块。这使得 Chunk 中的 next 和 prev 指针可以是 ChunkHandle（4 字节），而不是 Chunk*（8 字节）。

- 当 Chunk 对象不再使用时，我们维护一个 Chunk 对象空闲列表，其头部由 ChunkHandle free_chunks_list_ 指定，并且 Chunk->next 指向下一个空闲列表条目。结合 (1)，这使我们可以在分配器中避免 Chunk 对象的堆分配/释放，除了 vector<Chunk> 增长时（很少发生）。它还使 Chunk 对象的所有内存连续。

- 不再让 bins_ 数据结构成为 std::set 并使用 lower_bound 根据 byte_size 定位合适的 bin，而是使用一个 bin 数组，并通过一个值为 log₂(byte_size/256) 的函数进行索引。这样可以通过少量位操作定位 bin，而不是进行二叉搜索树查找。它还允许我们在连续数组中为所有 Bin 数据结构分配存储空间，而不是分散在许多不同的缓存行中。这减少了多个线程进行分配时必须在核心之间来回移动的缓存行数量。

- 为 GPUBFCAllocator::AllocateRaw 添加了快速路径，首先尝试在不涉及 retry_helper_ 的情况下分配内存。如果初始尝试失败（返回 nullptr），则进入 retry_helper_，但通常我们可以避免多层过程调用，以及带多个参数的 std::function 的分配/释放。

- 注释掉了大部分 VLOG 调用。需要用于调试时，可以通过取消注释并重新编译来有选择地重新启用这些调用。

添加了多线程基准测试，用于测试竞争情况下的分配。

在我配备 Titan X 显卡的台式机上，将 ptb_word_lm 从每秒 8036 个词提升到每秒 8272 个词（+2.9%）。

运行于 (40 X 2801 MHz CPUs)；2016/02/16-15:12:49

CPU：Intel Ivybridge with HyperThreading (20 cores) dL1:32KB dL2:256KB dL3:25MB

基准测试                          基线 (ns)  新版本 (ns) 改进

------------------------------------------------------------------

BM_Allocation                            347       184    +47.0%

BM_AllocationThreaded/1                  351       181    +48.4%

BM_AllocationThreaded/4                 2470      1975    +20.0%

BM_AllocationThreaded/16               11846      9507    +19.7%

BM_AllocationDelayed/1                   392       199    +49.2%

BM_AllocationDelayed/10                  285       169    +40.7%

BM_AllocationDelayed/100                 245       149    +39.2%

BM_AllocationDelayed/1000                238       151    +36.6%

通过一组杂项变更，将 Pathways 吞吐量提升约 20%。

- 将一批特殊的快速描述符解析函数统一到单个 ParsedDescriptor 类中，并在更多位置使用该类，以避免代价高昂的完整解析调用。

- 将多个 protocol buffer 字段从 string 改为 bytes（避免不必要的 utf-8 检查及相关错误处理代码）。

- DescriptorProto.inlined_contents 现在是 string，而不是 Cord（预期它只用于较小的张量）。这就需要在 tensor_util.cc 中添加一批复制辅助函数（现在需要同时支持 string 和 Cord）。

- 在若干位置使用 flat_hash_map 代替 std::unordered_map。

- 添加了 MemoryManager::LookupMany，供 Stack op 使用，以代替对每个批元素调用 Lookup。此变更减少了锁定等设置开销。

- 移除了 TransferDispatchOp 中一些不必要的字符串创建。

- 在同一进程中将一批 1000 个 1KB 张量从一个组件传输到另一个组件的性能结果：

之前：227.01 steps/sec

之后：272.52 steps/sec（+20% 吞吐量）

通过一系列变更，将 XLA 编译器性能提升约 15%。

一些用于加速 XLA 编译的变更：

- 在 SortComputationsByContent 中，如果比较函数中 a == b，则返回 false，以避免序列化和指纹化很长的 computation 字符串。

- 将 CHECK 改为 DCHECK，以避免在 HloComputation::ComputeInstructionPostOrder 中触及额外的缓存行。

- 避免在 CoreSequencer::IsVectorSyncHoldSatisfied() 中对 front instruction 进行代价高昂的复制。

- 重构双参数 HloComputation::ToString 和 HloComputation::ToCord 例程，使大部分工作通过追加到 std::string 来完成，而不是追加到 Cord。

- 将 PerformanceCounterSet::Increment 改为只执行一次哈希表查找，而不是两次。

- 精简 Scoreboard::Update 代码。

对于一个重要模型，XLA 编译时间整体加速 14%。

加速 Google Meet 应用代码中的底层日志记录。

加速 ScopedLogId，它位于每个数据包的关键路径上。

- 移除了那些似乎只是为了查看不变量是否被违反而存在的 LOG_EVERY_N(ERROR, ...) 消息。

- 内联 PushLogId 和 PopLogid() 例程（因为没有 LOG_EVERY_N_SECONDS(ERROR, ...) 语句后，它们现在已经足够小，可以内联）。

- 改为使用大小为 4 的固定数组和一个 'int size' 变量来维护线程本地状态，而不是使用 InlinedVector<...>。由于我们本来就从未增长到超过大小 4，InlinedVector 的功能比需要的更通用。

基线：基线加上 scoped_logid_test.cc 中用于添加基准测试的代码

新版本：此变更列表

CPU：Intel Ivybridge with HyperThreading (20 cores) dL1:32KB dL2:256KB dL3:25MB

基准测试                                      基线 (ns)    新版本 (ns) 改进

----------------------------------------------------------------------------

BM_ScopedLogId/threads:1                               8           4    +52.6%

BM_ScopedLogId/threads:2                               8           4    +51.9%

BM_ScopedLogId/threads:4                               8           4    +52.9%

BM_ScopedLogId/threads:8                               8           4    +52.1%

BM_ScopedLogId/threads:16                             11           6    +44.0%

通过改进 Shape 处理，将 XLA 编译时间减少约 31%。

用于提高 XLA 编译器性能的若干变更：

- 通过几种方式改进了 ShapeUtil::ForEachIndex... 迭代的性能：

  - 在 ShapeUtil::ForEachState 中，只保存由 span 表示的数组的指针，而不是完整的 span 对象。

  - 预先形成一个指向 ShapeUtil::ForEachState::indexes vector 的 ShapeUtil::ForEachState::indexes_span，而不是在每次循环迭代时从该 vector 构造此 span。

  - 保存一个指向 ShapeUtil::ForEachState::indexes vector 后备存储的 ShapeUtil::ForEachState::indexes_ptr 指针，从而允许在 ShapeUtil::ForEachState::IncrementDim() 中使用简单的数组操作，而不是更昂贵的 vector::operator[] 操作。

  - 保存一个 ShapeUtil::ForEachState::minor_to_major 数组指针，在构造函数中通过调用 shape.layout().minor_to_major().data() 初始化，而不是对每个维度的每次迭代都调用 LayoutUtil::Minor(...)。

  - 内联 ShapeUtil::ForEachState 构造函数和 ShapeUtil::ForEachState::IncrementDim() 例程。

- 对于不需要在传入函数中返回 Status 功能的调用点，改进了 ShapeUtil::ForEachIndex 迭代的性能。通过引入 ShapeUtil::ForEachIndexNoStatus 变体实现这一点，该变体接受 ForEachVisitorFunctionNoStatus（返回普通 bool）。这比接受 ForEachVisitorFunction 的 ShapeUtil::ForEachIndex 例程更快（后者返回 StatusOr<bool>，而这要求我们迭代的每个元素都调用一次代价高昂的 StatusOr<bool> 析构函数）。

  - 在 LiteralBase::Broadcast 和 GenerateReduceOutputElement 中使用了这个 ShapeUtil::ForEachIndexNoStatus 变体。

- 通过几种方式改进了 LiteralBase::Broadcast 的性能：

  - 在 literal.cc 中引入模板化的 BroadcastHelper 例程，针对不同的 primitive 字节大小进行特化（如果没有这一点，primitive_size 就是运行时变量，因此编译器无法很好地优化每个元素发生的 memcpy，而会调用通用 memcpy 路径，该路径假定字节数相当大，尽管在我们的情况下它是很小的 2 的幂（通常为 1、2、4 或 8））。

  - 通过在 LiteralBase::Broadcast 例程开始时只调用一次 'shape()'，避免每次 Broadcast 调用中约 ~(5 + num_dimensions + num_result_elements) 次虚调用中的除一次外的所有调用。那些散布各处、看似无害的 'shape()' 调用最终会归结为 "root_piece().subshape()"，其中 subshape() 是虚函数。

  - 在 BroadcastHelper 例程中，对源维度为一的情况进行特殊处理，并在这种情况下避免调用 IndexUtil::MultiDimensionalIndexToLinearIndex。

  - 在 BroadcastHelper 中，使用 scratch_source_array 指针变量指向 scratch_source_index vector 的后备存储，并直接使用它，以避免在逐元素代码中执行 vector::operator[] 操作。还在 BroadcastHelper 的逐元素循环之外预先计算一个指向 scratch_source_index vector 的 scratch_source_span，以避免在每个元素上都从该 vector 构造 span。

  - 引入了新的三参数 IndexUtil::MultiDimensionalIndexToLinearIndex 变体，由调用者传入与 shape 参数关联的 minor_to_major span。在 BroadcastHelper 中使用它，以便每次 Broadcast 对 src 和 dst shape 各计算一次，而不是对复制的每个元素计算一次。

- 在 ShardingPropagation::GetShardingFromUser 中，对于 HloOpcode::kTuple 情况，只有在我们发现 operand 值得关注时，才调用 user.sharding().GetSubSharding(...)。避免急切调用它，可将一次长时间编译中此例程的 CPU 时间从 43.7s 降到 2.0s。

- 为 ShapeUtil::ForEachIndex、Literal::Broadcast 以及新的 ShapeUtil::ForEachIndexNoStatus 添加了基准测试。

基线包含以下基准测试新增项：

BM_ForEachIndex 和 BM_BroadcastVectorToMatrix（以及用于添加基准测试依赖项的 BUILD 文件变更），但没有其他变更。

新版本是此 CL

运行于 (72 X 1357.56 MHz CPU s) CPU Caches: L1 Data 32 KiB (x36)

L1 Instruction 32 KiB (x36) L2 Unified 1024 KiB (x36) L3 Unified 25344 KiB (x2)

基准测试                                      基线 (ns)    新版本 (ns) 改进

----------------------------------------------------------------------------

BM_MakeShape                                       18.40       18.90     -2.7%

BM_MakeValidatedShape                              35.80       35.60     +0.6%

BM_ForEachIndex/0                                  57.80       55.80     +3.5%

BM_ForEachIndex/1                                  90.90       85.50     +5.9%

BM_ForEachIndex/2                               1973606     1642197     +16.8%

新添加的 ForEachIndexNoStatus 明显快于 ForEachIndex 变体（它只存在于这个新 CL 中，但 BM_ForEachIndexNoStatus/NUM 完成的基准测试工作量与上面的 BM_ForEachIndex/NUM 结果可比）。

基准测试                                      基线 (ns)    新版本 (ns) 改进

----------------------------------------------------------------------------

BM_ForEachIndexNoStatus/0                             0        46.90    ----

BM_ForEachIndexNoStatus/1                             0        65.60    ----

BM_ForEachIndexNoStatus/2                             0     1001277     ----

Broadcast 性能提升约 58%。

基准测试                                      基线 (ns)    新版本 (ns) 改进

----------------------------------------------------------------------------

BM_BroadcastVectorToMatrix/16/16                   5556        2374     +57.3%

BM_BroadcastVectorToMatrix/16/1024               319510      131075     +59.0%

BM_BroadcastVectorToMatrix/1024/1024           20216949     8408188     +58.4%

对大型语言模型进行提前编译得到的宏观结果（程序所做的不仅仅是 XLA 编译，但花在 XLA 相关代码中的时间略少于一半）：

基线程序整体：573 秒 使用此 CL 的程序整体：465 秒（+19% 改进）

运行此程序时编译两个最大 XLA 程序所花的时间：

基线：141s + 143s = 284s 使用此 CL：99s + 95s = 194s（+31% 改进）

在 Plaque（一个分布式执行框架）中将大型程序的编译时间减少约 22%。

用于将编译加速约 22% 的小调整。

- 加速检测两个节点是否共享一个公共源。以前，我们会按排序顺序获取每个节点的源，然后执行排序交集。现在，我们将一个节点的源放入哈希表，然后遍历另一个节点的源并检查该哈希表。

- 在第 1 步中复用同一个临时哈希表。

- 生成已编译 proto 时，保留一个以 pair<package, opname> 为键的单个 btree，而不是 btree 的 btree。

- 在前述 btree 中存储指向 opdef 的指针，而不是将 opdef 复制到 btree 中。

大型程序（约 45K ops）的速度测量：

name             old time/op  new time/op  delta

BM_CompileLarge   28.5s ± 2%   22.4s ± 2%  -21.61%  (p=0.008 n=5+5)

MapReduce 改进（wordcount 基准测试约 2X 加速）。

Mapreduce 加速：

- SafeCombinerMapOutput 类的 combiner 数据结构已更改。不再使用 hash_multimap<SafeCombinerKey, StringPiece>（表中插入的每个唯一 key/value 都有一个哈希表条目），而是使用 hash_map<SafeCombinerKey, ValuePtr*>（其中 ValuePtr 是值及重复计数的链表）。这在三个方面有帮助：

  - 它显著降低了内存使用，因为我们对每个值只使用 "sizeof(ValuePtr) + value_len" 字节，而不是对每个值使用 "sizeof(SafeCombinerKey) + sizeof(StringPiece) + value_len + new hash table entry overhead"。这意味着我们刷新 reducer 缓冲区的频率更低。

  - 它显著更快，因为当我们为表中已存在的 key 插入新值时，可以避免额外的哈希表条目（取而代之的是，只需将该值挂接到该 key 的值链表中）。

  - 由于我们在链表中为每个值关联一个重复计数，因此可以将以下序列表示为：

  - Output(key, "1");

  - Output(key, "1");

  - Output(key, "1");

  - Output(key, "1");

  - Output(key, "1");

作为 "key" 的链表中的单个条目，并带有重复计数 5。在内部，我们会向用户级 combining 函数 yield "1" 五次。（类似技巧也许可以应用在 reduce 端）。

- （次要）向默认 MapReductionBase::KeyFingerprintSharding 函数添加了对 "nshards == 1" 的测试：如果我们只使用 1 个 reduce shard，就完全避免对 key 做指纹计算（因为在这种情况下可以不检查 key 而直接返回 0）。

- 将 combiner 中每次添加 key/value 时都会调用的代码路径中的一些 VLOG(3) 语句改为 DVLOG(3)。

将一个 wordcount 基准测试的时间从 12.56s 降低到 6.55s。

重构 SelectServer 中的 alarm 处理代码，以显著提高其性能（添加+移除一个 alarm 从 771 ns 降到 271 ns）。

重构了 SelectServer 中的 alarm 处理代码，以显著提高其性能。

变更：

- 改为使用 AdjustablePriorityQueue<Alarm> 而不是 set<Alarm*> 作为 AlarmQueue。这显著加快了 alarm 处理，将添加和移除一个 alarm 所需时间从 771 nanoseconds 降到 281 nanoseconds。此变更避免了每次 alarm 设置时的一次分配/释放（针对 STL set 对象中的红黑树节点），并且还提供了好得多的缓存局部性（因为 AdjustablePriorityQueue 是用 vector 实现的堆，而不是红黑树），在 selectserver 循环的每一轮中操作 AlarmQueue 时触及的缓存行更少。

- 将 Alarmer 中的 AlarmList 从 hash_map 转换为 dense_hash_map，以避免每次添加/删除 alarm 时的另一次分配/释放（这也改善了添加/移除 alarm 时的缓存局部性）。

- 移除了 num_alarms_stat_ 和 num_closures_stat_ MinuteTenMinuteHourStat 对象以及相应的导出变量。虽然监控这些看起来不错，但在实践中它们会给关键网络代码增加显著开销。如果我把这些变量保留下来并用 Atomic32 变量而不是 MinuteTenMinuteHourStat，它们仍然会把添加和移除 alarm 的成本从 281 nanoseconds 增加到 340 nanoseconds。

基准测试结果

基准测试                      Time(ns)  CPU(ns) Iterations

-----------------------------------------------------------

BM_AddAlarm/1                       902      771     777777

使用此变更后

基准测试                      Time(ns)  CPU(ns) Iterations

-----------------------------------------------------------

BM_AddAlarm/1                       324      281    2239999

索引服务速度达到 3.3X 性能！

在 2001 年计划从磁盘索引服务切换到内存索引服务时，我们发现了许多性能问题。此变更修复了其中很多问题，并使我们从每秒 150 次内存查询提升到超过 500 次（在双处理器 Pentium III 机器上的 2 GB 内存索引）。

- 对索引块解码速度进行了大量性能改进（微基准测试中从 8.9 MB/s 提升到 13.1 MB/s）。

- 我们现在会在解码期间对块进行校验和检查。这使我们能够实现所有 getsymbol 操作，而无需进行任何边界检查。

- 我们使用一些粗糙的宏，在整个循环期间将 BitDecoder 的各个字段保存在局部变量中，然后在循环结束时再写回。

- 我们使用内联汇编来访问 Intel 芯片上的 `bsf` 指令，用于 getUnary（查找一个字中第一个 1 位的索引）

- 将值解码到 vector 中时，我们在循环外调整 vector 的大小，并且只沿着 vector 移动指针，而不是通过带边界检查的访问来存储每个值。

- 在 docid 解码期间，我们将 docid 保持在局部 docid 空间中，以避免乘以 num_shards_。只有在需要实际 docid 值时，我们才乘以 num_shards_ 并加上 my_shard_。

- IndexBlockDecoder 现在导出一个接口 `AdvanceToDocid`，它返回第一个 docid ≥ “d” 的索引。这允许扫描以局部 docid 为单位完成，而不是在客户端对块中的每个索引调用 GetDocid(index) 时，强制将每个局部 docid 转换为全局 docid。

- 文档位置数据的解码现在按需进行，而不是在客户端请求块内任何文档的位置数据时，就急切地为整个块完成解码。

- 如果正在解码的索引块在距离页边界 4 字节以内的位置结束，我们会将它复制到本地缓冲区。这使我们始终可以通过 4 字节加载来加载位解码缓冲区，而无需担心越过 mmapped 页末尾时发生段错误。

- 我们只初始化各种评分数据结构的前 nterms_ 个元素，而不是初始化所有 MAX_TERMS 个元素（在某些情况下，我们曾经会在每个被评分的文档上不必要地 memset 20K 到 100K 的数据）。

- 当正在计算的值为 0 时，避免对中间评分值执行 round_to_int 和后续计算（在这些情况下，后续计算只是把 `0` 写到我们已经 memset 为 0 的位置，而这也是最常见的情况）。

- 将评分数据结构上的边界检查改为调试模式断言。

进一步阅读

以下是作者认为有帮助的性能相关书籍和文章列表，排名不分先后：

- Agner Fog 的《Optimizing software in C++》。描述了许多用于提升性能的有用底层技术。

- Richard L. Sites 的《Understanding Software Dynamics》。涵盖了用于诊断和修复性能问题的专家方法和高级工具。

- Performance tips of the week - 有用技巧合集。

- Performance Matters - 关于性能的文章合集。

- Daniel Lemire 的博客 - 有趣算法的高性能实现。

- Building Software Systems at Google and Lessons Learned - 一个视频，描述了 Google 十多年间遇到的系统性能问题。

- Jon Bentley 的《Programming Pearls》和《More Programming Pearls: Confessions of a Coder》。这些文章讲述如何从算法出发，最终得到简单而高效的实现。

- Henry S. Warren 的《Hacker’s Delight》。用于解决一些常见问题的位级和算术算法。

- John L. Hennessy 和 David A. Patterson 的《Computer Architecture: A Quantitative Approach》- 涵盖了计算机体系结构的许多方面，包括关注性能的软件开发者应当了解的内容，如缓存、分支预测器、TLB 等。

建议引用

如果你想引用本文档，我们建议：

Jeffrey Dean & Sanjay Ghemawat, Performance Hints, 2025, https://abseil.io/fast/hints.html

或者使用 BibTeX：

@misc{DeanGhemawatPerformance2025,

author = {Dean, Jeffrey and Ghemawat, Sanjay},

title = {Performance Hints},

year = {2025},

howpublished = {\url{https://abseil.io/fast/hints.html}},

}

致谢

许多同事为本文档提供了有益反馈，包括：

- Adrian Ulrich

- Alexander Kuzmin

- Alexei Bendebury

- Alexey Alexandrov

- Amer Diwan

- Austin Sims

- Benoit Boissinot

- Brooks Moses

- Chris Kennelly

- Chris Ruemmler

- Danila Kutenin

- Darryl Gove

- David Majnemer

- Dmitry Vyukov

- Emanuel Taropa

- Felix Broberg

- Francis Birck Moreira

- Gideon Glass

- Henrik Stewenius

- Jeremy Dorfman

- John Dethridge

- Kurt Kluever

- Kyle Konrad

- Lucas Pereira

- Marc Eaddy

- Michael Marty

- Michael Whittaker

- Mircea Trofin

- Misha Brukman

- Nicolas Hillegeer

- Ranjit Mathew

- Rasmus Larsen

- Soheil Hassas Yeganeh

- Srdjan Petrovic

- Steinar H. Gunderson

- Stergios Stergiou

- Steven Timotius

- Sylvain Vignaud

- Thomas Etter

- Thomas Köppe

- Tim Chestnutt

- Todd Lipcon

- Vance Lankhaar

- Victor Costan

- Yao Zuo

- Zhou Fang

- Zuguang Yang

- 性能提示

  - 思考性能的重要性

  - 估算

    - 示例：对十亿个 4 字节数字进行快速排序所需时间

    - 示例：生成包含 30 个图片缩略图的网页所需时间

  - 测量

    - 性能剖析工具和技巧

    - 当性能剖析结果很平坦时该怎么办

  - API 考量

    - 批量 API

    - 视图类型

    - 预分配/预计算参数

    - 线程兼容类型与线程安全类型

  - 算法改进

  - 更好的内存表示

    - 紧凑数据结构

    - 内存布局

    - 用索引代替指针

    - 批量存储

    - 内联存储

    - 不必要的嵌套 map

    - Arena

    - 用数组代替 map

    - 用位向量代替 set

  - 减少分配

    - 避免不必要的分配

    - 调整容器大小或预留容量

    - 尽可能避免复制

    - 复用临时对象

  - 避免不必要的工作

    - 常见情况的快速路径

    - 一次性预计算昂贵信息

    - 将昂贵计算移出循环

    - 推迟昂贵计算

    - 专门化代码

    - 使用缓存避免重复工作

    - 让编译器的工作更容易

    - 降低统计信息收集成本

    - 避免在热代码路径上记录日志

  - 代码大小考量

    - 修剪常被内联的代码

    - 谨慎内联

    - 减少模板实例化

    - 减少容器操作

  - 并行化与同步

    - 利用并行性

    - 摊销锁获取成本

    - 保持临界区简短

    - 通过分片减少争用

    - SIMD 指令

    - 减少伪共享

    - 降低上下文切换频率

    - 使用带缓冲的 channel 进行流水线处理

    - 考虑无锁方法

  - Protocol Buffer 建议

  - C++ 专项建议

    - absl::flat_hash_map（以及 set）

    - absl::btree_map/absl::btree_set

    - util::bitmap::InlinedBitVector

    - absl::InlinedVector

    - gtl::vector32

    - gtl::small_map

    - gtl::small_ordered_set

    - gtl::intrusive_list

    - 限制 absl::Status 和 absl::StatusOr 的使用

  - 批量操作

  - 展示多种技术的变更列表（CL）

  - 进一步阅读

  - 建议引用

  - 致谢
