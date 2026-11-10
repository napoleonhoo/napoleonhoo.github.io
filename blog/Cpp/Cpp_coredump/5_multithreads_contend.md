# C++ Coredump：竞态问题

> 在多线程服务中，竞态问题并不少见，其中最常见的一类就是多个线程同时访问共享数据，并且至少有一个线程在写。接下来，我会介绍几个我在生产环境中遇到的例子。当然，这远远不是全部。

## Case 1: 误将 `static const` 改为 `static`

这件事情的起因是这个样子的，这是一份原始代码：

``` cpp
void function() {
    static const int kA = 1;
    static const int kB = 2;
    static const int kC = 3;
    static const std::vector<int> vec = {kA, kB, kC};
}
```

有人想修改 vec 当中的内容，于是将代码改成了这个样子：

``` cpp
void function() {
    static const int kA = 1;
    static const int kB = 2;
    static const int kC = 3;
    static std::vector<int> vec = {kA, kB, kC};
    static const int kD = 4;
    vec.push_back(kD);
}
```

即去掉了 vec 的 `const` 修饰，并向 vec 中添加了元素。但是，他没有经过全面的分析与测试，没有发现这个函数其实是多线程调用的。因此，多线程并发向 vec 写入，并且没有加任何保护，而 `push_back()` 并不是线程安全的。

当多个线程并发执行 `push_back()` 时，它们可能同时修改 `vector` 的元素数量、容量和底层存储。特别是在扩容过程中，一个线程可能正在搬移元素，另一个线程却还在访问旧的内存，最终造成数据竞争和未定义行为，线上也因此出现了 coredump。

这类问题可以使用加锁的方式来进行简单修复。这个案例中，更值得注意的并不是“忘了加锁”，而是在修改代码之前，没有先弄清楚这份数据为什么被设计成 `static const`。

> 但更核心的问题，是从函数设计、业务实现的层面找到一条更好的解决路径，不要死磕一段代码与一个函数。要改代码之前，先了解现在为什么是这样写的，然后再上手改。

## Case 2：读数据时的竞态

下面用一段简化代码描述当时的问题：

``` cpp
struct ContendTest {
    std::string buffer;
};

void function() {
    ContendTest test;
    test.buffer = "hello";
    std::thread t1([&]() {
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        test.buffer.clear();  // 语句 1
    });

    std::thread t2([&]() {
        int size = test.buffer.size();  // 语句 2
        // std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::string_view view(test.buffer.c_str(), size);  // 语句 3
    });

    t1.join();
    t2.join();
}
```

当整理出来，这个问题就很简单了。在执行顺序上，语句 “2 -> 1 -> 3”。语句 3 使用的 `size` 参数是在语句 2 计算出来的，但是语句 1 已经将这个字符串清空了。因此构建 `string_view` 的时候，字符串指针指向的内容和 `size` 是不匹配的。因此，导致了 coredump。（注：这个也不是 100% 必现的，取决于 `buffer` 指向的内存状态。）

其实，只要一个线程在修改 `std::string`，另一个线程又在没有同步的情况下读取它，就已经触发了未定义行为。程序可能 coredump，也可能暂时正常运行，还可能在更晚的位置才暴露出问题。

修复方式可以通过加锁等线程间同步方式来进行，这里就不再赘述。

## 总结

多线程引起的 Coredump 在现实生产环境中太常见了，更复杂的问题有的是，再结合内存踩踏等问题，可能会造成诸多“悬案”。

关于多线程导致的 Coredump，比较好的预防办法就是高并发、长时间的进行模拟线上的测试，甚至是通过线上灰度的方法，避免大规模影响线上。

> 人力实在不及的时候，只能通过二分查找 commit 的手段来排查问题。更令人烦躁的是偶发 Coredump，对于现实情况来说，如果这个特别“偶发”的话，已经不会去查了。