# C++ Coredump：排序

> 本文主要介绍几个与排序有关的 C++ 问题。这些问题，细究起来会发现它们最终还是落在无效迭代器、越界访问等常见问题上，只是从代码表面来看，问题是在排序过程中出现的，因此这里把它们归到“排序”一类。

## Case 1: 自定义比较器不符合严格弱序

先来看一段代码：

``` cpp
std::sort(vec.begin(), vec.end(), [](const SortTest1& a, const SortTest1& b) {
    return a.num <= b.num;
});
```

这里会出现什么问题呢？我们先看 `std::sort` 的函数签名，它允许第三个参数传入一个比较函数：

``` cpp
sort(first, last, comp);
```

这个函数要求遵循严格弱序（strict weak ordering），即：

> 1. 非自反性：comp(x, x) = false
> 2. 反对称性：如果 comp(x, y) = true， 那么 comp(y, x) = false
> 3. 传递性：如果 comp(x, y) = comp(y, z) = true，那么 comp(x, z) = true
> 4. 等价的传递性：如果 ! comp(x, y) & ! comp(y, x)，则认为 x、y 等价。如果 x、y 等价，y、z 等价，则 x、z 等价。

这里的比较符号使用了“<=”，当 `a.num == b.num` 时，`comp(a, b)` 和 `comp(b, a)` 都会返回 `true`；甚至将一个元素与自身比较，结果也会是 `true`。这不满足非自反性和非对称性。

所以，上面程序可能发生 coredump，也可能正常结束但排序结果不符合预期，（不是 100% 必现的。）

这里的修改逻辑比较简单，直接使用“<”即可。

## Case 2: 传给排序函数的迭代器越界

先来看一段代码：

``` cpp
void function(std::vector<SortTest1> vec, int n) {
    std::partial_sort(
        vec.begin(), vec.begin() + n, vec.end(),
        [](const SortTest1 &a, const SortTest1 &b) { return a.num < b.num; });
}
```

这里可能会导致什么问题呢？

先来看看 `partial_sort` 的函数签名：

``` cpp
partial_sort(first, middle, last, comp);
```

这里有一个基本的要求，即 `middle` 的取值区间为 `[first, last)`。这里 `middle = vec.begin() + n`。如果 `n < 0`，或者 `n > vec.size()`，那么在调用 `partial_sort` 之前，程序就已经构造出了一个越界的迭代器，行为是未定义的，可能因此发生 coredump。这里的修正方法，可以在 `partial_sort` 前加一句：

``` cpp
// 也可以再加上 < 0 保护
int n = std::min(n, vec.size());
```

## 总结

排序引发的错误较少，因为自定义迭代器的规则虽然复杂，但使用的地方相对较少些。`partial_sort` 也不是随处可见的函数。上线前配合单元测试和模拟环境测试，大部分问题都能暴露出来。
