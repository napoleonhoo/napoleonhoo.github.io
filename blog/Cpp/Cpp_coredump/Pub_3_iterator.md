# C++ coredump：无效的迭代器

> 迭代器是 C++11 引入的概念，它的概念本身相较于空指针、越界访问等更复杂一些，因此出现的错误更隐蔽，不易被发现。

## Case 1: 无效的迭代器

``` cpp
void function(std::unordered_map<int, int>& umap, std::vector<int>& vec) {
    auto iter = umap.find(1);
    if (iter != umap.end()) {
        std::cout << iter->second << std::endl;
    }
    // ...
    vec.emplace_back(iter->second);
}
```

这里的问题在于在`emplace_back` 时，未关注到迭代器 `iter` 是否有效。当其无效时，`emplace_back` 会 coredump。这个问题其实很常见：代码写得一长，开发者很容易忘记前面只是在某个局部逻辑中判断过迭代器是否有效。

解决方案也很简单，在 `iter == umap.end()` 时，直接 `return`。

## Case 2: for 循环中 erase (1)

``` cpp
void function(std::vector<std::unique_ptr<Item>>& vec) {
    for (int i = 0; i < vec.size();) {
        auto& item = vec[i];
        vec.erase(vec.begin() + i);
        // 访问 item
    }
}
```

大家有看出上面的问题以及原因吗？我们来做如下测试：

``` cpp
struct Item {
    int a{0};
};

void function() {
    std::vector<std::unique_ptr<Item>> vec;
    for (int i = 0; i < 3; ++i) {
        vec.emplace_back(std::make_unique<Item>());
        vec.back()->a = i + 1;
    }
    for (int i = 0; i < vec.size();) {
        auto& item = vec[i];
        std::cout << vec[i]->a << std::endl;
        vec.erase(vec.begin() + i);
        std::cout << item->a << std::endl;
    }
}
```

输出是：

```
1
2
2
3
3
Segmentation fault (core dumped)
```

这里 case 的成因其实很有意思：`auto& item = vec[i]` 绑定的是vector 槽位 i，无论这个槽位里面是什么。

上面这段代码的执行过程如下：

- 第一次循环（不coredump，但实际上已经不是预期行为了）：item 指向槽位 0，erase 把后面槽位 1 的元素搬到槽位 0，因此，`item->a = 2`。
- 第二次循环，类似第一次，`item->a = 3`。
- 第三次循环（coredump）：此时只剩 1 个元素，erase 后，`item` 指向 nullptr。

为验证上面的想法，我们可以做如下测试：

```cpp
void function2() {
    std::vector<std::unique_ptr<Item>> vec;
    vec.emplace_back(std::make_unique<Item>());
    vec[0]->a = 1;
    std::cout << vec[0]->a << std::endl;

    auto& item = vec[0];
    auto new_item = std::make_unique<Item>();
    new_item->a = 10;
    vec[0] = std::move(new_item);
    std::cout << item->a << std::endl;
}
```

它的输出是：

```
1
10
```

理解了这里的原理之后，修复可以有以下三种方法：

### 方案1: erase 前先把所有权转移

``` cpp
for (size_t i = 0; i < vec.size();) {
    std::unique_ptr<Item> item = std::move(vec[i]);
    vec.erase(vec.begin() + i);
    std::cout << item->a << std::endl
}
```

### 方案2: 提前拷贝

``` cpp
for (size_t i = 0; i < vec.size();) {
    int a = vec[i]->a;
    vec.erase(vec.begin() + i);
    std::cout << a << std::endl;
}
```

### 方案3: 使用 erase 返回的新迭代器

``` cpp
for (auto it = vec.begin(); it != vec.end(); ) {
    if (need_remove(*it)) it = vec.erase(it);
    else ++it;
}
```

## Case 3: for 循环中 erase (2)

先看这段代码，它可以正常执行预期的功能：

``` cpp
void function(std::vector<int>& vec) {
    for (int i = 0; i < vec.size();) {
        if (...) {
            vec.erase(vec.begin() + i);
            ++i;
        } else {
        }
    }
}
```

有位开发者觉得每次循环都调用 `vec.size()` 会比较耗时，于是把长度提前保存了下来：

``` cpp
void function(std::vector<int>& vec) {
    int s = vec.size();
    for (int i = 0; i < s;) {
        if (...) {
            vec.erase(vec.begin() + i);
            ++i;
        } else {
        }
    }
}
```

问题也随之出现了：`size` 保存的是进入循环前的长度，但每次 `erase()` 之后，`vec.size()` 都会减一。当 `i` 仍然小于旧的 `size`、却已经不小于当前的 `vec.size()` 时，再访问 `vec[i]` 就会越界。简单的修复方法是，在上面 if 块里面，`++i` 之后加上 `--s`。

> 循环中进行 erase 本身没有啥问题，但是因为出错率太高，导致现在很多规范都不推荐这种写法了。

## 结论

如果想要玩儿转迭代器，还是得去深入了解下标准库相关函数的底层实现逻辑。如果没有深入了解的话，建议使用一些常见的写作模式，更安全些。当然，记得判断 `iter != xx.end()`！