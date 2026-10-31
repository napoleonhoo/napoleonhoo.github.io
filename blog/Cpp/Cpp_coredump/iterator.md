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

这里的问题在于在`emplace_back` 时，未关注到迭代器 `iter` 是否有效。当其无效时，`emplace_back` 会 coredump。这其实是一个很现实的 case，也很容易避免。开发者在写了一长串代码时，早已忘记了前面的代码，于是不加检查的就直接使用了 `iter`。解决方案也很简单，在 `iter == umap.end()` 时，直接 `return`。

## Case 2: for 循环中 `erase` (1)

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