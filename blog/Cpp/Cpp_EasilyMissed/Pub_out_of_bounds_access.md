# C++ coredump：几个越界访问的 case

> 相比空指针，越界访问导致的 core dump 可能会少一些，因为大部分越界问题都可以通过边界检查提前避免。不过，真实的线上问题往往没有这么简单：一段数据的写入方和使用方可能是不同的开发者，而使用方又过于信任上游，没有对数据做必要的校验。一旦上游调整了数据格式，原本正常运行的代码就可能出问题。

## Case 1 数组越界访问

``` cpp
void function(std::string& s) {
	std::vector<std::string> tmp_str_list;

	// StrSplitList 将传入的字符串 s，按照“，”分割
	// 并将分割后的每一个元素放入 tmp_str_list
	StrSplitList(s.c_str(), ",", tmp_str_list);

	float f = atof(tmp_str_list[0].c_str());
	int i = atoi(tmp_str_list[1].c_str())
}
```

这里造成 coredump 的原因很明显，就是在读取 `tmp_str_list[0]` 和 `tmp_str_list[1]` 的时候，没有检查 `tmp_str_list` 的 size，但是这里 case 的成因很有意思：

`s` 是从某个配置文件中读入的一行，比如“123.5,23”（这里具体业务背景不再详述了）。而有一天，某个人决定对这里的功能进行扩展，在配置文件中支持如“123.5“这样的形式，却没有在这里进行兼容。此时，分割结果中只有一个元素，再去访问 `tmp_str_list[1]`，就发生越界了，进而导致了 coredump。

需要注意的是，`std::vector::operator[]` 不会检查下标是否合法。下标越界属于未定义行为：程序可能直接崩溃，也可能读到一块无效数据后继续运行。后者往往更加麻烦，因为问题不一定会在第一时间暴露出来。

另外，补充一点，如果使用 `at()` 的话，当下标越界时，`at()` 会抛出 `std::out_of_range` 异常，可以让问题更快暴露出来。不过，`at()` 不能代替正常的输入校验；如果异常没有被捕获，程序最终还是会终止。

## Case 2 `std::string` 越界访问

``` cpp
void function(std::string& s) {
    auto start_pos = s.find("://") + 3;
    std::string tmp = s.substr(start_pos);
    std::cout << tmp << std::endl;
}
```

这段代码想取出 `"://"` 后面的内容。看起来没有什么问题，但它漏掉了一种情况：如果字符串中根本没有 `"://"`，`find()` 会返回什么？答案是 `std::string::npos`。

`npos` 通常是无符号整数类型能够表示的最大值。直接对它执行 `+ 3`，会发生无符号整数回绕，最后得到一个看似正常、实际上无效的位置。接着再把这个位置传给 `substr()`，就可能看到下面的报错

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  basic_string::substr: __pos (which is 2) > this->size() (which is 1)
Aborted (core dumped)
```

这里并不是 `substr()` 直接导致了 coredump。更准确地说，是 `substr()` 发现起始位置超过字符串长度后，抛出了 `std::out_of_range` 异常；异常没有被捕获，于是运行时调用 `std::terminate()` 终止程序。在开启 coredump 的环境中，最终就可能生成 core 文件。

## 总结

越界访问本身并不难避免，真正容易被忽略的是输入数据的变化。

代码上线时，上游数据可能完全符合约定；但随着业务迭代，配置格式、字段数量甚至数据来源都可能发生变化。如果消费方没有做好边界检查，一个原本很小的改动，就可能变成一次线上 coredump。

所以，这里最重要的一个 tips：**不要过度信任上游。**