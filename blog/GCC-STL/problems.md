# 问题集锦

1. 为啥`std::optional`及很多类存在着如下两种构造形式呢？

``` cpp
// ctor1
template <class... Args> 
constexpr explicit optional(in_place_t, Args&&... args); 

// ctor2
template <class U, class... Args>
constexpr explicit optional(in_place_t, initializer_list<U> il, Args&&... args);
```

万能引用不是可以指代任何东西（包括初始化列表）吗？为啥需要这两种？

参考解答：https://stackoverflow.com/questions/27263365/optional-constructor-with-initializer-list

存在这两种的原因是：

假设有这么一个类：

```
struct foo
{
    foo(std::initializer_list<int>) {}
};
```

如果没有ctor2，那么下面这种初始化形式是编译不过的：

``` cpp
std::optional<foo> o(std::in_place, {1, 2, 3});
```

因为花括号初始化列表是没有类型的，所以类型推导过程会失败。

> The above fails because a *braced-init-list* has no type, so template argument deduction fails. 

所以，要写成这样才能成功：

``` cpp
auto il = {1, 2, 3};
std::optional<foo> o(std::in_place, il);
```

但是有了ctor2之后，之前那种比较方便、简单的写法就能通过编译了。



2. SMF是什么意思？

Special Member Function. 

>  default constructors, copy constructors, move constructors, copy assignment operators, move assignment operators, and prospective destructors.

参考见：https://quuxplusone.github.io/blog/2019/08/02/the-tough-guide-to-cpp-acronyms/#smf