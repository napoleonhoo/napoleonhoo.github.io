---
layout: default
titile: Modern C++ Tutorial
---

# Modern C++ Tutorial 疑问与增补

## 1 实现Fibonacci多种方式之比较

Fibonacci的计算大家都熟悉了，在举例`constexpr`的时候，书本*Modern C++ Tutorial*用其做了例子。于是想自己手动试一下，通过一下三种方法的实现，做了比较：

三种方法：
1. 模板实现
2. `constexpr`函数实现
3. 普通函数实现

实现的代码可参考[template / constexpr function / normal function ways to calculate fibonacci
](https://gist.github.com/napoleonhoo/1fb042495546868a0adbe95add1bb789)。

编译选项：GCC10，std=C++20，默认优化。

经测试，在`n <= 35`的时候，`template`、`constexpr` function速度相当，约为0s。当`n > 35`时，`constexpr` function速度退化到和一般实现形式差不多。`template`方式始终很快。测试结果摘录如下：

```
calucate fibonacci of 35
9227465
template style: 0.005 s
9227465
constexpr style: 0.003 s
9227465
normal style: 109.355 s

calucate fibonacci of 36
14930352
template style: 0.005 s
14930352
constexpr style: 177.132 s
14930352
normal style: 176.874 s
```

那么，为什么`constexpr` function会有如此不同的表现呢？原因是：`constexpr`关键字只是一种提示信息，告诉编译器可以按照编译期常量来处理，即不是必须要完成编译期的求值计算。（机制有点儿类似`inline`）

## 2 `constexpr`释义

参看[constexpr 说明符 - cppreference](https://zh.cppreference.com/w/cpp/language/constexpr)

> `constexpr`说明符声明编译时可以对函数或变量求值。这些变量和函数（给定了合适的函数实参的情况下）即可用于需要编译期常量表达式的地方。

`constexpr`关键字只是一种提示信息，告诉编译器可以按照编译期常量来处理，即不是必须要完成编译期的求值计算。（机制有点儿类似`inline`）

## 3 `decltype`、`auto`、`decltype(auto)`

TODO

## 4 nested dependency type

TODO

## 5 `decltype(auto)`的使用场景

参看：[What are some uses of decltype(auto)? - stackoverflow](https://stackoverflow.com/questions/24109737/what-are-some-uses-of-decltypeauto)

简单总结高赞回答如下：

1. Return type forwarding in generic code（泛型代码返回类型转发）
2. Delaying return type deduction in recursive templates（递归模板中延迟返回类型推导）

## 6 `extern`模板

### 模板类

参看：[类模板 - cppreference](https://zh.cppreference.com/w/cpp/language/class_template#Class_template_instantiation)

> 显式实例化声明（extern 模板）跳过隐式的实例化步骤：本来会导致隐式实例化的代码会以别处提供的显式实例化定义（如果这种实例化不存在，则导致连接错误）取代。这个机制可被用于减少编译时间：通过在除了一个文件以外的所有源文件中明确声明模板实例化，而在剩下的那个文件中明确定义它。

### 模板函数

参看：[函数模板 - cppreference](https://zh.cppreference.com/w/cpp/language/function_template#Function_template_instantiation)

> 显式实例化声明（extern 模板）阻止隐式实例化：本来会导致隐式实例化的代码必须改为使用已在程序的别处所提供的显式实例化。

## 7 折叠表达式（Fold Expression）

C++17新出现的高级玩意儿。参看[折叠表达式 - cppreference](https://zh.cppreference.com/w/cpp/language/fold)

## 8 non-type template parameter deduction

非类型模板参数推导。简单来讲，一般的模板参数是：`template<typename T> class x;`，这个`T`就是类型模板参数。另一种模板参数如：`template<int SIZE> class y;`，这个`SIZE`就是非类型模板参数，简单来说，它的意思不是一个“类型”。

## 9 `std::underlying_type`

参看：[std::underlying_type](https://zh.cppreference.com/w/cpp/types/underlying_type)

若`T`是完整枚举类型，则提供指名`T`底层类型的成员`typedef type`。

## 10 lambda函数是使用的新的栈吗？还是类似于`inline`？

经测试，lambda函数是使用一个新的栈，类似函数调用。

## 11 类型安全（type-safe）？

参考问题[What is Type-safe? - stackoverflow](https://stackoverflow.com/questions/260626/what-is-type-safe)

以下摘抄自高赞答案：

> Type safety means that the compiler will validate types while compiling, and throw an error if you try to assign the wrong type to a variable.

简单总结下，类型安全就是编译器会为你的代码检查下类型是否正确，如果对变量赋值不同类型的数据就会报错。比如C/C++中的`printf`、`memcpy`函数都不是类型安全的。

## 12 返回值优化（Return Value Optimisation, RVO）问题

这里补充一下书中”移动语义“一节，因为RVO的存在，当函数返回的是临时对象时，不需要多次拷贝。

## 13 `vec.push_back(std::move(str))`是不需要进行内存拷贝的吗？

其实也需要拷贝吧。`vector`是个容器，在这种情况下，它里面的元素都是`string`。不拷贝的话，如何把`string`放进去呢？

## 14 `std::array`的内存分配？

`std::array`的大小是固定的，不涉及动态调整内存大小。没有对单个元素的删除操作，也不能增加元素。

## 15 结合可变参数模板、模板继承、递归等手段实现`std::tuple`？

官方说法叫**recursive inheritance template，即递归继承模板**。

这是我的一个[实现](https://gist.github.com/napoleonhoo/485aa2ceda8efbf87ade7d79d67982a3)，写了一下基本功能，调了一下午，略难。

## 16 为啥`conditional_variable`需要`unique_lock`？为了使用`lock`和`unlock`？

不是为了使用`lock`和`unlock`。那么为啥需要使用`unique_lock`呢？

在我看来，原因主要是，`condition_variable`的实现主要是使用了`pthread_cond`，而如`pthread_cond_wait_*`等函数需要一个`mutex`作为输入变量。而在接口设计上，只有`unique_lock`可以获取它的`mutex`，而`lock_guard`、`scoped_lock`等不能。

## 17 coroutine

## 18 为啥说`conditional_variable`解决死锁呢？

能。但不是只有`conditional_variable`能。

## 19 默认什么memory order？

Reference: [std::atomic - cppreference](https://en.cppreference.com/w/cpp/atomic/atomic)

`std::memory_order_seq_cst`。

## 20 自定义字面量（Custom Literal）使用方法？

更准确的英文叫法为User-defined literals。

Reference: [User-defined literals - cppreference](https://en.cppreference.com/w/cpp/language/user_literal)

## 21 看*Ulrich Drepper, What Every Programmers Should Know about Memory*
