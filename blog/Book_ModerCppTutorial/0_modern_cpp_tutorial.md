---
layout: default
titile: Modern C++ Tutorial
---

# Modern C++ Tutorial 阅读笔记

- [Modern C++ Tutorial: C++ 11/14/17/20 On the Fly](https://changkun.de/modern-cpp/zh-cn/00-preface/)
- [现代C++教程：高速上手C++ 11/14/17/20](https://changkun.de/modern-cpp/en-us/00-preface/)

*注：本文注明引用部分皆来源于此书。本文并非对此书的总结，而是我对一些我不常见到的进行摘录，对一些难点进行解释和增补。*

## 1 要点摘录

### 1.1 勇于尝试

> In conclusion, as an advocate and practioner of C++, we always maintain an open mind to accept new things, and we can promote the development of C++ faster, making this old and novel language more vibrant.

近些年来，C++的标准开始迭代升级得很猛。作为真正的C++开发者，就要勇于接受新的、好的事物。那些可以帮助代码提效、开发者提效的东西，要勇于接受、尝试、使用、推广。

### 1.2 好的编程习惯

> When wrting C++, you should also avoid using program styles such as `void*` whenever possible. When you have to use C, you should pay attention to the use `extern "C"`, seperate the C language code from the C++ code, and then unify the link.

### 1.3 `NULL`

> 传统C++把`NULL`和`0`当做一个东西。一些编译器定义`NULL`为`((void*)0)`，另一些直接定义它为`0`。。
>
> `nullptr`的类型是`nullptr_t`，它可以被隐式地转换为任意的pointer或member pointer类型。

### 1.4 `constexpr`

> array的长度必须是常量表达式。
>
> C++11提供`constexpr`来允许用户明确地将函数或对象的构造函数定义为编期的常量表达式。
>
> 从C++14开始，`constexpr`函数内部可以使用简单的语句，如局部变量、循环、分支语句。（即C++14之前不可以）

### 1.5 `if-switch`内声明临时变量

> C++17之前，`if`和`switch`内不能声明临时变量。

**<font color=blue>经测试，C++17之前的版本，在`if`或`switch`中声明临时变量并不一定会编译失败。这和GCC的版本有关。</font>**
具体见下：

|   GCC version     |   C++ std         |   编译表现    |
|   ---             |   ---             |   ---         |
|   4.8.2           |   C++11           |   Error       |
|                   |   C++14           |   N/A         |
|   8.2             |   C++11/C++14     |   Warning     |
|   10              |   C++11/C++14     |   Warning     |

### 1.6 Structured Binding

下面这种写法叫做Structured Binding：

``` cpp
auto [x, y, z] = f();
```

### 1.7 `auto`

> 之前`auto`指示的是存储类型，和`register`相对。传统C++，如果一个变量没有被声明为`register`的，那么就是`auto`的。
>
> `auto`不能用来推断array类型，如以下是错误的：
``` cpp
auto arr = new auto(10); // legal
auto auto_arr2[10] = {arr}; // illegal
```

### 1.8 `decltype(auto)`

> `decltype(auto)`主要用来推断转发函数或包的返回类型，这种函数不需求必须要提供`decltype`内的表达式。

### 1.9 可变参数模板（Variadic template）

> 允许任何数量、任何类型的模板参数。举例：
``` cpp
template<typename... Ts> class Magic;
```
> 实例化时，不加任何参数（即参数的个数为0）也是允许的。
>
> 如果不想让其成为只有0个参数的模板，可以类似如下来声明至少有一个模板参数：
``` cpp
template<typename Require, typename... Args> class Magic;
```
> 使用可变参数模板的print函数的三种方法
``` cpp
// 递归模板参数
template<typename T0>
void printf1(T0 value) {
    std::cout << value << std::endl;
}
template<typename T, typename... Ts>
void printf1(T value, Ts... args) {
    std::cout << value << std::endl;
    printf1(args...);
}

// 变量参数模板展开
template<typename T0, typename... T>
void printf2(T0 t0, T... t) {
    std::cout << t0 << std::endl;
    if constexpr (sizeof...(t) > 0) printf2(t...);
}

// 初始化列表展开
template<typename T, typename... Ts>
auto printf3(T value, Ts... args) {
    std::cout << value << std::endl;
    (void) std::initializer_list<T>{([&args] {
        std::cout << args << std::endl;
    }(), value)...};
}
```

### 1.10 折叠表达式（Fold Expression）

> 举例如下：
``` cpp
template<typename... T>
auto sum(T... t) {
    return (t + ...);
}

int main() {
    std::cout << sum(1, 2, 3, 4, 5) << std::endl;
}
```

### 1.11 `std::function`

> `std::function`是一个泛型、多态的函数wrapper，它的实例可以保存、拷贝、调用任何可以被调用的实体。
>
> 一个类型安全的实体的包装，也就是说，函数的容器。

### 1.12 右值

> 纯右值（pvalue，pure rvalue），是字面值（如`10`，`true`等）；或者是等同于字面值的计算的结果，或者类似的临时变量（如`1+2`）。非引用返回的临时变量、运算表达式产生的临时变量、 原始字面量、Lambda 表达式都属于纯右值。注意，字符串字面值是左值，类型是`const char`数组。
>
> 将亡值 (xvalue, expiring value)，是 C++11 为了引入右值引用而提出的概念（因此在传统 C++ 中， 纯右值和右值是同一个概念），也就是即将被销毁、却能够被移动的值。
>
> 右值引用的声明让这个临时值的生命周期得以延长、只要变量还活着，那么将亡值将继续存活。
> 
> 不允许非常量引用绑定到非左值。

关于此部分的一个问题，有下面的代码：
``` cpp
void increase(int& v) {
    ++v;
}

void foo() {
    double s = 1;
    increase(s);
}
```

在编译中就会报错如下：

> int_ref_cast_test.cpp:9:14: error: cannot bind non-const lvalue reference of type 'int&' to an rvalue of type 'int'
> 
>    9 |     increase(s);
> 
>      |              ^
> 
> int_ref_cast_test.cpp:3:20: note:   initializing argument 1 of 'void increase(int&)'
> 
>    3 | void increase(int& v) {
> 
>      |               ~~~~~^

由此可见不会进入执行。*备注：GCC version 10，C++ std 20。*

### 1.13 完美转发`std::forward`

> 一个声明的右值引用其实是一个左值。
>
> 引用坍缩规则：在传统 C++ 中，我们不能够对一个引用类型继续进行引用， 但 C++ 由于右值引用的出现而放宽了这一做法，从而产生了引用坍缩规则，允许我们对引用进行引用， 既能左引用，又能右引用。但是却遵循如下规则：

|   Function parameter type |   Argument parameter type |   Post-derivation function parameter type |
|   ---                     |   ---                     |   ---                                     |
|   `T&`                    |   lvalue ref              |   `T&`                                    |
|   `T&`                    |   rvalue ref              |   `T&`                                    |
|   `T&&`                   |   lvalue ref              |   `T&`                                    |
|   `T&&`                   |   rvalue ref              |   `T&&`                                   |

> 无论模板参数是什么类型的引用，当且仅当实参类型为右引用时，模板参数才能被推导为右引用类型。
>
> 完美转发，就是为了让我们在传递参数的时候， 保持原来的参数类型（左引用保持左引用，右引用保持右引用）。
>
> 无论传递参数为左值还是右值，普通传参都会将参数作为左值进行转发；`std::move`总会接受到一个左值，输出右值引用。
>
> 唯独`std::forward`即没有造成任何多余的拷贝，同时完美转发（传递）了函数的实参给了内部调用的其他函数。
>
> `std::forward`和`std::move`一样，没有做任何事情，`std::move`单纯的将左值转化为右值，`std::forward`也只是单纯的将参数做了一个类型的转换，从现象上来看，`std::forward<T>(v)`和`static_cast<T&&>(v)`是完全一样的。
>
> 在使用循环语句的过程中，`auto&&`是最安全的方式。因为当`auto`被推导为不同的左右引用时，与`&&`的坍缩组合是完美转发。

### 1.14 无序容器

`std::map & std::set`是有order的，但是这个order是按key排序的order，而不是插入的order。

### 1.15 `tuple`

> `std::tie`：元组拆包。用法如下：
``` cpp
std::tie(gpa, grad, name) = get_student(1);
```
> `std::get`除了使用常量获取元组对象外，C++14 增加了使用类型来获取元组中的对象：
``` cpp
std::cout << std::get<std::string>(t) << std::endl;
```
> std::get<>依赖一个编译期的常量，所以下面的方式是不合法的：
``` cpp
int index = 1;
std::get<index>(t);
```
> 那么要怎么处理？使用`tuple_index`，其基于`std::variant`。举例如下：
``` cpp
int i = 1;
std::cout << tuple_index(t, i) << std::endl;
```
> 合并元组，使用`tuple_cat`，举例如下：
``` cpp
auto t3 = std::tuple_cat(t1, t2);
```
> 元组长度与遍历方法，举例如下：
``` cpp
template <typename T>
auto tuple_len(T &tpl) {
    return std::tuple_size<T>::value;
}
// 迭代
for(int i = 0; i != tuple_len(new_tuple); ++i)
    // 运行期索引
    std::cout << tuple_index(new_tuple, i) << std::endl;
```

### 1.16 `std::shared_ptr`

> `std::make_shared`就能够用来消除显式的使用`new`，所以`std::make_shared`会分配创建传入参数中的对象， 并返回这个对象类型的`std::shared_ptr`指针。

### 1.17 `std::unique_ptr`

`make_unique`已在C++14实现。

### 1.18 `std::weak_ptr`

> `std::weak_ptr`是一种弱引用（相比较而言`std::shared_ptr`就是一种强引用）。弱引用不会引起引用计数增加。
感觉`weak_ptr`不是很常用啊。

### 1.18 正则表达式

> 在后台服务的开发中，对 URL 资源链接进行判断时， 使用正则表达式也是工业界最为成熟的普遍做法。

### 1.19 互斥量与临界区

> `std::unique_lock`更加灵活， `std::unique_lock`的对象会以独占所有权（没有其他的`unique_lock`对象同时拥有某个`mutex`对象的所有权）的方式管理`mutex`对象上的上锁和解锁的操作。所以在并发编程中，推荐使用`std::unique_lock`。
>
> `std::lock_guard`不能显式的调用`lock`和`unlock，`而`std::unique_lock`可以在声明后的任意位置调用，可以缩小锁的作用范围，提供更高的并发度。
> 
> 如果你用到了条件变量`std::condition_variable::wait`则必须使用`std::unique_lock`作为参数。

### 1.20 `future`

> 试想，如果我们的主线程A希望新开辟一个线程B去执行某个我们预期的任务，并返回我一个结果。而这时候，线程A可能正在忙其他的事情，无暇顾及B的结果，所以我们会很自然的希望能够在某个特定的时间获得线程B的结果。
>
> 自然地，我们很容易能够想象到把它作为一种简单的线程同步手段，即屏障（barrier）。
>
> 条件变量std::condition_variable是为了解决死锁而生，当互斥操作不够用而引入的。比如，线程可能需要等待某个条件为真才能继续执行，而一个忙等待循环中可能会导致所有其他线程都无法进入临界区使得条件为真时，就会发生死锁。

### 1.21 原子操作与内存序

TODO

### 1.22 `noexecpt`

> C++11之前，几乎没有人去使用在函数名后书写异常声明表达式，从C++11开始，这套机制被弃用。
>
> **使用`noexcept`修饰过的函数如果抛出异常，编译器会使用`std::terminate()`来立即终止程序运行。**
>
> `noexcept`还能够做操作符，用于操作一个表达式，当表达式无异常时，返回`true`，否则返回`false`。
>
> `noexcept`修饰完一个函数之后能够起到封锁异常扩散的功效，如果内部产生异常，外部也不会触发。

1. 只要不写`noexecpt`就可能throw。

### 1.23 原始字符串字面值

> 一个包含 HTML 本体的字符串需要添加大量的转义符。
>
> C++11提供了原始字符串字面量的写法，可以在一个字符串前方使用`R`来修饰这个字符串，同时，将原始字符串使用括号包裹。
``` cpp
    std::string str = R"(C:\File\To\Path)";
```

### 1.24 自定义字面值

> C++11引进了自定义字面量的能力，通过重载双引号后缀运算符实现。

### 1.25 内存对齐

> C++11还引入了`alignas`来重新修饰某个结构的对齐方式。
>
> 其中`std::max_align_t`要求每个标量类型的对齐方式严格一样，因此它几乎是最大标量没有差异，进而大部分平台上得到的结果为`long double`。

## 2 疑问与增补内容

参见[疑问与增补内容](./1_modern_cpp_tutorial_supplement.html)

## 3 英语学习

- advent: 出现，到来
- infuse: 注入，使具有（某特性）
- vitality: 活力，生命力，热情
- vibrant: 生机勃勃的，充满生机的
- majeure: 不可抗力
- circumvent: 规避，绕过
- synonymous: 同义的，等同于……的
- asymmetry: 不对称
- heinous: 令人发指的，道德败坏的，极恶毒的
- ontology: 本体论