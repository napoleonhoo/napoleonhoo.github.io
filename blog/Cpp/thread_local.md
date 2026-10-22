# C++ 易错点：`thread_local`

同样定义在函数内部，`int a` 和 `thread_local int a` 看起来只差一个关键字，生命周期却完全不同：前者属于一次函数调用，后者属于当前线程。这个区别决定了变量能否跨调用保留，以及多个线程之间是否共享状态。本文通过一个简短示例说明二者的核心差异。

```cpp
void func()
{
    int a = 0;
    thread_local int tls_a = 0;
}
```

两者的实例数量和生命周期如下：

- `int a`：每次函数调用都会创建一个新实例。
- `thread_local int tls_a`：每个线程拥有一个实例，同一线程内的多次函数调用会访问同一个实例。

## 示例代码

```cpp
#include <cstdio>
#include <thread>

void test(const char* name)
{
    int a = 0;
    thread_local int tls_a = 0;

    std::printf("%s: a=%d, tls_a=%d\n",
                name, ++a, ++tls_a);
}

void worker(const char* name)
{
    test(name);
    test(name);
    test(name);
}

int main()
{
    std::thread t1(worker, "thread-1");
    t1.join();

    std::thread t2(worker, "thread-2");
    t2.join();

    return 0;
}
```

编译命令：

```bash
clang++ -std=c++11 -pthread demo.cpp -o demo
```

一种可能的输出如下：

```text
thread-1: a=1, tls_a=1
thread-1: a=1, tls_a=2
thread-1: a=1, tls_a=3
thread-2: a=1, tls_a=1
thread-2: a=1, tls_a=2
thread-2: a=1, tls_a=3
```

## 为什么结果不同

普通局部变量 `a` 具有自动存储期。每次进入 `test()` 时，它都会被重新创建并初始化；函数返回后，该实例随即销毁：

```text
第 1 次调用：0 → 1 → 销毁
第 2 次调用：0 → 1 → 销毁
```

`tls_a` 具有线程存储期。每个线程拥有一个独立实例，该实例在线程内只初始化一次，并持续存在到线程结束：

```text
thread-1：0 → 1 → 2 → 3
thread-2：0 → 1 → 2 → 3
```

核心区别可以概括为：

| 声明 | 实例数量 | 生命周期 |
|---|---|---|
| `int a` | 每次函数调用一份 | 当前函数调用期间 |
| `thread_local int a` | 每个线程一份 | 当前线程运行期间 |

`thread_local` 适合保存每线程缓存、计数器、日志上下文或随机数生成器状态。需要注意的是，在线程池中，同一个工作线程可能连续处理多个任务，因此线程局部状态也会跨任务保留。

## 附：C 语言中的 `thread_local`

C11 引入了 `_Thread_local` 存储类说明符，用于定义具有线程存储期的对象。每个线程拥有一个独立实例，该实例在线程开始时完成初始化，并持续存在到线程结束。

在文件作用域中，`_Thread_local` 可以单独使用：

```c
_Thread_local int counter = 0;
```

此时，程序中的每个线程都有自己的 `counter`。一个线程修改该变量，不会影响其他线程中的实例。

在函数或其他块作用域中，`_Thread_local` 不能单独使用，必须与 `static` 或 `extern` 组合：

```c
void func(void)
{
    static _Thread_local int counter = 0;
    ++counter;
}
```

这里的 `counter` 仍然是每个线程一份。同一线程多次调用 `func()` 时，其值会持续累加；不同线程之间则相互隔离。

在支持 `<threads.h>` 的 C11/C17 实现中，也可以使用该头文件提供的 `thread_local` 宏。C23 进一步将 `thread_local` 定义为关键字，并将 `_Thread_local` 标记为弃用：

```c
void func(void)
{
    static thread_local int counter = 0;
}
```

需要注意，标准 C 即使使用 C23，也不允许在块作用域中单独声明：

```c
thread_local int counter = 0;  // 标准 C 中不合法
```

而下面的写法在 C++11 及以后是合法的：

```cpp
thread_local int counter = 0;
```

因此，若要直接比较函数内部的 `int a` 和 `thread_local int a`，应以 C++ 语义为准；在标准 C 中，则需要遵守块作用域必须搭配 `static` 或 `extern` 的规则。
