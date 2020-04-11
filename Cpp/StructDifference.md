# C/C++中struct的区别

- C++中的struct可以有成员函数，C中函数不可以有。
- C++中可以使用直接初始化，C中不可以。

示例（C语言）：

```c
struct Point
{
  int x;
  int y;
};
```

示例（C++）：

```cpp
struct Point {
  int x = 1;
  int y = 2;
};
```

如果将上面这段C++代码在C下编译，则会产生编译错误：

```
root@linuxkit-025000000001:/data0/xiaofei23/go/src/top.com/cpp_test# gcc c_struct_test.c -o c_struct_test
c_struct_test.c:5:9: error: expected ':', ',', ';', '}' or '__attribute__' before '=' token
    5 |   int x = 1;
      |         ^
```

- C中定义struct类型的变量需要使用关键字*struct*，C++中则不需要。

示例（C语言）：

```c
struct Point p;
```

示例（C++语言）：

```cpp
Point p;
```

- C的struct不能有static成员变量，C++可以有。
- C中对一个空的struct取sizeof得到的结果是0，C++中得到的结果是1。
- C++中的struct可以有存取控制（Access Modifiers）、数据隐藏（Data Hiding）等功能（此时类似于类/Class），C不可以。