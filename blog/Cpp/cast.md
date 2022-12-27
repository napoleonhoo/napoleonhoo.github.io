# C++中的cast（强制类型转换）

## 1. static_cast

《C++ Primer》中写到：

> 任何具有明确意义的类型转换，只要不包含底层const，都可以使用static_cast。
>
> 当需要把一个较大的算术类型复制给较小的类型时，static_cast非常有用。此时，强制类型转换告诉程序的读者和编译器：我们知道并且不在乎潜在的精度损失。
>
> static_cast对于编译器无法自动执行的类型转换也非常有用。

Example(C++):

```cpp
long l = 123l;
int i = static_cast<long>(l);

double d = 1.23;
void *p = &d;
double *dp = static_cast<double *>(p);
```

## 2. const_cast

《C++ Primer》中写到：

> const_cast只能改变运算对象的底层const。
>
> 如果对象本身不是一个常量，使用强制类型转换获得写权限是合法的行为。

Example(C++):

```cpp
const char *cp;
char *p = const_cast<char *>(pc);
```

## 3. reinterpret_cast

《C++ Primer》中写到：

> reinterpret_cast通常为运算对象的位模式提供较低层次上的重新解释。
>
> ……编译器不会发出任何警告或错误信息：

Example(C++):

```cpp
int *ip;
char *pc = reinterpret_cast<char *>(ip);
```

## 4. dynamic_cast

《C++ Primer》中写到：

> dynamic_cast运算符，用于将*基类*的指针或引用安全地转换成*派生类*的指针或引用。
>
> ……使用形式如下所示：
>
> dynamic_cast\<type*\>(e)
>
> dynamic_cast\<type&\>(e)
>
> dynamic_cast\<type&&\>(e)
>
> 其中，type必须是一个类类型，并且通常情况下该类型应该含有虚函数。
>
> 如果一条dynamic_cast的转换目标是指针类型并且失败了，则结果为0。如果转换目标是引用类型并且失败了，则dynamic_cast运算符将抛出一个bad_cast异常。

## 4.1 指针类型

使用示例：

```cpp
if (Derived *dp = dynamic_cast<Derived *>(bp)) {
  // bp是一个指向基类的指针。
}
```

> 我们可以对一个*空指针*执行dynamic_cast，结果是所需类型的空指针。

## 4.2 引用类型

使用示例：

```cpp
try {
  const Derived &d = dynamic_cast<const Derived&>(b);
} catch (std::bad_cast) {
  
}
```

> 因为不存在所谓的*空引用*，所以对于引用类型来说无法使用与指针类型完全相同的错误报告策略。
>
> 当对引用的类型转换失败时，程序抛出一个名为*std::bad_cast*的异常，该异常定义在*typeinfo*标准库头文件中。

