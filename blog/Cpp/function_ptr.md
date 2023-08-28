---
layout: default
---

# 函数指针

## 函数指针

##1 最简单的一个函数指针

下面：*func*是一个函数指针，指向一个输入参数为int，返回值为int的函数。

```cpp
int (*int_function_ptr)(int);
```

示例（C++）：

```cpp
#include <iostream>

int int_function(int a) { return 12; }

int main(int argc, char *argv[]) {
  int (*int_function_ptr)(int) = &int_function;
  std::cout << int_function_ptr(2) << std::endl;

  return 0;
}
```

## 2 typedef一个函数指针类型

下面：*string_function_ptr_type*是一个函数指针*类型*。

```cpp
typedef std::string (*string_function_ptr_type)(int);
```

使用示例（C++）：

```cpp
#include <iostream>
#include <string>

typedef std::string (*string_function_ptr_type)(int);

std::string string_function(int b) { return "string"; }

int main(int argc, char *argv[]) {
  string_function_ptr_type string_function_ptr = &string_function;
  std::cout << string_function_ptr(3) << std::endl;

  return 0;
}
```

