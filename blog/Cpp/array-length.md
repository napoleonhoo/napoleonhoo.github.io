---
layout: default
---

# 求数组的长度

## 1 公式

```cpp
size_t array_length = sizeof (array) / sizeof (array[0])
```

## 2 测试

```cpp
#include <iostream>

int
main (int argc, char *argv[])
{
  char char_array[10];
  int int_array[10];
  std::cout << sizeof (char_array) / sizeof (char_array[0]) << std::endl;
  std::cout << sizeof (int_array) / sizeof (int_array[0]) << std::endl;
}
```

输出：

```shell
10
10
```