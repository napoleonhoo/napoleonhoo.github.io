---
layout: default
---

# sizeof & size_t & %zu

## 1. sizeof的两种用法

sizeof是最常用的C语言表达式之一了，然而今天却仍然让我大吃一惊了。最近我才了解到，sizeof有两种。

```C
sizeof( type ) // 第一种
sizeof expression // 第二种
```

- 最常用的就是第一种了，第二种在很多时候是和第一种通用的。
- 但是，当对于一个固定的类型，如：int/float，之类的时候，只能使用第一种。

```C
sizeof(flaot); // 正确
sizeof float; // 错误
```

## 2. size_t & %zu

size_t是sizeof的返回值类型，也是一种比较常见的数据类型了。如果用printf来打印一个size_t的值，有一个特殊的format，即"%zu"。这里的z是一个专门为size_t类型设计的标记，是gcc的扩展用法。以下的文章摘录自[Linux Man Pages printf.3](http://man7.org/linux/man-pages/man3/printf.3.html)

```
       z      A following integer conversion corresponds to a size_t or
              ssize_t argument, or a following n conversion corresponds to a
              pointer to a size_t argument.
```

举例：

```c
#include <stdio.h>
#include <stdlib.h>

int main()
{
	printf("%zu\n", sizeof(float));
}
```

虽然在一般情况下，使用"%lu"或者其他的标记也可以代替，但总是"%zu"更为专业、稳妥一些。