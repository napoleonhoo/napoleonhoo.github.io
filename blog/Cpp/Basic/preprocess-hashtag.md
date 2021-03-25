# 宏中的#

## 1.\#（只能在宏中使用）

\#会显示调用点处作为输入参数的变量名。示例（C语言）：

```c
#include <stdio.h>

#define FUNC(x) printf("input name: " #x " is: %d\n", x)

int main(int argc, char *argv[])
{
  int i = 2;
  FUNC(i);
  
  return 0;
}
```

输出：

```
input name: i is: 2
```

## 2.\#\#（只能在宏中使用）

\#\#类似于字符串拼接时的*+*，可以用来连接**变量名**。示例（C语言）：在*FUNC*的调用点，输入参数的名字是a，*x\#\#b*被转换为：*ab*。

```c
#include <stdio.h>

#define FUNC(x) printf("print 1 + " #x "(%d) is %d\n", x, x##b)

int main(int argc, char *argv[])
{
  int a = 10;
  int ab = 11;
  FUNC(a);

  return 0;
}
```

输出：

```sh
print 1 + a(10) is 11
```
