# 预定义宏及其他相关

## 1. \_\_LINE\_\_&\_\_FILE\_\_

\_\_LINE\_\_是当前代码所在的行，\_\_FILE\_\_是指代码所在的文件名。示例（C语言）：

```c
#include <stdio.h>

int main(int argc, char *argv[])
{
  printf("%d\n", __LINE__);
  printf("%s\n", __FILE__);
  return 0;
}
```

输出：

```
5
predefined_test.cc
```



## 2. \_\_TIME\_\_&\_\_DATE\_\_

\_\_TIME\_\_指系统当前时间（有：时分秒），\_\_DATE\_\_是系统当前日期（有：年月日）。示例（C语言）：

```cpp
#include <stdio.h>

int main(int argc, char *argv[])
{
  printf("%s\n", __DATE__);
  printf("%s\n", __TIME__);

  return 0;
}
```

输出：

```
Apr  4 2020
22:42:06
```

## 3.\_\_STDC\_\_

当程序是标准C程序时，_\_STDC\_\_为1。示例（C语言）：

```c
#include <stdio.h>

int main(int argc, char *argv[])
{
  printf("%s\n", __STDC__);

  return 0;
}
```

输出：

```
1
```

## 4.\_\_cplusplus

当程序是C++程序时，该宏才有定义，且为C++版本号。示例（C++语言）：

```cpp
#include <stdio.h>

int main(int argc, char *argv[]) {
  printf("%ld\n", __cplusplus);

  return 0;
}
```

当编译命令为：

```shell
g++ test.cc -o test -std=c++11
```

输出：

```
201103
```

当编译命令为：

```shell
g++ test.cc -o test -std=c++17
```

输出：

```
201703
```

## 5.\#（只能在宏中使用）

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

## 6.\#\#（只能在宏中使用）

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

```
print 1 + a(10) is 11
```

## 7.\_\_VA\_ARGS\_\_&\#\#\_\_VA\_ARGS\_\_（只能在宏中使用）

\_\_VA_ARGS\_\_和#\#\_\_VA\_ARGS\_\_都可以用在*可变参数*替换宏中。

\_\_VA_ARGS\_\_可以将省略号（...）的内容照抄下来，此时不可以使用\#\#\_\_VA\_ARGS\_\_。

```c
#define LOG1(...) printf(##__VA_ARGS__)

LOG1("pi is %d.\n", 3);
LOG1("log1\n");
```

当可变参数的个数为0时，#\#\_\_VA\_ARGS\_\_可以省略前面的逗号，即*format*后面的*逗号*。*错误*示例：

```c
#define LOG2(format, ...) printf(format, __VA_ARGS__)

LOG2("log2\n");
```

这种写法会在编译（准确说是*预编译*）时报错。

示例（C语言）：

```cpp
#include <stdio.h>

#define LOG1(...) printf(__VA_ARGS__)
#define LOG2(format, ...) printf(format, ##__VA_ARGS__)

int main(int argc, char *argv[])
{
  LOG1("pi is %d.\n", 3);
  LOG1("log1\n");
  LOG2("log2\n");
  LOG2("pi is %f.\n", 3.141592);

  return 0;
}
```

输出：

```
pi is 3.
log1
log2
pi is 3.141592.
```



