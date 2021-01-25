# §2.6 Env

## 1 路径

* include/leveldb/env.h
* util/env.cc

## 2 功能

* 虚基类，抽象类
* 定义了一些要在不同的平台上实现的共同的虚函数。

## 3 主要成员函数

* 文件类虚函数。*备注：以下谈到的移除（Remove）、删除（Delete）的功能是相同的，之前的版本中多用Delete相关函数，现在及以后准备使用Remove相关函数，*
  * 创建新的顺序文件（SequentialFile）、创建随机存取文件（RandomAccessFile）、创建可读文件（WritableFile）、创建可追加文件（AppendableFile）。
  * 获取文件夹的字结构（文件、文件夹）
  * 移除文件、删除文件
  * 创建文件夹、移除文件夹、删除文件夹
  * 获取文件大小、重命名文件
  * 对文件加锁、解锁
* 线程类虚函数
  * 在后台开启一个线程
  * 创建一个新线程
* 测试类虚函数：获取一个用于测试的临时的文件夹。
* 日志类虚函数：创建一个日志文件。
* 时间勒虚函数：返回现在的时间、睡眠一段时间（微秒）。

## 4 相关类

### 4.1 SequentialFile

* 功能：顺序文件的虚基类
* 主要成员函数：读（Read）、跳跃（Skip）几个字节

### 4.2 RandomAccessFile

* 功能：随机存取文件的虚基类
* 主要成员函数：读（Read）

### 4.3 WritableFile

* 功能：可以顺序写的文件
* 主要成员函数：Append、Close、Flush、Sync

### 4.4 Logger

* 功能：记录日记的虚基类
* 主要成员函数：写日志（Logv）

### 4.5 FileLock

* 功能：指明一个被上锁的文件

### 4.6 EnvWrapper

* 功能：
  * 将所有的对Env的调用委托给另一个实现。
  * 主要用于某个对Env的实现并不想完全实现一个和已经提供的Env的实现不同的版本，即只想实现一部分函数。
  * 继承自Env。
* 主要成员变量：`Env* target_;`
* 主要成员函数：实现了全部的Env的函数，使用时是用`target_`调用的，即起到了代理委托的工作。

## 5 相关函数

* 函数：`WriteStringToFile`、`ReadFileToString`
* 函数声明：将format写入info_log中。代码如下：

```cpp
// Log the specified data to *info_log if info_log is non-null.
void Log(Logger* info_log, const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((__format__(__printf__, 2, 3)))
#endif
    ;
```

其中，`__attribute__((__format__(__printf__, 2, 3)))`含义如下，参见[GCC在线文档](https://gcc.gnu.org/onlinedocs/gcc-4.7.2/gcc/Function-Attributes.html)（备注：以下的翻译没有涉及Windows操作系统及MinGW的相关信息）：

GNU C中，你可以对函数声明某些特定的事情来帮助编译器来优化函数调用和更仔细地检查你的代码。
*format(archetype, string-index, first-to-check)*

*format*特性指定了一个将`printf`、`scanf`、`strftime`、`strfmon`格式作为参数的函数，需要进行类型检查。如这样的函数声明：

```c
extern int
my_printf (void *my_object, const char *my_format, ...)
    __attribute__ ((format (printf, 2, 3)));
```

使得调用`my_printf`的函数的参数与`pirntf`格式的一致性。

参数*archetype*定义了字符串格式应该怎么解释，其应该是 `printf`、`scanf`、`strftime`、`gnu_printf`、`gnu_scanf`、`gnu_strftime`或 `strfmon`。（也可以用`__printf__`、`__scanf__`、`__strftime__`或 `__strfmon__`。）*archetype*是类似于`printf`的，需要和系统的C运行时库可接受的格式一致；前缀带有`gnu_`的，需要和GNUC库的格式一致。*string-index*指明了哪一个参数（从1开始的索引）是字符串format参数。*first-to-check*指明了第一个需要检查是否和格式一致的参数。由于非晶态的C++方法隐含有一个*this*指针，则索引应该从2开始。

备注：

* 编译时，声明`-Wformat`时，如果格式检查失败，就会产生warning。
* attribute声明只能和函数**声明**一起用，不能和定义一起。
* attribute后面的内容，前后加双下划线和完全不加下划线是一样的，即`__format__`和`format`相同。

以下是例子，

代码如下：

```c
#include <stdarg.h>
#include <stdio.h>

void log_print (const char *format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__ ((__format__ (printf, 1, 2)))
#endif
    ;

void
log_print (const char *format, ...)
{
  va_list ap;
  va_start (ap, format);
  vprintf (format, ap);
  va_end (ap);
}

int
main (int argc, char *argv[])
{
  int i = 3;
  float f = 4;
  log_print ("%d and %f \n", i);
}
```

注意第二行，这时的格式应该是错误的。当使用编译方式如下时：

```shell
gcc -o attribute_format_test attribute_format_test.cc  -Wformat
```

会产生如下的输出：

```shell
attribute_format_test.cc: In function 'int main(int, char**)':
attribute_format_test.cc:24:23: warning: format '%f' expects a matching 'double' argument [-Wformat=]
   24 |   log_print ("%d and %f \n", i);
      |                      ~^
      |                       |
      |                       double
```

