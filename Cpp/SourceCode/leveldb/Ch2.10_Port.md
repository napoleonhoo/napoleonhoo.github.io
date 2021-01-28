# §2.10 Port

## 1 路径

* port/port\_stdcxx.h
* include/port/port\_config.h（具体见第4条）

## 2 功能



## 3 类

### 3.1 Mutex

对`std::mutex`的封装，加入了clang的一些线程安全的attribute，另外，声明了`friend class CondVar;`。

主要函数对应：`Lock()`-->`lock()`，`Unlock()`-->`unlock()`。

### 3.2 CondVar

对`std::condition_variable`的封装，构造时传入上面提到的`Mutex`类对象，调用这个类的函数，自动完成和互斥量相关的信号量。

主要函数对应：`Wait()`-->`wait()`，`Signal()`-->`notify_one()`，`SignalAll()`-->`notify_all()`。

## 4 函数

* Snappy相关函数签名。

```cpp
inline bool Snappy_Compress(const char* input, size_t length,
                            std::string* output);
inline bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                         size_t* result);
inline bool Snappy_Uncompress(const char* input, size_t length, char* output);
```

* CRC32相关函数

```cpp
inline uint32_t AcceleratedCRC32C(uint32_t crc, const char* buf, size_t size);
```

## 5 文件说明

在利用CMake编译时，CMake程序会将检测一些库、符号等是否存在，如：

```cmake
check_cxx_symbol_exists(fdatasync "unistd.h" HAVE_FDATASYNC)
```

这一句检测头文件“unistd.h”中是否存在符号fdatasync，并且将结果缓存给变量HAVE_FDATASYNC。而这一段：

```cmake
configure_file(
  "port/port_config.h.in"
  "${PROJECT_BINARY_DIR}/${LEVELDB_PORT_CONFIG_DIR}/port_config.h"
)
```

会利用已经被缓存的变量，根据port/prot\_config.h.in的内容，写入include/port/port\_config.h。port/prot\_config.h.in中的代码如：

```cpp
// Define to 1 if you have a definition for fdatasync() in <unistd.h>.
#if !defined(HAVE_FDATASYNC)
#cmakedefine01 HAVE_FDATASYNC
#endif  // !defined(HAVE_FDATASYNC)
```

会被转换到include/port/port\_config.h中：

```cpp
// Define to 1 if you have a definition for fdatasync() in <unistd.h>.
#if !defined(HAVE_FDATASYNC)
#define HAVE_FDATASYNC 1
#endif  // !defined(HAVE_FDATASYNC)
```

