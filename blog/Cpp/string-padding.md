---
layout: default
---

# 使用string的基本操作实现padding

## 1 来源

* 项目：leveldb
* 文件：table/format.cc
* 相关代码：

```cpp
void Footer::EncodeTo(std::string* dst) const {
  const size_t original_size = dst->size();
  metaindex_handle_.EncodeTo(dst);
  index_handle_.EncodeTo(dst);
  dst->resize(2 * BlockHandle::kMaxEncodedLength);  // Padding
  PutFixed32(dst, static_cast<uint32_t>(kTableMagicNumber & 0xffffffffu));
  PutFixed32(dst, static_cast<uint32_t>(kTableMagicNumber >> 32));
  assert(dst->size() == original_size + kEncodedLength);
  (void)original_size;  // Disable unused variable warning.
}
```

## 2 测试代码

```cpp
#include <cstdio>
#include <iostream>
#include <string>

int
main (int argcc, char *argv[])
{
  std::string str = "12345678";
  str.resize (10);
  str.append ("12345");
  std::cout << str << std::endl;
  std::cout << str.size () << std::endl;
  for (int i = 0; i < str.size (); ++i)
    {
      std::printf ("%c(%d)\t", str[i], int(str[i]));
    }
  std::cout << std::endl;
}
```

输出：

```shell
1234567812345
15
1(49)	2(50)	3(51)	4(52)	5(53)	6(54)	7(55)	8(56)	(0)	(0)	1(49)	2(50)	3(51)	4(52)	5(53)
```

## 3 原理

resize函数，来源[cppreference_string_resize](https://en.cppreference.com/w/cpp/string/basic_string/resize)：

> If the current size is less than `count`, additional characters are appended.
>
> ... initializes new characters to CharT() ...

当使用resize对string对象调整了一个比当前size更大的size时，会在多余的空间内追加几个空字符（ASCII：0）。再append时，会追加到当前的size后。

