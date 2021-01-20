# §2.3 Slice

## 1 代码路径

* include/leveldb/slice.h

## 2 功能

* 字符串类，指向一个在外部存储的字符串，因此，Slice类的使用者需要保证在使用期间，外部存储的字符串不被销毁。
* 此类不对其中指向的字符串进行修改。

## 3 主要成员变量

* `const char* data_;`
* `size_t size_;`

## 4 主要成员函数

* 构造函数：可以从空、const char*、std::string中构造。
* 默认的拷贝构造和赋值。
* Accessor: `data()`，`size()`，`empty()`，`operator[]()`，`ToString()`。
* Modifier:
  * `clear()`将指针指向一个空串。
  * `remove_prefix()`指针向前移动n位。
  * `compare()`，`starts_with()`

## 5 相关函数

* `operator==()`
* `operator!=()`