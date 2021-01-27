# §2.9 Status

## 1 路径

* include/leveldb/status.h util/status.cc

## 2 功能

* Status类封装了一个操作的结果。它可能表示成功，或错误（包括错误信息）。

## 3 主要成员变量

* `const char* state_;`当status为OK时，`state_`为`nullptr`；否则，`state_`拥有以下结构：`state_[0...3]`为错误信息的长度，`state_[4]`为错误码，`state_[5...]`为具体的错误信息。
* `private: enum Code;`具体如下：

```cpp
enum Code {
  kOk = 0,
  kNotFound = 1,
  kCorruption = 2,
  kNotSupported = 3,
  kInvalidArgument = 4,
  kIOError = 5
};
```

## 4 主要成员函数

* 构造函数：初始化`state_`为`nullptr`。析构函数：将`state_`析构。
* `private: const char* CopyState(const char* state);`将输入参数`state`复制一份到输出参数。
* `private: Status(Code code, const Slice&msg, const Slice& msg);`私有的一个构造函数，只能在code不为kOk时调用，如果msg2不为空，state\_的错误信息段格式为：`msg: msg2`。
* 复制构造函数：当state\_不等于nullptr时，调用`CopyState(rhs.state_)`。
* 复制赋值函数：当state\_不等于nullptr时，将state\_析构，并调用`CopyState`。
* 移动构造函数：转移`state_`，rhs的`state_`设为`nullptr`。
* 移动赋值函数：与rhs的`state_`交换。
* `static Status OK();`返回一个默认构造的Status。
* `static Status NotFound(const Slice& msg, const Slice& msg2 = Slice());`返回`Status(kNotFound, msg, msg2);`。其它函数名为`Corruption`、`NotSupported`、`InvalidArgument`、`IOError`的和这种情况基本类似。<u>这些函数类似于正确或错误的直接构造函数。</u>
* `private: Code code() const;`返回state\_中的code段，即第4段。
* `bool ok() const;`返回state\_是否是nullptr。
* `bool IsNotFound() const;`返回`code() == kNotFound`。其它函数名为`IsCorruption`、`IsIOError`、`IsNotSupportedError`、`IsInvalidArgument`等的和这种情况类似。
* `std::string ToString() const;`转化为人方便识别的string类型。