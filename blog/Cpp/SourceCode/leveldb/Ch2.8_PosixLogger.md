# §2.8 PosixLogger

## 2.8.1 功能

* POSIX系统上对Logger的实现，不允许被继承（声明了final）。

## 2.8.2 主要成员变量

* `std::FILE* const fp_;`log文件的指针。

## 2.8.3 主要成员函数

* 构造函数：将传入的`fp`赋值给`fp_`。
* 析构函数：对`fp_`调用`std::close`关闭文件。
* `void Logv(const char* format, std::valist arguments) override;`
  * 调用C库函数`gettimeofday`、`localtime_r`获取当前的时间：`struct std::tm now_components;`。
  * 通过`std::this_thread::get_id()`获取当前的`thread_id`，并resize到长度为32（`kMaxThreadIdSize`）。
  * 分配长度为512（`kStackBufferSize`）的栈上`stack_buffer`，`dynamic_buffer_size`初始化为0。
  * 总共要经过两次循环，将数据写入文件。
  * 在第一遍循环时，`buffer`是`stack_buffer`，`buffer_size`是`kStackBufferSize`。
    * 【第一步】打印基础信息到`buffer`，主要包括年、月、日、时、分、秒、微秒、`thread_id`。
    * 【第二步】将传入的`format`和`arguments`数据写入`buffer`中。
    * 【第三步】得到写入数据的总量`buffer_offset`。
    * 【第四步】当`buffer_offset >= buffer_size - 1`（预留了一位换行符）时，`dynamic_buffer_size = buffer_offset + 2;`，直接进行下一次循环。<u>这一步实际上计算了应该进行动态分配的内存大小。</u>
    * 【第五步】在`buffer`末尾添加一个换行符。
    * 【第六步】调用`std::fwrite`、`std::fflush`将`buffer`写入`fp_`。
  * 在第二遍循环时，`buffer`是根据第一遍得到的`dynamic_buffer_size`分配的内存，`buffer_size`为`dynamic_buffer_size`。
    * 第一、二、三步同第一遍循环时类似。
    * 理论上，第四步的情况不应该发生。
    * 第五、六步同第一遍循环时类似。
    * 将动态分配的内存析构。