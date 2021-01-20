# §2.2 SkipList

##  1 主要路径

* db/skiplist.h

## 2 功能

* MemTable中最主要的数据结构，存储的数据实际在此。

## 3 模板参数

* `Key`，SkipList存储的key（数据）的类型
* `Comparator`，Key的比较器类型，用来比较key的大小

## 4 主要成员变量

* `std::atomic<int> max_height_;`
* `enum {  kMaxHeight = 12; }`
* `Arena* const arena_;`主要用于内存分配。
* `Random rnd;`主要用于产生随机数。

## 5 主要成员函数

* 不允许拷贝构造和赋值。

* `int RandomHeight();`重要代码如下（Line244~249）：

  ```cpp
  static const unsigned int kBranching = 4;
  int height = 1;
  while (height < kMaxHeight && ((rnd_.Next() % kBranching) == 0)) {
    height++;
  }
  ```

  分析：修改kMaxHeight ，在数值变小时，性能上有明显下降，但当数值增大时，甚至增大到10000时，和默认的kMaxHeight =12相比仍旧无明显差异，内存使用上也是如此。关键在于while循环中的判定条件：`height < kMaxHeight && ((rnd_.Next() % kBranching) == 0)`。除了kMaxHeight 判定外，`(rnd_.Next() % kBranching) == 0)`判定使得上层节点的数量约为下层的1/4。那么，当设定kMaxHeight=12时，根节点为1时，约可均匀容纳Key的数量为$4^{11}=4194304$（约为400W）。因此，当单独增大kMaxHeight时，并不会使得SkipList的层级提升。kMaxHeight=12为经验值，在百万数据规模时，尤为适用。

## 6 内部类

* `class Iterator;`
* `struct Node;`
  * 主要成员变量：
    * 原子`Node*`类型数组，`next_`，长度为1，相当于通常实现中的*forward*参数，实际使用时，其长度为node的level。
  * 主要成员函数：
    * `Next/SetNext`：`next_`的 load（`std::memory_order_acquire`），store（`std::memory_order_release`）。
    * `NoBarrier_Next/SetNext`，`next_`的load/store（`std::memory_order_relaxed`）。

## 7 使用方法

* 关于线程安全：写需要外部的同步（一般是mutex）。读需要保证SkipList不会被销毁。除此之外，读不需要内部的锁或同步。
* 不变量：
  * 直到SkipList被销毁时，node不会被销毁。
  * 在node被链入list之后，除了prev和next指针，node的内容是不可变的。只有Insert函数可以修改list。

