# §Ch2.1 Arena

## 1 代码路径

* util/arena.h
* util/arena.cc

## 2 主要功能

* 负责内存分配和释放。

## 3 主要成员变量

* `char *alloc_ptr_;`标记下一个可以被分配的位置的指针。
* `size_t alloc_bytes_remaining_;`本次分配的还剩下的内存空间。
* `std::vector<char*> blocks_;`此类所有的被分配的空间的vector。
* `std::atomic<size_t> memory_usage_; `此类所有分配的空间总数。其load、fetch_add的操作的内存顺序为`std::memory_order_relaxed`。

## 4 相关变量

* `static const int kBlockSize = 4096;`单位：byte，每个block的大小。

## 5 主要成员函数

* 析构函数：将blocks_中所有此类所分配的空间全部delete。
* `inline char* Arena::Allocate(size_t bytes);`
  * 分配`bytes`个byte的空间，并返回其起始地址。这个函数的`bytes`必须要大于0。
  * 如果`bytes <= alloc_bytes_remainning_`，在当前的内存空间内分配`bytes`个单位，并更新`alloc_ptr_`、`alloc_bytes_remaining_`。
  * 如果`bytes > alloc_bytes_remaining_`，调用`AllocateFallback(bytes)`。

* `char* Arena::AllocateAligned(size_t bytes);`

  * align赋值：当指针的size大于8时，align为指针大小；否则，为8。另外，align必须为2的指数。判断方式：

    ```cpp
    static_assert((align & (align -1)) == 0, "Pointer size should be a power of 2");
    ```

  * 计算slop，应是当前`alloc_ptr_`离align的举例，代码如下：

    ```cpp
    size_t current_mod = reinterpret_cast<uintptr_t>(alloc_ptr) & (align - 1);
    size_t slop = (current_mod == 0 ? 0 : align - current_mod);
    ```

  * 当`needed <= alloc_bytes_remaining_`时，返回`result`为`alloc_ptr_`按align对齐的结果，即`result = alloc_ptr_ + slop`，更新`alloc_ptr_`、`alloc_bytes_remaining_`。

## 6 使用举例

```cpp
// from skiplist.h
template <typename Key, class Comparator>
typename SkipList<Key, Comparator>::Node* SkipList<Key, Comparator>::NewNode(
    const Key& key, int height) {
  char* const node_memory = arena_->AllocateAligned(
      sizeof(Node) + sizeof(std::atomic<Node*>) * (height - 1));
  return new (node_memory) Node(key);
}
```

