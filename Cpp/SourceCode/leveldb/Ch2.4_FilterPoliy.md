# § 2.4 FilterPolicy

## 1 代码路径

* include/leveldb/filter_policy.h

## 2 功能

* 虚基类
* 用于`DB::Get()`的调用时，减少磁盘的访问次数。

## 3 重要成员函数

* `Name()`返回过滤器的名字。
* `virtual void CreateFilter(const Slice* keys, int n, std::string* dst) const = 0;`
* `virtual bool KeyMayMatch(const Slice& key, const Slice& filter) const = 0;`

## 4 重要相关函数

* `const FilterPolicy* NewBloomFilterPolicy(int bits_per_key);`返回一个Bloom Filter，这个函数在这里只是声明。

