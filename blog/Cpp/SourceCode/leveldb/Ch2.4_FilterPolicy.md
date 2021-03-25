# § 2.4 FilterPolicy

## 1 代码路径

* include/leveldb/filter_policy.h

## 2 功能

* 虚基类
* 用于`DB::Get()`的调用时，减少磁盘的访问次数。

## 3 重要成员函数

* `Name()`返回过滤器的名字。
* `virtual void CreateFilter(const Slice* keys, int n, std::string* dst) const = 0;`这个函数的输入是一组key，参数`keys`为指向这一组的指针，相当于数组，`n`为数组大小，函数会将过滤器`append`到`dst`上。
* `virtual bool KeyMayMatch(const Slice& key, const Slice& filter) const = 0;`这个函数的输入是一个`key`，和一个来自于上面`CreateFilter`生成的`filter`，返回值为能否找到这个`key`，即true or false。

## 4 重要相关函数

* `const FilterPolicy* NewBloomFilterPolicy(int bits_per_key);`返回一个Bloom Filter，这个函数在这里只是声明。

