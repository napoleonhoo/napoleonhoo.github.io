# §2.13 Comparator

## 1 路径

* util/comparator.cc
* include/leveldb/compartor.h

## 2 功能

* 这是用来减少如index块等的内部数据结构的空间占用。

## 3 虚基类：Comparator

* FindShortestSeperator，将start修改为在区间`[start, limit)`中（按字典序排序）。

```cpp
// If *start < limit, changes *start to a short string in [start,limit).
// Simple comparator implementations may return with *start unchanged,
// i.e., an implementation of this method that does nothing is correct.
virtual void FindShortestSeparator(std::string* start,
                                   const Slice& limit) const = 0;
```

* FindShortSuccessor，将key修改为`>= key`（按字典序排序）。

```cpp
// Changes *key to a short string >= *key.
// Simple comparator implementations may return with *key unchanged,
// i.e., an implementation of this method that does nothing is correct.
virtual void FindShortSuccessor(std::string* key) const = 0;
```

## 4 实现类：BitwiseCompartorImpl

* Compare

```cpp
int Compare(const Slice& a, const Slice& b) const override {
  return a.compare(b);
}
```

* FindShortestSeparator，主要步骤如下：
  * 取diff\_index为start和limit的共同prefix的索引。
  * 如果diff\_index大于等于min\_length的话，直接结束函数。
  * 取diff\_byte为`(*start)[diff_index]`，当其小于0xff且diff\_byte+1小于`limit[diff_index]`时，将`(*start)[diff_index]`加一，并将start调整大小为diff\_index+1。*备注：当string的size()大于resize(size_type count)的count的话，会截断string为前count个。*

```cpp
void FindShortestSeparator(std::string* start,
                           const Slice& limit) const override {
  // Find length of common prefix
  size_t min_length = std::min(start->size(), limit.size());
  size_t diff_index = 0;
  while ((diff_index < min_length) &&
         ((*start)[diff_index] == limit[diff_index])) {
    diff_index++;
  }

  if (diff_index >= min_length) {
    // Do not shorten if one string is a prefix of the other
  } else {
    uint8_t diff_byte = static_cast<uint8_t>((*start)[diff_index]);
    if (diff_byte < static_cast<uint8_t>(0xff) &&
        diff_byte + 1 < static_cast<uint8_t>(limit[diff_index])) {
      (*start)[diff_index]++;
      start->resize(diff_index + 1);
      assert(Compare(*start, limit) < 0);
    }
  }
}
```

举例如下（以下用`std::string`代替`Slice`）：

举例一，输出“prefix4567”：

```cpp
  std::string start ("prefix4567");
  std::string limit ("prefix5678");
  FindShortestSeparator (&start, limit);
  std::cout << start << std::endl;
```

举例二，输出“prefix2”：

```cpp
  std::string start ("prefix1234");
  std::string limit ("prefix5678");
  FindShortestSeparator (&start, limit);
  std::cout << start << std::endl;
```

举例三，输出“prefix�234”，其中不可打印的字符即为0xff：

```cpp
  std::string start ("prefix");
  start += '\xff';
  start += "234";
  std::string limit ("prefix5678");
  FindShortestSeparator (&start, limit);
  std::cout << start << std::endl;
```

* FindShortSuccessor
  * 找到第一个小于0xff的字符，将其加一，将key修改大小为i+1，然后返回。
  * 如果都是等于0xff的字符，直接返回原string。

```cpp
void FindShortSuccessor(std::string* key) const override {
  // Find first character that can be incremented
  size_t n = key->size();
  for (size_t i = 0; i < n; i++) {
    const uint8_t byte = (*key)[i];
    if (byte != static_cast<uint8_t>(0xff)) {
      (*key)[i] = byte + 1;
      key->resize(i + 1);
      return;
    }
  }
  // *key is a run of 0xffs.  Leave it alone.
}
```

## 5 其它函数

* 利用NoDestructor构造一个BytewiseComparatorImpl的singleton。

```cpp
const Comparator* BytewiseComparator() {
  static NoDestructor<BytewiseComparatorImpl> singleton;
  return singleton.get();
}
```

