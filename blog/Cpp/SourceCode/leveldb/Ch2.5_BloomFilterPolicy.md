# § 2.5 BloomFilterPolicy

## 1 代码路径

* util/bloom.cc

## 2 功能

* 这个类继承了§2.4所讲述的FilterPolicy类。
* 实现的是一个著名的用于快速查找的，基于Hash原理的数据结构，即Bloom Filter（布隆过滤器）。

## 3 背景知识

### 3.1 Bloom Filter

### 3.2 Double Hashing

基本公式为：
$$
h(i, k)\ =\ h_1(k)\ +\ i\ \times\ h_2(i,\ k)\ MOD\ |T|
$$
其中，$${h_1}$$、$${h_2}$$是两个Hash函数，$${i}$$是Hash表中的位置，$${k}$$是位置$${i}$$对应的值，$${|T|}$$是Hash表的大小。

## 4 主要成员变量

* `size_t bits_per_key_;`一个key用多少位来表示。
* `size_t k_;`探测次数，哈希函数的个数。

## 5 主要成员函数

* 构造函数：`explicit BloomFilterPolicy(int bits_per_key);`根据输入的`bits_per_key`赋值成员变量`bits_per_key_`，然后算出来`k_`的大小。另外这个会规则化到[1,  30]之间。核心代码如下：

```cpp
k_ = static_cast<size_t>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
```

* `void CreateFilter(const Slice* keys, int n, std::string* dst) const override;`

  * 计算bloom filter的大小`size_t bits = n * bits_per_key_`，修正`bits`的值，使其最小为64。计算`bits`的所占用的空间`bytes`（单位是字节），利用`bytes`再计算`bits`，目的是将其化成8的整数倍。<u>总之来讲，计算了这些key所需的字节数和位数。</u>相关代码：

  ```cpp
  size_t bytes = (bits / 7) * 8;
  bits = bytes * 8;
  ```

  * 将`k_`记录在`dst`的最后一个元素中，表示此filter需要探测的次数。
  * 对每个key，计算一个值放入bitmap（实际上是`char*`）。此处用到的公式是Double Hashing。$${h_1}$$是名为`BloomHash`的一个自定义Hash函数，$${h_2}$$是delta，即h右移17位的结果。根据Hash函数，将过滤器的某一位置1。具体代码如下：

  ```cpp
  const size_t init_size = dst->size();
  dst->resize(init_size + bytes, 0);
  dst->push_back(static_cast<char>(k_));  // Remember # of probes in filter
  char* array = &(*dst)[init_size];
  for (int i = 0; i < n; i++) {
    // Use double-hashing to generate a sequence of hash values.
    // See analysis in [Kirsch,Mitzenmacher 2006].
    uint32_t h = BloomHash(keys[i]);
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (size_t j = 0; j < k_; j++) {
      const uint32_t bitpos = h % bits; // 计算需要的是哪一个位置为1。
      array[bitpos / 8] |= (1 << (bitpos % 8)); // 将相应位置1。
      h += delta; // 每次循环h都加上delta，相当于：h1 + i * h2。
    }
  }
  ```

* `void bool KeyMayMatch(const Slice& key,  const Slice& bloom_filter) const override;`

  * 当`bloom_filter`的大小小于2时，返回false。
  * 得到上一个函数存储在`bloom_filter`中的`k_`参数，即探测次数。当其大于30时，返回true。*备注：这一点和构造函数中的策略是不同的，代码中的注释说到这个是为了以后的扩展所保留了。*
  * 剩下的基本上和`CreateFilter`是类似的操作了，先用Double Hashing算出这个key的各个应该为1的位置，如果这些位置为0的话，返回false。

## 6 相关函数

* `static int BloomHash(const Slice& key);`背后使用了LevelDB自己实现的一版类murmur hash算法。