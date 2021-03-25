# §2.11 Cache

## 1 路径

* include/leveldb/cache.h
* util/cache.cc

## 2 功能

* 提供leveldb的缓存。

## 3 类

### 3.1 Cache

### 3.2 LRUHandle

#### 3.2.1 功能

* struct，是LRU的主要存储结构。

#### 3.2.2 主要成员变量

- `void* value;`
- `void (*deleter)(const Slice&, void* value);`相当于析构函数，只是不会释放内存空间。
- `LRUHandle* next_hash;`
- `LRUHandle* next;`
- `LRUHandle* prev;`
- `size_t charge; `
- `size_t key_length;`key的长度。
- `bool in_cache; `
- `uint32_t refs;`
- `uint32_t hash;`
- `char key_data[1]; `指向key的指针。

#### 3.2.3 主要成员函数

* `Slice key() const;`利用`key_data`和`key_length`构造一个Slice并返回。

### 3.3 HandleTable

#### 3.3.3.1 功能

#### 3.3.3.2 主要成员变量

* `uint32_t length_;`哈希表的长度。
* `uint32_t elems_;`哈希表中元素的个数。备注：哈希表的程度时指`LRUHandle*`数组（即`list_`）的长度。一般情况下，`length_`大于`elems_`，但调用`Insert()`后，`elems_`又可能大于`length_`，这时需要`Resize()`。
* `LRUHandle** list_;`

#### 3.3.3.3 主要成员函数

* 构造函数：`length_`、`elems_`赋值为0，`list_`赋值nullptr，并调用`Resize()`。
* 析构函数：析构`list_`。

* `private: LRUHandle** FindPointer(const Slice& key, uint32_t hash);`当key存在于`list_`时，返回指向key的上一个元素的`next_hash`的指针，实际上是一个**二级指针**；当key不存在于`list_`中时，返回nullptr。注意这一行：

```cpp
LRUHandle** ptr = &list_[hash & (length_ - 1)];
```

由下面的`Resize()`函数来看，`length_`为2的幂，则`hash & (length_ - 1)`等于`hash % length_`。

* `private: void Resize();`

* `LRUHandle* Insert(LRUHandle* h);`

```cpp
LRUHandle* Insert(LRUHandle* h) {
  LRUHandle** ptr = FindPointer(h->key(), h->hash);           // Step1
  LRUHandle* old = *ptr;                                      // Step2
  h->next_hash = (old == nullptr ? nullptr : old->next_hash); // Step3
  *ptr = h;                                                   // Step4
  if (old == nullptr) {
    ++elems_;
    if (elems_ > length_) {
      // Since each cache entry is fairly large, we aim for a small
      // average linked list length (<= 1).
      Resize();
    }
  }
  return old;
}
```



![示意图](./LevelDB_Cache_Insert.png)



* `private: LRUHandle* Remove(const Slice& key, uint32_t hash);`这里的逻辑和`Insert`的逻辑类似，重点都是**二级指针**，另外会对`elems_`减1。另外，这里不会`Resize()`。

### 3.4 LRUCache

#### 3.4.1 功能

* Sharded cache的一个单Shard。

#### 3.4.2 主要成员变量

- `size_t capacity_;`
- `mutable port::Mutex mutex_;`
- `size_t usage_ GUARDED_BY(mutex_);`
- `LRUHandle lru_ GUARDED_BY(mutex_);`lru\_.next是最旧的，lru\_.prev是最新的。里面的元素满足条件：1）in\_cache=true；2）refs=1。
- `LRUHandle in_use_ GUARDED_BY(mutex_);`正在被客户端使用的。里面的元素满足条件：1）in\_cache=true；2）refs>=2。
- `HandleTable table_ GUARDED_BY(mutex_);`

#### 3.4.3 主要成员函数

* 构造函数：`capacity_`、`usage_`赋值为0，`lru_`、`in_use_`赋值为空循环链表。

```cpp
LRUCache::LRUCache() : capacity_(0), usage_(0) {
  // Make empty circular linked lists.
  lru_.next = &lru_;
  lru_.prev = &lru_;
  in_use_.next = &in_use_;
  in_use_.prev = &in_use_;
}
```

* 析构函数：对所有lru\_.next指向的数据及之后的数据，进行删除。
* `private: void LRU_Remove(LRUHandle* e) ;`将e从双向链表中剔除（对next和prev指针的操作），但不析构。
* `private: void LRU_Append(LRUHandle* list, LRUHandle* e);`将e放到list前，也只是对next和prev指针操作。
* `private: void Ref(LRUHandle* e);`当e的refs等于1，且in_cache为true时，意味着其存在于`lru_`里面，则将其从`lru_`中删除，并放入`in_use_`中（使用`LRU_Remove(e); LRU_Append(&in_use_, e);`）。最后，要增加e的refs。
* `private: void Unref(LRUHandle* e);`首先对e的refs减一。当refs等于0时，调用deleter，并将e释放内存空间（调用free）。当e的in_cache为true，且refs为1时，将其放入`lru_`中（使用`LRU_Remove(e); LRU_Append(&lru_, e);`）。*备注：这里是唯一进行内存释放的地方。*
* `Cache::Handle* Lookup(const Slice& key, uint32_t hash);`先利用mutex\_加锁，然后调用`table_.Lookup()`，如果找得到，则对其调用Ref，增加引用计数。
* `void Release(Cache::Handle* handle);`先利用mutex\_加锁，然后对handle调用Unref。
* `private: bool FinishErase(LRUHandle* e);`调用`LRU_Remove()`将e移除，设置in_cache为false，usage\_减去e的charge，并对e调用Unref。

```cpp
// If e != nullptr, finish removing *e from the cache; it has already been
// removed from the hash table.  Return whether e != nullptr.
bool LRUCache::FinishErase(LRUHandle* e) {
  if (e != nullptr) {
    assert(e->in_cache);
    LRU_Remove(e);
    e->in_cache = false;
    usage_ -= e->charge;
    Unref(e);
  }
  return e != nullptr;
}
```

* `void Erase(const Slice& key, uint32_t hash);`这是用mutex\_锁起来的。

```cpp
void LRUCache::Erase(const Slice& key, uint32_t hash) {
  MutexLock l(&mutex_);
  FinishErase(table_.Remove(key, hash));
}
```

* `void Prune();`删掉lru\_.next指向的以及之后的数据。

```cpp
void LRUCache::Prune() {
  MutexLock l(&mutex_);
  while (lru_.next != &lru_) {
    LRUHandle* e = lru_.next;
    assert(e->refs == 1);
    bool erased = FinishErase(table_.Remove(e->key(), e->hash));
    if (!erased) {  // to avoid unused variable when compiled NDEBUG
      assert(erased);
    }
  }
}
```

* `Cache::Handle* LRUCache::Insert(const Slice& key, uint32_t hash, void* value, size_t charge, void (*deleter)(const Slice& key, void* value));`

  * 先利用传入的参数构造一个LRUHandle对象e。
  * 然后再判断capacity\_是否大于0，如果大于的话，则将其插入到in\_use\_和table\_里面，注意这边usage\_要加上charge。否则将e的next指针设为nullptr。*注意：这里同时将其插入in\_use和table\_。*主要代码如下：

  ```cpp
  if (capacity_ > 0) {
    e->refs++;  // for the cache's reference.
    e->in_cache = true;
    LRU_Append(&in_use_, e);
    usage_ += charge;
    FinishErase(table_.Insert(e));
  } else {  // don't cache. (capacity_==0 is supported and turns off caching.)
    // next is read by key() in an assert, so it must be initialized
    e->next = nullptr;
  }
  ```

  * 另外，这个时候可能会进行一轮的删除操作，主要代码如下：

  ```cpp
  while (usage_ > capacity_ && lru_.next != &lru_) {
    LRUHandle* old = lru_.next;
    assert(old->refs == 1);
    bool erased = FinishErase(table_.Remove(old->key(), old->hash));
    if (!erased) {  // to avoid unused variable when compiled NDEBUG
      assert(erased);
    }
  }
  ```

  * 这个函数是用mutex\_保护的。

* 其它函数

```cpp
void SetCapacity(size_t capacity) { capacity_ = capacity; }
size_t TotalCharge() const {
  MutexLock l(&mutex_);
  return usage_;
}
```

#### 3.4.4 备注

LRUCache对一个key的插入（不涉及内存分配的情况下）、删除效率都是O(1)的。插入时，同时插入HandleTable和LRUHandle\*链表；删除时，先在HandleTable中删除并得到节点的指针，而后直接利用此指针在LRUHandle\*链表中删除。

###3.5 ShardedLRUCache

#### 3.5.1 功能

* 继承自Cache，是外部实际使用的Cache。
* 多Shard的LRUCache。

#### 3.5.2 相关变量

```cpp
static const int kNumShardBits = 4;
static const int kNumShards = 1 << kNumShardBits; // 16
```

#### 3.5.3 主要成员变量

*  `LRUCache shard_[kNumShards];`共有16个Shard
* `port::Mutex id_mutex_;`专门用在保护last\_id\_的。
* `uint64_t last_id_;`ShardCache的全局ID。

#### 3.5.4 主要成员函数

* `private: `得到一个Slice的Hash值，这里用的Hash是Murmur Hash。

```cpp
static inline uint32_t HashSlice(const Slice& s) {
  return Hash(s.data(), s.size(), 0);
}
```

* `private:`根据Hash值得到相应的Shard。

```cpp
static uint32_t Shard(uint32_t hash) { return hash >> (32 - kNumShardBits); 
```

* 构造函数：设置每一个Shard的初始Capacity。

```cpp
explicit ShardedLRUCache(size_t capacity) : last_id_(0) {
  const size_t per_shard = (capacity + (kNumShards - 1)) / kNumShards;
  for (int s = 0; s < kNumShards; s++) {
    shard_[s].SetCapacity(per_shard);
  }
}
```

* Insert、Lookup、Release、Erase函数，实际上的实现都是调用了先得到key相对应的Hash值，然后得到对应的Shard，最后调用LRUCache对应的Insert、Lookup、Release、Erase函数。

```cpp
Handle* Insert(const Slice& key, void* value, size_t charge,
               void (*deleter)(const Slice& key, void* value)) override {
  const uint32_t hash = HashSlice(key);
  return shard_[Shard(hash)].Insert(key, hash, value, charge, deleter);
}
Handle* Lookup(const Slice& key) override {
  const uint32_t hash = HashSlice(key);
  return shard_[Shard(hash)].Lookup(key, hash);
}
void Release(Handle* handle) override {
  LRUHandle* h = reinterpret_cast<LRUHandle*>(handle);
  shard_[Shard(h->hash)].Release(handle);
}
void Erase(const Slice& key) override {
  const uint32_t hash = HashSlice(key);
  shard_[Shard(hash)].Erase(key, hash);
}
```

* NewId，获取一个新的ID，这是用mutex保护的。

```cpp
uint64_t NewId() override {
  MutexLock l(&id_mutex_);
  return ++(last_id_);
}
```

#### 3.5.5 相关函数

* 下面的函数不属于任何类，会返回一个ShardedLRUCache。

```cpp
Cache* NewLRUCache(size_t capacity) { return new ShardedLRUCache(capacity); 
```

