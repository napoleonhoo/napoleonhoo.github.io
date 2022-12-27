# §Ch3.8 TwoLevelIterator

## 1 路径

* table/two_level_iterator.cc
* table/two_level_iterator.h

## 2 功能

* 继承自Iterator。

## 3 成员变量

```cpp
BlockFunction block_function_;
void* arg_;
const ReadOptions options_;
Status status_;
IteratorWrapper index_iter_;
IteratorWrapper data_iter_;  // May be nullptr
// If data_iter_ is non-null, then "data_block_handle_" holds the
// "index_value" passed to block_function_ to create the data_iter_.
std::string data_block_handle_;
```

## 4 成员函数

### 4.1 构造函数

```cpp
TwoLevelIterator::TwoLevelIterator(Iterator* index_iter,
                                   BlockFunction block_function, void* arg,
                                   const ReadOptions& options)
    : block_function_(block_function),
      arg_(arg),
      options_(options),
      index_iter_(index_iter),
      data_iter_(nullptr) {}
```

*备注：其它函数（如析构、复制、移动）为默认。*

### 4.2 Seek

#### 4.2.1 InitDataBlock

* 当index\_iter\_无效时，将data\_iter\_置为nullptr，否则
* 得到index\_iter\_指向的value，即handle。
* 当data\_iter\_不为nullptr，且handle等于data\_block\_handle\_时，意味着data\_iter\_已经被设置过了，因此不再做任何操作。否则，
* 根据传入的block\_function\_、arg\_、options\_、handle等构造一个iter（Iterator类型），将handle赋给data\_block\_handle\_，将data\_iter\_设置为iter。

<u>总结：利用当前的indexBlockIterator来设置dataBlockIterator指向当前indexBlock所指向的区域。</u>

```cpp
void TwoLevelIterator::InitDataBlock() {
  if (!index_iter_.Valid()) {
    SetDataIterator(nullptr);
  } else {
    Slice handle = index_iter_.value();
    if (data_iter_.iter() != nullptr &&
        handle.compare(data_block_handle_) == 0) {
      // data_iter_ is already constructed with this iterator, so
      // no need to change anything
    } else {
      Iterator* iter = (*block_function_)(arg_, options_, handle);
      data_block_handle_.assign(handle.data(), handle.size());
      SetDataIterator(iter);
    }
  }
}

void TwoLevelIterator::SetDataIterator(Iterator* data_iter) {
  if (data_iter_.iter() != nullptr) SaveError(data_iter_.status());
  data_iter_.Set(data_iter);
}
```

#### 4.2.2 SkipEmptyDataBlocksForward

SkipEmptyDataBlocksForward的主要流程如下：

* 当data\_iter\_指向的区域为空、无效时，才需要跳过这些区域
* index\_iter\_调用Next，指向下一个indexBlock
* 调用IniDataBlock，更新data\_iter\_，并令其调用SeekToFirst

```cpp
void TwoLevelIterator::SkipEmptyDataBlocksForward() {
  while (data_iter_.iter() == nullptr || !data_iter_.Valid()) {
    // Move to next block
    if (!index_iter_.Valid()) {
      SetDataIterator(nullptr);
      return;
    }
    index_iter_.Next();
    InitDataBlock();
    if (data_iter_.iter() != nullptr) data_iter_.SeekToFirst();
  }
}
```

#### 4.2.3 Seek

Seek的主要流程如下：

* indexIterator对target调用Seek
* dataIterator对target调用Seek
* 调用SkipEmptyDataBlocksForward

```cpp
void TwoLevelIterator::Seek(const Slice& target) {
  index_iter_.Seek(target);
  InitDataBlock();
  if (data_iter_.iter() != nullptr) data_iter_.Seek(target);
  SkipEmptyDataBlocksForward();
}
```

### 4.3 SeekToFirst、SeekToLast

这两个函数先将indexIter走到第一个（最后一个），再将dataIter走到最第一个（最后一个），最后向前（后）跳过空白区域。

```cpp
void TwoLevelIterator::SeekToFirst() {
  index_iter_.SeekToFirst();
  InitDataBlock();
  if (data_iter_.iter() != nullptr) data_iter_.SeekToFirst();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::SeekToLast() {
  index_iter_.SeekToLast();
  InitDataBlock();
  if (data_iter_.iter() != nullptr) data_iter_.SeekToLast();
  SkipEmptyDataBlocksBackward();
}
```

### 4.4 Next、Prev

都是对dataIter调用Next（Prev），然后跳过空区域。

```cpp
void TwoLevelIterator::Next() {
  assert(Valid());
  data_iter_.Next();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::Prev() {
  assert(Valid());
  data_iter_.Prev();
  SkipEmptyDataBlocksBackward();
}
```

### 4.5 SkipEmptyDataBlocksBackward

和SkipEmptyDataBlocksForward正好相反。

```cpp
void TwoLevelIterator::SkipEmptyDataBlocksBackward() {
  while (data_iter_.iter() == nullptr || !data_iter_.Valid()) {
    // Move to next block
    if (!index_iter_.Valid()) {
      SetDataIterator(nullptr);
      return;
    }
    index_iter_.Prev();
    InitDataBlock();
    if (data_iter_.iter() != nullptr) data_iter_.SeekToLast();
  }
}
```

## 5 相关函数

注意：TwoLevelIterator的声明和定义都在匿名命名空间内，外界是通过全局函数调用的。

```cpp
Iterator* NewTwoLevelIterator(Iterator* index_iter,
                              BlockFunction block_function, void* arg,
                              const ReadOptions& options) {
  return new TwoLevelIterator(index_iter, block_function, arg, options);
}
```

