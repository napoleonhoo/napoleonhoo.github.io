# §3.1 Iterator

## 1 路径

* include/levedl/iterator.h
* table/iterator.cc

## 2 功能

## 3 虚基类：Iterator

### 3.1 主要接口

```cpp
// An iterator is either positioned at a key/value pair, or
// not valid.  This method returns true iff the iterator is valid.
virtual bool Valid() const = 0;

// Position at the first key in the source.  The iterator is Valid()
// after this call iff the source is not empty.
virtual void SeekToFirst() = 0;

// Position at the last key in the source.  The iterator is
// Valid() after this call iff the source is not empty.
virtual void SeekToLast() = 0;

// Position at the first key in the source that is at or past target.
// The iterator is Valid() after this call iff the source contains
// an entry that comes at or past target.
virtual void Seek(const Slice& target) = 0;

// Moves to the next entry in the source.  After this call, Valid() is
// true iff the iterator was not positioned at the last entry in the source.
// REQUIRES: Valid()
virtual void Next() = 0;

// Moves to the previous entry in the source.  After this call, Valid() is
// true iff the iterator was not positioned at the first entry in source.
// REQUIRES: Valid()
virtual void Prev() = 0;

// Return the key for the current entry.  The underlying storage for
// the returned slice is valid only until the next modification of
// the iterator.
// REQUIRES: Valid()
virtual Slice key() const = 0;

// Return the value for the current entry.  The underlying storage for
// the returned slice is valid only until the next modification of
// the iterator.
// REQUIRES: Valid()
virtual Slice value() const = 0;

// If an error has occurred, return it.  Else return an ok status.
virtual Status status() const = 0;
```

### 3.2 构造函数和析构函数

在构造时，`cleanup_head_.function`和`cleanup_head_.next`初始化为nullptr。

在析构时，当`cleanup_head_`为空，即`cleanup_head_.function`为nullptr时，调用`cleanup_head_`的Run方法，进行**Cleanup**。并且顺着单向链表，取next，调用Run方法，并将其析构（delete）。

```cpp
Iterator::Iterator() {
  cleanup_head_.function = nullptr;
  cleanup_head_.next = nullptr;
}

Iterator::~Iterator() {
  if (!cleanup_head_.IsEmpty()) {
    cleanup_head_.Run();
    for (CleanupNode* node = cleanup_head_.next; node != nullptr;) {
      node->Run();
      CleanupNode* next_node = node->next;
      delete node;
      node = next_node;
    }
  }
}
```

### 3.3 Cleanup相关

* `RegisterCleanup`不是virtual的，不需要实现类进行重写。创建一个`CleanupNode`指针node：
  * 当`cleanup_head_`为Empty，即其function为nullptr时，将node指向`cleanup_head_`；
  * 当其不为Empty时，为node分配空间，并放在`cleanup_head_`的next处（紧挨着）。

```cpp
// Clients are allowed to register function/arg1/arg2 triples that
// will be invoked when this iterator is destroyed.
//
// Note that unlike all of the preceding methods, this method is
// not abstract and therefore clients should not override it.
using CleanupFunction = void (*)(void* arg1, void* arg2);
void RegisterCleanup(CleanupFunction function, void* arg1, void* arg2) {
  assert(func != nullptr);
  CleanupNode* node;
  if (cleanup_head_.IsEmpty()) {
    node = &cleanup_head_;
  } else {
    node = new CleanupNode();
    node->next = cleanup_head_.next;
    cleanup_head_.next = node;
  }
  node->function = func;
  node->arg1 = arg1;
  node->arg2 = arg2;
}
```

* Cleanup函数是存储在单向链表中。

```cpp
// Cleanup functions are stored in a single-linked list.
// The list's head node is inlined in the iterator.
struct CleanupNode {
  // True if the node is not used. Only head nodes might be unused.
  bool IsEmpty() const { return function == nullptr; }
  // Invokes the cleanup function.
  void Run() {
    assert(function != nullptr);
    (*function)(arg1, arg2);
  }

  // The head node is used if the function pointer is not null.
  CleanupFunction function;
  void* arg1;
  void* arg2;
  CleanupNode* next;
};
CleanupNode cleanup_head_;
```

## 4 实现类：EmptyIterator

主要代码如下所示。可以看出，EmptyIterator基本上就是名字所示的样子，一个Empty的。

```cpp
namespace {

class EmptyIterator : public Iterator {
 public:
  EmptyIterator(const Status& s) : status_(s) {}
  ~EmptyIterator() override = default;

  bool Valid() const override { return false; }
  void Seek(const Slice& target) override {}
  void SeekToFirst() override {}
  void SeekToLast() override {}
  void Next() override { assert(false); }
  void Prev() override { assert(false); }
  Slice key() const override {
    assert(false);
    return Slice();
  }
  Slice value() const override {
    assert(false);
    return Slice();
  }
  Status status() const override { return status_; }

 private:
  Status status_;
};

}  // anonymous namespace
```

相关函数：

```cpp
Iterator* NewEmptyIterator() { return new EmptyIterator(Status::OK()); }

Iterator* NewErrorIterator(const Status& status) {
  return new EmptyIterator(status);
}
```

## 5 IteratorWrapper

路径：table/iterator_wrapper.h

主要代码如下。IteratorWrapper提供了和Iterator相似的接口函数，并将`valid()`和`key()`的结果缓存下来。这可以避免虚函数调用和增加cache局部性。

```cpp
// A internal wrapper class with an interface similar to Iterator that
// caches the valid() and key() results for an underlying iterator.
// This can help avoid virtual function calls and also gives better
// cache locality.
class IteratorWrapper {
 public:
  IteratorWrapper() : iter_(nullptr), valid_(false) {}
  explicit IteratorWrapper(Iterator* iter) : iter_(nullptr) { Set(iter); }
  ~IteratorWrapper() { delete iter_; }
  Iterator* iter() const { return iter_; }

  // Takes ownership of "iter" and will delete it when destroyed, or
  // when Set() is invoked again.
  void Set(Iterator* iter) {
    delete iter_;
    iter_ = iter;
    if (iter_ == nullptr) {
      valid_ = false;
    } else {
      Update();
    }
  }

  // Iterator interface methods
  bool Valid() const { return valid_; }
  Slice key() const {
    assert(Valid());
    return key_;
  }
  Slice value() const {
    assert(Valid());
    return iter_->value();
  }
  // Methods below require iter() != nullptr
  Status status() const {
    assert(iter_);
    return iter_->status();
  }
  void Next() {
    assert(iter_);
    iter_->Next();
    Update();
  }
  void Prev() {
    assert(iter_);
    iter_->Prev();
    Update();
  }
  void Seek(const Slice& k) {
    assert(iter_);
    iter_->Seek(k);
    Update();
  }
  void SeekToFirst() {
    assert(iter_);
    iter_->SeekToFirst();
    Update();
  }
  void SeekToLast() {
    assert(iter_);
    iter_->SeekToLast();
    Update();
  }

 private:
  void Update() {
    valid_ = iter_->Valid();
    if (valid_) {
      key_ = iter_->key();
    }
  }

  Iterator* iter_;
  bool valid_;
  Slice key_;
};
```

