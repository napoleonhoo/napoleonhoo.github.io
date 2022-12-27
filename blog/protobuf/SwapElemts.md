# repeated field的删除操作

看到这么一段代码：

```cpp
// sample是一个pb，context是其中的repeated field。
auto contexts = sample.mutable_context();
for (int32_t i = contexts->size() - 1; i >= 0; --i) {
    uint64_t id = contexts->Get(i).content_feature().id();
    if ((it = umap.find(id)) == umap.end()) {
        
        auto tail_index = contexts->size() - 1;
        if (i != tail_index) {
            contexts->SwapElements(i, tail_index);
        }
        contexts->RemoveLast();
        
        continue;
    }
}
```

其中，在对pb中Repeated字段进行删除的时候，会先将其Swap到最后一个位置，然后在将其Remove掉。这样真的会有收益吗？先看一下源代码（src/google/protobuf/repeated_field.h）：

```cpp
// class RepeatedField
// A note on the representation here (see also comment below for
// RepeatedPtrFieldBase's struct Rep):
//
// We maintain the same sizeof(RepeatedField) as before we added arena support
// so that we do not degrade performance by bloating memory usage. Directly
// adding an arena_ element to RepeatedField is quite costly. By using
// indirection in this way, we keep the same size when the RepeatedField is
// empty (common case), and add only an 8-byte header to the elements array
// when non-empty. We make sure to place the size fields directly in the
// RepeatedField class to avoid costly cache misses due to the indirection.
int current_size_;
int total_size_;
struct Rep {
  Arena* arena;
  Element elements[1];
};
Rep* rep_;

template <typename Element>
inline Element* RepeatedField<Element>::Add() {
  if (current_size_ == total_size_) Reserve(total_size_ + 1);
  return &rep_->elements[current_size_++];
}

template <typename Element>
void RepeatedField<Element>::SwapElements(int index1, int index2) {
  using std::swap;  // enable ADL with fallback
  swap(rep_->elements[index1], rep_->elements[index2]);
}

template <typename Element>
inline void RepeatedField<Element>::RemoveLast() {
  GOOGLE_DCHECK_GT(current_size_, 0);
  current_size_--;
}
```

注意到：

1. RepeatedField是对原生类型，即非string、Message类型的Repeated Field的实现；对于string、Message的实现类是：RepeatedPtrFiled（会调用RepeatedPtrFieldBase类）。目前以RepeatedField为例。
2. RepeatedFiled的具体数据元素在struct Rep里面，一个是arena，protobuf的内存分配器；一个是elements数组，存储Element元素，此处的Element即为repeated field的元素类型，如int、float等。为什么说这里是一个数组呢？可以从Add函数看出来。这是对数组的一个trick。也可以看一下官方的注释，这样做的好处是使得所有的RepeatedField的变量的大小保持一致。elements有大有小，但elements本质上只是一个指针，并且还在Arena上分配。
3. SwapElements直接调用std::swap。对于不同的类型，可能有不同的具体操作。因为其调用了std中的swap，具体还要看gcc/clang等的实现才行。
4. RemoveLast直接将size减一，不需要做其他的任何操作，这就是有自己内存分配器的好处。