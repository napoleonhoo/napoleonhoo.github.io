# aligned buffer

## 1 `__aligned_membuf`

``` cpp
// A utility type containing a POD object that can hold an object of type
// _Tp initialized via placement new or allocator_traits::construct.
// Intended for use as a data member subobject, use __aligned_buffer for
// complete objects.
template <typename _Tp>
struct __aligned_membuf {
    // Target macro ADJUST_FIELD_ALIGN can produce different alignment for
    // types when used as class members. __aligned_membuf is intended
    // for use as a class member, so align the buffer as for a class member.
    // Since GCC 8 we could just use alignof(_Tp) instead, but older
    // versions of non-GNU compilers might still need this trick.
    struct _Tp2 {
        _Tp _M_t;
    };
		// 上面的注释解释了为什么写为 __alignof__(_Tp2::_M_t)，
    // 这里面的struct _Tp2就是为了获得一个_Tp类型的对齐。
    // 在GCC8之后可以直接写作：alignof(_Tp)
    alignas(__alignof__(_Tp2::_M_t)) unsigned char _M_storage[sizeof(_Tp)];

    __aligned_membuf() = default;

    // Can be used to avoid value-initialization zeroing _M_storage.
    __aligned_membuf(std::nullptr_t) {}

    void* _M_addr() noexcept { return static_cast<void*>(&_M_storage); }

    const void* _M_addr() const noexcept { return static_cast<const void*>(&_M_storage); }

    _Tp* _M_ptr() noexcept { return static_cast<_Tp*>(_M_addr()); }

    const _Tp* _M_ptr() const noexcept { return static_cast<const _Tp*>(_M_addr()); }
};
```

这个类的主要是用来作为类中某一个成员变量的对齐的内存，可以用placement new或者`alloc_traits::construct`来使用内存。

而如果对一个类使用对齐的内存的话，需要使用`__aligned_buffer`。



## 2 `__aligned_buffer`

``` cpp
#if _GLIBCXX_INLINE_VERSION
template <typename _Tp>
using __aligned_buffer = __aligned_membuf<_Tp>;
#else
// Similar to __aligned_membuf but aligned for complete objects, not members.
// This type is used in <forward_list>, <future>, <bits/shared_ptr_base.h>
// and <bits/hashtable_policy.h>, but ideally they would use __aligned_membuf
// instead, as it has smaller size for some types on some targets.
// This type is still used to avoid an ABI change.
template <typename _Tp>
struct __aligned_buffer : std::aligned_storage<sizeof(_Tp), __alignof__(_Tp)> {
    typename std::aligned_storage<sizeof(_Tp), __alignof__(_Tp)>::type _M_storage;

    __aligned_buffer() = default;

    // Can be used to avoid value-initialization
    __aligned_buffer(std::nullptr_t) {}

    void* _M_addr() noexcept { return static_cast<void*>(&_M_storage); }

    const void* _M_addr() const noexcept { return static_cast<const void*>(&_M_storage); }

    _Tp* _M_ptr() noexcept { return static_cast<_Tp*>(_M_addr()); }

    const _Tp* _M_ptr() const noexcept { return static_cast<const _Tp*>(_M_addr()); }
};
```

这个是对整个类来声明对齐的。

