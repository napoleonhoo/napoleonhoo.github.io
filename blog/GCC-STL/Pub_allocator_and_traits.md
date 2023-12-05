# allocator & allocator_traits

## 1 allocator

``` cpp
      // NB: __n is permitted to be 0.  The C++ standard says nothing
      // about what the return value is when __n == 0.
      _GLIBCXX_NODISCARD _Tp*
      allocate(size_type __n, const void* = static_cast<const void*>(0))
      {
	if (__builtin_expect(__n > this->_M_max_size(), false))
	  {
	    // _GLIBCXX_RESOLVE_LIB_DEFECTS
	    // 3190. allocator::allocate sometimes returns too little storage
	    if (__n > (std::size_t(-1) / sizeof(_Tp)))
	      std::__throw_bad_array_new_length();
	    std::__throw_bad_alloc();
	  }

#if __cpp_aligned_new
	if (alignof(_Tp) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
	  {
	    std::align_val_t __al = std::align_val_t(alignof(_Tp));
	    return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp), __al));
	  }
#endif
	return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
      }

      // __p is not permitted to be a null pointer.
      void
      deallocate(_Tp* __p, size_type __t __attribute__ ((__unused__)))
      {
#if __cpp_aligned_new
	if (alignof(_Tp) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
	  {
	    ::operator delete(__p,
# if __cpp_sized_deallocation
			      __t * sizeof(_Tp),
# endif
			      std::align_val_t(alignof(_Tp)));
	    return;
	  }
#endif
	::operator delete(__p
#if __cpp_sized_deallocation
			  , __t * sizeof(_Tp)
#endif
			 );
      }
```

allocate函数调用的是`operator new`来分配空间，返回的未被初始化的空间。deallocate函数调用的是`operator delete`。



``` cpp
  template<typename _Up, typename... _Args>
	void
	construct(_Up* __p, _Args&&... __args)
	noexcept(std::is_nothrow_constructible<_Up, _Args...>::value)
	{ ::new((void *)__p) _Up(std::forward<_Args>(__args)...); }

  template<typename _Up>
	void
	destroy(_Up* __p)
	noexcept(std::is_nothrow_destructible<_Up>::value)
	{ __p->~_Up(); }
```

construct函数，调用placement new，在传入的内存上，构建对象。destroy函数，调用传入对象的析构函数。



``` cpp
      _GLIBCXX_CONSTEXPR size_type
      _M_max_size() const _GLIBCXX_USE_NOEXCEPT
      {
#if __PTRDIFF_MAX__ < __SIZE_MAX__
	return std::size_t(__PTRDIFF_MAX__) / sizeof(_Tp);
#else
	return std::size_t(-1) / sizeof(_Tp);
#endif
      }
```

这是分配的最大可能空间。



```cpp
  template<typename _Up>
	friend _GLIBCXX20_CONSTEXPR bool
	operator==(const new_allocator&, const new_allocator<_Up>&)
	_GLIBCXX_NOTHROW
	{ return true; }

#if __cpp_impl_three_way_comparison < 201907L
      template<typename _Up>
	friend _GLIBCXX20_CONSTEXPR bool
	operator!=(const new_allocator&, const new_allocator<_Up>&)
	_GLIBCXX_NOTHROW
	{ return false; }
#endif
```

根据[cppreference](https://en.cppreference.com/w/cpp/memory/allocator/operator_cmp)上的解释，allocator是无状态的，两个默认的allocator是相同的。



## 2 allocator_traits

Allocator_traits提供了各种allocator对外界的统一接口，提供了诸如：`allocate`、`deallocate`、`construct`、`destroy`、`max_size`等的函数。

