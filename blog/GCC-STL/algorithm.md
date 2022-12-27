# § algorithm

## copy

``` cpp
  template<typename _II, typename _OI>
    _GLIBCXX20_CONSTEXPR
    inline _OI
    copy(_II __first, _II __last, _OI __result)
    {
      // concept requirements
      __glibcxx_function_requires(_InputIteratorConcept<_II>)
      __glibcxx_function_requires(_OutputIteratorConcept<_OI,
	    typename iterator_traits<_II>::value_type>)
      __glibcxx_requires_can_increment_range(__first, __last, __result);

      return std::__copy_move_a<__is_move_iterator<_II>::__value>
	     (std::__miter_base(__first), std::__miter_base(__last), __result);
    }
```
#### \_\_copy\_move\_a

函数调用路线：

`std::copy --> __copy_move_a --> __copy_move_a1 --> __copy_move_a2 --> __copy_move::__copy_m`

`__copy_move`模板有多个特化版本，目的就是两个：
- 尽量将copy替换为memmove，注意不是memcpy，因为输入和输出的范围是允许重叠的。
- 如果使用随机访问迭代器，将循环写成for循环，并有一个明确的计数器。

`__copy_move`的定义及各个特化：

| move iterator? | 可以memcopy？ | 迭代器 | 动作 | 备注 |
| -               | -            | -     | -   | -   |
| N/A | N/A | N/A          | - | 从first到last逐个赋值到result指向的地方 | 定义 |
| Y   | N   | N/A          | - | 用`std::move`赋值 | 特化 |
| N   | N   | 随机访问迭代器 | 用difference_type作为计数器，逐个赋值 | - |
| Y   | N   | 随机访问迭代器 | 用difference_type作为计数器，逐个赋值 | - |
| N/A | Y   | 随机访问迭代器 | 使用`__builtin_memmove`            | - |

源码：

``` cpp
  template<bool _IsMove, typename _II, typename _OI>
    _GLIBCXX20_CONSTEXPR
    inline _OI
    __copy_move_a(_II __first, _II __last, _OI __result)
    {
      return std::__niter_wrap(__result,
		std::__copy_move_a1<_IsMove>(std::__niter_base(__first),
					     std::__niter_base(__last),
					     std::__niter_base(__result)));
    }

  template<bool _IsMove, typename _II, typename _OI>
    _GLIBCXX20_CONSTEXPR
    inline _OI
    __copy_move_a1(_II __first, _II __last, _OI __result)
    { return std::__copy_move_a2<_IsMove>(__first, __last, __result); }

  template<bool _IsMove, typename _II, typename _OI>
    _GLIBCXX20_CONSTEXPR
    inline _OI
    __copy_move_a2(_II __first, _II __last, _OI __result)
    {
      typedef typename iterator_traits<_II>::iterator_category _Category;
      return std::__copy_move<_IsMove, __memcpyable<_OI, _II>::__value,
			      _Category>::__copy_m(__first, __last, __result);
    }

  // All of these auxiliary structs serve two purposes.  (1) Replace
  // calls to copy with memmove whenever possible.  (Memmove, not memcpy,
  // because the input and output ranges are permitted to overlap.)
  // (2) If we're using random access iterators, then write the loop as
  // a for loop with an explicit count.

  template<bool _IsMove, bool _IsSimple, typename _Category>
    struct __copy_move
    {
      template<typename _II, typename _OI>
	_GLIBCXX20_CONSTEXPR
	static _OI
	__copy_m(_II __first, _II __last, _OI __result)
	{
	  for (; __first != __last; ++__result, (void)++__first)
	    *__result = *__first;
	  return __result;
	}
    };

  template<typename _Category>
    struct __copy_move<true, false, _Category>
    {
      template<typename _II, typename _OI>
	_GLIBCXX20_CONSTEXPR
	static _OI
	__copy_m(_II __first, _II __last, _OI __result)
	{
	  for (; __first != __last; ++__result, (void)++__first)
	    *__result = std::move(*__first);
	  return __result;
	}
    };

  template<>
    struct __copy_move<false, false, random_access_iterator_tag>
    {
      template<typename _II, typename _OI>
	_GLIBCXX20_CONSTEXPR
	static _OI
	__copy_m(_II __first, _II __last, _OI __result)
	{
	  typedef typename iterator_traits<_II>::difference_type _Distance;
	  for(_Distance __n = __last - __first; __n > 0; --__n)
	    {
	      *__result = *__first;
	      ++__first;
	      ++__result;
	    }
	  return __result;
	}
    };

  template<>
    struct __copy_move<true, false, random_access_iterator_tag>
    {
      template<typename _II, typename _OI>
	_GLIBCXX20_CONSTEXPR
	static _OI
	__copy_m(_II __first, _II __last, _OI __result)
	{
	  typedef typename iterator_traits<_II>::difference_type _Distance;
	  for(_Distance __n = __last - __first; __n > 0; --__n)
	    {
	      *__result = std::move(*__first);
	      ++__first;
	      ++__result;
	    }
	  return __result;
	}
    };

  template<bool _IsMove>
    struct __copy_move<_IsMove, true, random_access_iterator_tag>
    {
      template<typename _Tp>
	_GLIBCXX20_CONSTEXPR
	static _Tp*
	__copy_m(const _Tp* __first, const _Tp* __last, _Tp* __result)
	{
#if __cplusplus >= 201103L
	  using __assignable = conditional<_IsMove,
					   is_move_assignable<_Tp>,
					   is_copy_assignable<_Tp>>;
	  // trivial types can have deleted assignment
	  static_assert( __assignable::type::value, "type is not assignable" );
#endif
	  const ptrdiff_t _Num = __last - __first;
	  if (_Num)
	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	  return __result + _Num;
	}
    };
```

#### \_\_memcpyable

``` cpp
  // A type that is safe for use with memcpy, memmove, memcmp etc.
  template<typename _Tp>
    struct __is_nonvolatile_trivially_copyable
    {
      enum { __value = __is_trivially_copyable(_Tp) };
    };

  // Cannot use memcpy/memmove/memcmp on volatile types even if they are
  // trivially copyable, so ensure __memcpyable<volatile int*, volatile int*>
  // and similar will be false.
  template<typename _Tp>
    struct __is_nonvolatile_trivially_copyable<volatile _Tp>
    {
      enum { __value = 0 };
    };

  // Whether two iterator types can be used with memcpy/memmove.
  template<typename _OutputIter, typename _InputIter>
    struct __memcpyable
    {
      enum { __value = 0 };
    };

  template<typename _Tp>
    struct __memcpyable<_Tp*, _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };

  template<typename _Tp>
    struct __memcpyable<_Tp*, const _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };

```

`__is_trivially_copyable`由编译器决定，以下的是trivialy copyable：
- 标量
- trivially copyable的类类型
  - 
- 以上类型的数组
- cv修饰的上述类型

## fill\_n

``` cpp
  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 865. More algorithms that throw away information
  // DR 426. search_n(), fill_n(), and generate_n() with negative n
  template<typename _OI, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline _OI
    fill_n(_OI __first, _Size __n, const _Tp& __value)
    {
      return std::__fill_n_a(__first, std::__size_to_integer(__n), __value,
			       std::__iterator_category(__first));
    }
```

#### \_\_fill\_n\_a

`__fill_n_a`根据传入`fill_n`函数的first迭代器的类型，重载三个函数：输出（调用`__fill_n_a1`）、输入（调用`__fill_n_a1`）、随机访问迭代器（调用`__fill_a`）。

`__fill_n_a1`有两种重载形式：`find`的value格式是标量的时候和不是标量的时候。区别在于，当value是标量的时候，会将value先复制出来，再赋值。

`__fill_a`直接调用`__fill_a1`，后者有两种重载形式。和`__fill_n_a1`的区别类似。

`__fill_n_a1`和`__fill_a`的定义区别在于是否是随机访问迭代器；实现的区别在于，前者根据函数传入的n作为for循环的计数来遍历赋值，后者根据这个first+n，生成last作为for循环终止的迭代器。

``` cpp
  template<typename _OutputIterator, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline _OutputIterator
    __fill_n_a(_OutputIterator __first, _Size __n, const _Tp& __value,
	       std::output_iterator_tag)
    {
#if __cplusplus >= 201103L
      static_assert(is_integral<_Size>{}, "fill_n must pass integral size");
#endif
      return __fill_n_a1(__first, __n, __value);
    }

  template<typename _OutputIterator, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline _OutputIterator
    __fill_n_a(_OutputIterator __first, _Size __n, const _Tp& __value,
	       std::input_iterator_tag)
    {
#if __cplusplus >= 201103L
      static_assert(is_integral<_Size>{}, "fill_n must pass integral size");
#endif
      return __fill_n_a1(__first, __n, __value);
    }

  template<typename _OutputIterator, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline _OutputIterator
    __fill_n_a(_OutputIterator __first, _Size __n, const _Tp& __value,
	       std::random_access_iterator_tag)
    {
#if __cplusplus >= 201103L
      static_assert(is_integral<_Size>{}, "fill_n must pass integral size");
#endif
      if (__n <= 0)
	return __first;

      __glibcxx_requires_can_increment(__first, __n);

      std::__fill_a(__first, __first + __n, __value);
      return __first + __n;
    }

  template<typename _OutputIterator, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline typename
    __gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, _OutputIterator>::__type
    __fill_n_a1(_OutputIterator __first, _Size __n, const _Tp& __value)
    {
      for (; __n > 0; --__n, (void) ++__first)
	*__first = __value;
      return __first;
    }

  template<typename _OutputIterator, typename _Size, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline typename
    __gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, _OutputIterator>::__type
    __fill_n_a1(_OutputIterator __first, _Size __n, const _Tp& __value)
    {
      const _Tp __tmp = __value;
      for (; __n > 0; --__n, (void) ++__first)
	*__first = __tmp;
      return __first;
    }

  template<typename _FIte, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline void
    __fill_a(_FIte __first, _FIte __last, const _Tp& __value)
    { std::__fill_a1(__first, __last, __value); }

  template<typename _ForwardIterator, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline typename
    __gnu_cxx::__enable_if<!__is_scalar<_Tp>::__value, void>::__type
    __fill_a1(_ForwardIterator __first, _ForwardIterator __last,
	      const _Tp& __value)
    {
      for (; __first != __last; ++__first)
	*__first = __value;
    }

  template<typename _ForwardIterator, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline typename
    __gnu_cxx::__enable_if<__is_scalar<_Tp>::__value, void>::__type
    __fill_a1(_ForwardIterator __first, _ForwardIterator __last,
	      const _Tp& __value)
    {
      const _Tp __tmp = __value;
      for (; __first != __last; ++__first)
	*__first = __tmp;
    }

  // Specialization: for char types we can use memset.
  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline typename
    __gnu_cxx::__enable_if<__is_byte<_Tp>::__value, void>::__type
    __fill_a1(_Tp* __first, _Tp* __last, const _Tp& __c)
    {
      const _Tp __tmp = __c;
#if __cpp_lib_is_constant_evaluated
      if (std::is_constant_evaluated())
	{
	  for (; __first != __last; ++__first)
	    *__first = __tmp;
	  return;
	}
#endif
      if (const size_t __len = __last - __first)
	__builtin_memset(__first, static_cast<unsigned char>(__tmp), __len);
    }

  template<typename _Ite, typename _Cont, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline void
    __fill_a1(::__gnu_cxx::__normal_iterator<_Ite, _Cont> __first,
	      ::__gnu_cxx::__normal_iterator<_Ite, _Cont> __last,
	      const _Tp& __value)
    { std::__fill_a1(__first.base(), __last.base(), __value); }

  template<typename _FIte, typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline void
    __fill_a(_FIte __first, _FIte __last, const _Tp& __value)
    { std::__fill_a1(__first, __last, __value); }


  // Used by fill_n, generate_n, etc. to convert _Size to an integral type:
  inline _GLIBCXX_CONSTEXPR int
  __size_to_integer(int __n) { return __n; }
  inline _GLIBCXX_CONSTEXPR unsigned
  __size_to_integer(unsigned __n) { return __n; }
  inline _GLIBCXX_CONSTEXPR long
  __size_to_integer(long __n) { return __n; }
  inline _GLIBCXX_CONSTEXPR unsigned long
  __size_to_integer(unsigned long __n) { return __n; }
  inline _GLIBCXX_CONSTEXPR long long
  __size_to_integer(long long __n) { return __n; }
  inline _GLIBCXX_CONSTEXPR unsigned long long
  __size_to_integer(unsigned long long __n) { return __n; }
```