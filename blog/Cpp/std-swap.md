# std::swap

std::swap的实现是（bits/move.h）：

```cpp
  /**
   *  @brief Swaps two values.
   *  @param  __a  A thing of arbitrary type.
   *  @param  __b  Another thing of arbitrary type.
   *  @return   Nothing.
  */
  template<typename _Tp>
    _GLIBCXX20_CONSTEXPR
    inline
#if __cplusplus >= 201103L
    typename enable_if<__and_<__not_<__is_tuple_like<_Tp>>,
			      is_move_constructible<_Tp>,
			      is_move_assignable<_Tp>>::value>::type
#else
    void
#endif
    swap(_Tp& __a, _Tp& __b)
    _GLIBCXX_NOEXCEPT_IF(__and_<is_nothrow_move_constructible<_Tp>,
				is_nothrow_move_assignable<_Tp>>::value)
    {
#if __cplusplus < 201103L
      // concept requirements
      __glibcxx_function_requires(_SGIAssignableConcept<_Tp>)
#endif
      _Tp __tmp = _GLIBCXX_MOVE(__a);
      __a = _GLIBCXX_MOVE(__b);
      __b = _GLIBCXX_MOVE(__tmp);
    }

  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 809. std::swap should be overloaded for array types.
  /// Swap the contents of two arrays.
  template<typename _Tp, size_t _Nm>
    _GLIBCXX20_CONSTEXPR
    inline
#if __cplusplus >= 201103L
    typename enable_if<__is_swappable<_Tp>::value>::type
#else
    void
#endif
    swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
    _GLIBCXX_NOEXCEPT_IF(__is_nothrow_swappable<_Tp>::value)
    {
      for (size_t __n = 0; __n < _Nm; ++__n)
	swap(__a[__n], __b[__n]);
    }
```

其实相对简单，去掉所有的宏、模版等的操作，留下的只有最后面的三行。

而其中的`_GLIBCXX_MOVE`的定义是（去掉其他的定义）：

```cpp
#if __cplusplus >= 201103L
#define _GLIBCXX_MOVE(__val) std::move(__val)
#else
#define _GLIBCXX_MOVE(__val) (__val)
#endif
```

