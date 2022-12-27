# § <ptr_traits>/pointer_traits

Ptr\_traits主要向外界提供了：

- 对函数`pointer_to`的封装，对于非原生pointer（称为fancy pointer），调用其`_Ptr::pointer_to`函数。对于原生pointer，调用`std::address_of`。

``` cpp
			// 一般性地
			static _Ptr
      pointer_to(__make_not_void<element_type>& __e)
      { return _Ptr::pointer_to(__e); }
			// raw pointer
			static _GLIBCXX20_CONSTEXPR pointer
      pointer_to(__make_not_void<element_type>& __r) noexcept
      { return std::addressof(__r); }
```



- 对函数`to_address`的封装（since C++20），调用了`std::__to_address`（见下）。

``` cpp
  /**
   * @brief Obtain address referenced by a pointer to an object
   * @param __ptr A pointer to an object
   * @return @c __ptr
   * @ingroup pointer_abstractions
  */
  template<typename _Tp>
    constexpr _Tp*
    to_address(_Tp* __ptr) noexcept
    { return std::__to_address(__ptr); }

  /**
   * @brief Obtain address referenced by a pointer to an object
   * @param __ptr A pointer to an object
   * @return @c pointer_traits<_Ptr>::to_address(__ptr) if that expression is
             well-formed, otherwise @c to_address(__ptr.operator->())
   * @ingroup pointer_abstractions
  */
  template<typename _Ptr>
    constexpr auto
    to_address(const _Ptr& __ptr) noexcept
    { return std::__to_address(__ptr); }
```



#### \_\_make\_not\_void

``` cpp
class __undefined;

template<typename _Tp>
    using __make_not_void
      = typename conditional<is_void<_Tp>::value, __undefined, _Tp>::type;
```

这个模板的目的是编译器*防止*输入的类型是void的。设计原理：当`_Tp`是void的时候，返回的类型是`__undefined`，恰如其名，这个类是未定义的。所以，这个时候会编译报错。



##### \_\_replace\_first\_arg

``` cpp
  // Given Template<T, ...> and U return Template<U, ...>, otherwise invalid.
  template<typename _Tp, typename _Up>
    struct __replace_first_arg
    { };

  template<template<typename, typename...> class _Template, typename _Up,
           typename _Tp, typename... _Types>
    struct __replace_first_arg<_Template<_Tp, _Types...>, _Up>
    { using type = _Template<_Up, _Types...>; };

  template<typename _Tp, typename _Up>
    using __replace_first_arg_t = typename __replace_first_arg<_Tp, _Up>::type;
```



##### \_\_get\_frist\_arg

``` cpp
  // Given Template<T, ...> return T, otherwise invalid.
  template<typename _Tp>
    struct __get_first_arg
    { using type = __undefined; };

  template<template<typename, typename...> class _Template, typename _Tp,
           typename... _Types>
    struct __get_first_arg<_Template<_Tp, _Types...>>
    { using type = _Tp; };

  template<typename _Tp>
    using __get_first_arg_t = typename __get_first_arg<_Tp>::type;
```

