# § type_traits

#### integral_const

``` cpp
  /// integral_constant
  template<typename _Tp, _Tp __v>
    struct integral_constant
    {
      static constexpr _Tp                  value = __v;
      typedef _Tp                           value_type;
      typedef integral_constant<_Tp, __v>   type;
      constexpr operator value_type() const noexcept { return value; }
#if __cplusplus > 201103L

#define __cpp_lib_integral_constant_callable 201304

      constexpr value_type operator()() const noexcept { return value; }
#endif
    };

  template<typename _Tp, _Tp __v>
    constexpr _Tp integral_constant<_Tp, __v>::value;
```

它接受一个类型（`_Tp`）和相应类型的一个值(`__v`)，作为模板参数。使用方法：

``` cpp
typedef std::integral_const<int, 2> two;
std::cout << two::value << std::endl; // output: 2
```



#### true_type & false_type

``` cpp
  /// The type used as a compile-time boolean with true value.
  typedef integral_constant<bool, true>     true_type;

  /// The type used as a compile-time boolean with false value.
  typedef integral_constant<bool, false>    false_type;
```

true_type和false_type是一种type，它们的value是true和false。这个很长使用。



#### condition

``` cpp
  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    struct conditional
    { typedef _Iftrue type; };

  // Partial specialization for false.
  template<typename _Iftrue, typename _Iffalse>
    struct conditional<false, _Iftrue, _Iffalse>
    { typedef _Iffalse type; };
```

默认将第二个模板参数（`_Iftrue`）设置为type，结合下面看，这是在第一个模板参数（`_Cond`）为true的情况。因为下面是conditional的第一个模板参数为false的情况下的偏特化版本，将第三个模板参数（`_Iffalse`）设置为type。



#### Is_void

``` cpp
  template<typename>
    struct __is_void_helper
    : public false_type { };

  template<>
    struct __is_void_helper<void>
    : public true_type { };

  /// is_void
  template<typename _Tp>
    struct is_void
    : public __is_void_helper<__remove_cv_t<_Tp>>::type
    { };
```



#### remove_cv

``` cpp
  /// remove_cv
  template<typename _Tp>
    struct remove_cv
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<const _Tp>
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<volatile _Tp>
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<const volatile _Tp>
    { using type = _Tp; };
```



##### \_\_detected_or

``` cpp
 /// Implementation of the detection idiom (negative case).
  template<typename _Default, typename _AlwaysVoid,
	   template<typename...> class _Op, typename... _Args>
    struct __detector
    {
      using value_t = false_type;
      using type = _Default;
    };

  /// Implementation of the detection idiom (positive case).
  template<typename _Default, template<typename...> class _Op,
	    typename... _Args>
    struct __detector<_Default, __void_t<_Op<_Args...>>, _Op, _Args...>
    {
      using value_t = true_type;
      using type = _Op<_Args...>;
    };

  // Detect whether _Op<_Args...> is a valid type, use _Default if not.
  template<typename _Default, template<typename...> class _Op,
	   typename... _Args>
    using __detected_or = __detector<_Default, void, _Op, _Args...>;

  // _Op<_Args...> if that is a valid type, otherwise _Default.
  template<typename _Default, template<typename...> class _Op,
	   typename... _Args>
    using __detected_or_t
      = typename __detected_or<_Default, _Op, _Args...>::type;
```

