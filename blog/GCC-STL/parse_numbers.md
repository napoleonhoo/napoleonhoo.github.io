# `parse_numbers`

## 1 `_select_int`

``` cpp
namespace __select_int {
template <unsigned long long _Val, typename... _Ints>
struct _Select_int_base;

template <unsigned long long _Val, typename _IntType, typename... _Ints>
struct _Select_int_base<_Val, _IntType, _Ints...>
    : conditional_t<(_Val <= __gnu_cxx::__int_traits<_IntType>::__max), integral_constant<_IntType, (_IntType)_Val>,
                    _Select_int_base<_Val, _Ints...>> {};

template <unsigned long long _Val>
struct _Select_int_base<_Val> {};

template <char... _Digs>
using _Select_int =
    typename _Select_int_base<__parse_int::_Parse_int<_Digs...>::value, unsigned char, unsigned short, unsigned int,
                              unsigned long, unsigned long long>::type;

}  // namespace __select_int
```

类`_Select_int_base`的作用是，判断`_Val`的值是否小于`_IntType`类型的最大可表示的值，如果小于的话，它的type则是`_IntType`类型的`_Val`；否则，就是使用`_Ints`类型递归调用这个类，直到找到可以满足判断条件的类。