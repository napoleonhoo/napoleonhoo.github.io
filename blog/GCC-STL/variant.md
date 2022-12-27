# variant

## 1 各式各样的helper

### 获得模板参数中的第一个参数的类型

``` cpp
template <size_t _Np, typename... _Types> struct _Nth_type;

// 形式1
template <size_t _Np, typename _First, typename... _Rest>
struct _Nth_type<_Np, _First, _Rest...> : _Nth_type<_Np - 1, _Rest...> {};

// 形式2
template <typename _First, typename... _Rest>
struct _Nth_type<0, _First, _Rest...> {
  using type = _First;
};
```

逻辑就比较简单了，举例说明。如果使用下面这种调用方式：

``` cpp
typename _Nth_type<5, _Types...>::type
```

直接调用的是特化形式1，在这个定义里面，递归地调用`_Nth_type`的形式1，并且将`_Np`这个参数-1，将`_Types`去掉第一个参数`_First`，只剩下`_Rest`。直到当`_Np`为0的时候，调用特化形式2，并将type设置为第一个参数的type。



### `_never_valueless`

``` cpp
// _Never_valueless_alt is true for variant alternatives that can
// always be placed in a variant without it becoming valueless.

// For suitably-small, trivially copyable types we can create temporaries
// on the stack and then memcpy them into place.
template <typename _Tp>
struct _Never_valueless_alt
    : __and_<bool_constant<sizeof(_Tp) <= 256>, is_trivially_copyable<_Tp>> {
};

// Specialize _Never_valueless_alt for other types which have a
// non-throwing and cheap move construction and move assignment operator,
// so that emplacing the type will provide the strong exception-safety
// guarantee, by creating and moving a temporary.
// Whether _Never_valueless_alt<T> is true or not affects the ABI of a
// variant using that alternative, so we can't change the value later!

// True if every alternative in _Types... can be emplaced in a variant
// without it becoming valueless. If this is true, variant<_Types...>
// can never be valueless, which enables some minor optimizations.
template <typename... _Types> constexpr bool __never_valueless() {
  return _Traits<_Types...>::_S_move_assign &&
         (_Never_valueless_alt<_Types>::value && ...);
}
```

当传入的类型模板参数的size小于256且可平凡拷贝的时候，`_Never_valueless_alt`为true。

当传入的模板参数是可移动构造和移动赋值的，且`_Never_valueless_alt`为true的时候，`_never_valueless`为true。



### `_Extra_visit_slot_needed`

``` cpp
template <typename _Maybe_variant_cookie, typename _Variant>
struct _Extra_visit_slot_needed {
  template <typename> struct _Variant_never_valueless;

  template <typename... _Types>
  struct _Variant_never_valueless<variant<_Types...>>
      : bool_constant<__variant::__never_valueless<_Types...>()> {};

  static constexpr bool value =
      (is_same_v<_Maybe_variant_cookie, __variant_cookie> ||
       is_same_v<
           _Maybe_variant_cookie,
           __variant_idx_cookie>)&&!_Variant_never_valueless<__remove_cvref_t<_Variant>>::
          value;
};
```

当`_Extra_visit_slot_needed`为true的时候：

- `_Maybe_variant_cookie`和`__variant_cookie`或`_variant_idx_cookie`的时候，且
- `_Variant`非`_Variant_never_valueless`的时候



### `_Multi_array`

``` cpp
// Used for storing a multi-dimensional vtable.
template <typename _Tp, size_t... _Dimensions> struct _Multi_array;

// Partial specialization with rank zero, stores a single _Tp element.
template <typename _Tp> struct _Multi_array<_Tp> {
template <typename> struct __untag_result : false_type {
  using element_type = _Tp;
};

template <typename... _Args>
struct __untag_result<const void (*)(_Args...)> : false_type {
  using element_type = void (*)(_Args...);
};

template <typename... _Args>
struct __untag_result<__variant_cookie (*)(_Args...)> : false_type {
  using element_type = void (*)(_Args...);
};

template <typename... _Args>
struct __untag_result<__variant_idx_cookie (*)(_Args...)> : false_type {
  using element_type = void (*)(_Args...);
};

template <typename _Res, typename... _Args>
struct __untag_result<__deduce_visit_result<_Res> (*)(_Args...)>
    : true_type {
  using element_type = _Res (*)(_Args...);
};

using __result_is_deduced = __untag_result<_Tp>;

constexpr const typename __untag_result<_Tp>::element_type &
_M_access() const {
  return _M_data;
}

typename __untag_result<_Tp>::element_type _M_data;
};

// Partial specialization with rank >= 1.
template <typename _Ret, typename _Visitor, typename... _Variants,
          size_t __first, size_t... __rest>
struct _Multi_array<_Ret (*)(_Visitor, _Variants...), __first, __rest...> {
  static constexpr size_t __index =
      sizeof...(_Variants) - sizeof...(__rest) - 1;

  using _Variant = typename _Nth_type<__index, _Variants...>::type;

  static constexpr int __do_cookie =
      _Extra_visit_slot_needed<_Ret, _Variant>::value ? 1 : 0;

  using _Tp = _Ret (*)(_Visitor, _Variants...);

  template <typename... _Args>
  constexpr decltype(auto) _M_access(size_t __first_index,
                                     _Args... __rest_indices) const {
    return _M_arr[__first_index + __do_cookie]._M_access(__rest_indices...);
  }

  _Multi_array<_Tp, __rest...> _M_arr[__first + __do_cookie];
};
```



### `__gen_vtable_impl`



### `__gen_vtable`

``` cpp
template <typename _Result_type, typename _Visitor, typename... _Variants>
struct __gen_vtable {
  using _Array_type =
      _Multi_array<_Result_type (*)(_Visitor, _Variants...),
                   variant_size_v<remove_reference_t<_Variants>>...>;

  static constexpr _Array_type _S_vtable =
      __gen_vtable_impl<_Array_type, std::index_sequence<>>::_S_apply();
};
```



### visit相关

``` cpp
template <typename _Result_type, typename _Visitor, typename... _Variants>
constexpr decltype(auto) __do_visit(_Visitor && __visitor,
                                    _Variants && ...__variants) {
  constexpr auto &__vtable =
      __detail::__variant::__gen_vtable<_Result_type, _Visitor &&,
                                        _Variants &&...>::_S_vtable;

  auto __func_ptr = __vtable._M_access(__variants.index()...);
  return (*__func_ptr)(std::forward<_Visitor>(__visitor),
                       std::forward<_Variants>(__variants)...);
}
```



## `variant`的base



### `_Move_assign_base`

``` cpp
template <bool, typename... _Types>
struct _Move_assign_base : _Copy_assign_alias<_Types...> {
  using _Base = _Copy_assign_alias<_Types...>;
  using _Base::_Base;

  _Move_assign_base &operator=(_Move_assign_base &&__rhs) noexcept(
      _Traits<_Types...>::_S_nothrow_move_assign) {
    __variant::__raw_idx_visit(
        [this](auto &&__rhs_mem, auto __rhs_index) mutable {
          if constexpr (__rhs_index != variant_npos) {
            if (this->_M_index == __rhs_index)
              __variant::__get<__rhs_index>(*this) = std::move(__rhs_mem);
            else
              __variant_cast<_Types...>(*this).template emplace<__rhs_index>(
                  std::move(__rhs_mem));
          } else
            this->_M_reset();
        },
        __variant_cast<_Types...>(__rhs));
    return *this;
  }

  _Move_assign_base(const _Move_assign_base &) = default;
  _Move_assign_base(_Move_assign_base &&) = default;
  _Move_assign_base &operator=(const _Move_assign_base &) = default;
};

template <typename... _Types>
struct _Move_assign_base<true, _Types...> : _Copy_assign_alias<_Types...> {
  using _Base = _Copy_assign_alias<_Types...>;
  using _Base::_Base;
};
```



### `_Variant_base`

这是`variant`的主要父类。

``` cpp
template <typename... _Types>
struct _Variant_base : _Move_assign_alias<_Types...> {
  using _Base = _Move_assign_alias<_Types...>;

  constexpr _Variant_base() noexcept(
      _Traits<_Types...>::_S_nothrow_default_ctor)
      : _Variant_base(in_place_index<0>) {}

  template <size_t _Np, typename... _Args>
  constexpr explicit _Variant_base(in_place_index_t<_Np> __i,
                                   _Args &&...__args)
      : _Base(__i, std::forward<_Args>(__args)...) {}

  _Variant_base(const _Variant_base &) = default;
  _Variant_base(_Variant_base &&) = default;
  _Variant_base &operator=(const _Variant_base &) = default;
  _Variant_base &operator=(_Variant_base &&) = default;
};
```

- 定义了两个构造函数，主要逻辑是调用`_Base(_Move_assign_alias)`的构造函数。
- 定义了拷贝构造、移动构造、拷贝赋值、移动赋值函数为default的。

其中，`_Move_assign_alias`的定义如下：

``` cpp
template <typename... _Types>
  using _Move_assign_alias =
      _Move_assign_base<_Traits<_Types...>::_S_trivial_move_assign, _Types...>;
```





# 问题

- 什么是SMF？
- `in_place_t`等相关的用法？



