# variant

## 1 variant的依赖

### `_Nth_type`：获得模板参数中的第一个参数的类型

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



### `variant_size`

``` cpp
template <typename _Variant>
struct variant_size;

template <typename _Variant>
struct variant_size<const _Variant> : variant_size<_Variant> {};

template <typename _Variant>
struct variant_size<volatile _Variant> : variant_size<_Variant> {};

template <typename _Variant>
struct variant_size<const volatile _Variant> : variant_size<_Variant> {};

template <typename... _Types>
struct variant_size<variant<_Types...>> : std::integral_constant<size_t, sizeof...(_Types)> {};

template <typename _Variant>
inline constexpr size_t variant_size_v = variant_size<_Variant>::value;
```



### `variant_alternative`

``` cpp
template <size_t _Np, typename _Variant>
struct variant_alternative;

template <size_t _Np, typename _First, typename... _Rest>
struct variant_alternative<_Np, variant<_First, _Rest...>> : variant_alternative<_Np - 1, variant<_Rest...>> {};

template <typename _First, typename... _Rest>
struct variant_alternative<0, variant<_First, _Rest...>> {
    using type = _First;
};

template <size_t _Np, typename _Variant>
using variant_alternative_t = typename variant_alternative<_Np, _Variant>::type;

template <size_t _Np, typename _Variant>
struct variant_alternative<_Np, const _Variant> {
    using type = add_const_t<variant_alternative_t<_Np, _Variant>>;
};

template <size_t _Np, typename _Variant>
struct variant_alternative<_Np, volatile _Variant> {
    using type = add_volatile_t<variant_alternative_t<_Np, _Variant>>;
};

template <size_t _Np, typename _Variant>
struct variant_alternative<_Np, const volatile _Variant> {
    using type = add_cv_t<variant_alternative_t<_Np, _Variant>>;
};

template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> &get(variant<_Types...> &);

template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> &&get(variant<_Types...> &&);

template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> const &get(const variant<_Types...> &);

template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> const &&get(const variant<_Types...> &&);
```



### `variant_npos`

``` cpp
inline constexpr size_t variant_npos = -1;
```



### `__variant_cast`

``` cpp
template <typename... _Types, typename _Tp>
decltype(auto) __variant_cast(_Tp && __rhs) {
    if constexpr (is_lvalue_reference_v<_Tp>) {
        if constexpr (is_const_v<remove_reference_t<_Tp>>)
            return static_cast<const variant<_Types...> &>(__rhs);
        else
            return static_cast<variant<_Types...> &>(__rhs);
    } else
        return static_cast<variant<_Types...> &&>(__rhs);
}
```





### `__do_visit`

``` cpp
template <typename _Result_type, typename _Visitor, typename... _Variants>
constexpr decltype(auto) __do_visit(_Visitor && __visitor, _Variants && ...__variants) {
    constexpr auto &__vtable =
        __detail::__variant::__gen_vtable<_Result_type, _Visitor &&, _Variants &&...>::_S_vtable;

    auto __func_ptr = __vtable._M_access(__variants.index()...);
    return (*__func_ptr)(std::forward<_Visitor>(__visitor), std::forward<_Variants>(__variants)...);
}
```



### `index_of`

``` cpp
// Returns the first appearance of _Tp in _Types.
// Returns sizeof...(_Types) if _Tp is not in _Types.
template <typename _Tp, typename... _Types>
struct __index_of : std::integral_constant<size_t, 0> {};

template <typename _Tp, typename... _Types>
inline constexpr size_t __index_of_v = __index_of<_Tp, _Types...>::value;

template <typename _Tp, typename _First, typename... _Rest>
struct __index_of<_Tp, _First, _Rest...>
    : std::integral_constant<size_t, is_same_v<_Tp, _First> ? 0 : __index_of_v<_Tp, _Rest...> + 1> {};
```



### `__variant_cookie`

``` cpp
// used for raw visitation
struct __variant_cookie {};
// used for raw visitation with indices passed in
struct __variant_idx_cookie {
    using type = __variant_idx_cookie;
};
```



### `__raw_visit`

``` cpp
// Visit variants that might be valueless.
template <typename _Visitor, typename... _Variants>
constexpr void __raw_visit(_Visitor &&__visitor, _Variants &&...__variants) {
    std::__do_visit<__variant_cookie>(std::forward<_Visitor>(__visitor), std::forward<_Variants>(__variants)...);
}

// Visit variants that might be valueless, passing indices to the visitor.
template <typename _Visitor, typename... _Variants>
constexpr void __raw_idx_visit(_Visitor &&__visitor, _Variants &&...__variants) {
    std::__do_visit<__variant_idx_cookie>(std::forward<_Visitor>(__visitor),
                                          std::forward<_Variants>(__variants)...);
}
```



### `_Unintialized`

``` cpp
// _Uninitialized<T> is guaranteed to be a trivially destructible type,
// even if T is not.
template <typename _Type, bool = std::is_trivially_destructible_v<_Type>>
struct _Uninitialized;

template <typename _Type>
struct _Uninitialized<_Type, true> {
    template <typename... _Args>
    constexpr _Uninitialized(in_place_index_t<0>, _Args &&...__args) : _M_storage(std::forward<_Args>(__args)...) {}

    constexpr const _Type &_M_get() const &noexcept { return _M_storage; }

    constexpr _Type &_M_get() &noexcept { return _M_storage; }

    constexpr const _Type &&_M_get() const &&noexcept { return std::move(_M_storage); }

    constexpr _Type &&_M_get() &&noexcept { return std::move(_M_storage); }

    _Type _M_storage;
};

template <typename _Type>
struct _Uninitialized<_Type, false> {
    template <typename... _Args>
    constexpr _Uninitialized(in_place_index_t<0>, _Args &&...__args) {
        ::new ((void *)std::addressof(_M_storage)) _Type(std::forward<_Args>(__args)...);
    }

    const _Type &_M_get() const &noexcept { return *_M_storage._M_ptr(); }

    _Type &_M_get() &noexcept { return *_M_storage._M_ptr(); }

    const _Type &&_M_get() const &&noexcept { return std::move(*_M_storage._M_ptr()); }

    _Type &&_M_get() &&noexcept { return std::move(*_M_storage._M_ptr()); }

    __gnu_cxx::__aligned_membuf<_Type> _M_storage;
};
```

两个特化的版本区别是bool类型的模板参数`is_trivially_destructible_v`。当其为true，即可以被平凡析构的时候，`_M_storage`是`_Type`类型的参数，直接调用构造函数；`_M_get()`相关的函数返回的是这个类型的变量。当其为false的时候，`_M_storage`为一段内存buf，要使用placement new来构造；并且`_M_get()`相关的函数返回的是`_M_storage`的指针。



### `__get`

``` cpp
template <typename _Union>
constexpr decltype(auto) __get(in_place_index_t<0>, _Union &&__u) noexcept {
    return std::forward<_Union>(__u)._M_first._M_get();
}

template <size_t _Np, typename _Union>
constexpr decltype(auto) __get(in_place_index_t<_Np>, _Union &&__u) noexcept {
    return __variant::__get(in_place_index<_Np - 1>, std::forward<_Union>(__u)._M_rest);
}

// Returns the typed storage for __v.
template <size_t _Np, typename _Variant>
constexpr decltype(auto) __get(_Variant &&__v) noexcept {
    return __variant::__get(std::in_place_index<_Np>, std::forward<_Variant>(__v)._M_u);
}
```



### `_Variadic_union`

``` cpp
// Defines members and ctors.
template <typename... _Types>
union _Variadic_union {};

template <typename _First, typename... _Rest>
union _Variadic_union<_First, _Rest...> {
    constexpr _Variadic_union() : _M_rest() {}

    template <typename... _Args>
    constexpr _Variadic_union(in_place_index_t<0>, _Args &&...__args)
        : _M_first(in_place_index<0>, std::forward<_Args>(__args)...) {}

    template <size_t _Np, typename... _Args>
    constexpr _Variadic_union(in_place_index_t<_Np>, _Args &&...__args)
        : _M_rest(in_place_index<_Np - 1>, std::forward<_Args>(__args)...) {}

    _Uninitialized<_First> _M_first;
    _Variadic_union<_Rest...> _M_rest;
};
```



### `_Never_valueless`

``` cpp
// _Never_valueless_alt is true for variant alternatives that can
// always be placed in a variant without it becoming valueless.

// For suitably-small, trivially copyable types we can create temporaries
// on the stack and then memcpy them into place.
template <typename _Tp>
struct _Never_valueless_alt : __and_<bool_constant<sizeof(_Tp) <= 256>, is_trivially_copyable<_Tp>> {};

// Specialize _Never_valueless_alt for other types which have a
// non-throwing and cheap move construction and move assignment operator,
// so that emplacing the type will provide the strong exception-safety
// guarantee, by creating and moving a temporary.
// Whether _Never_valueless_alt<T> is true or not affects the ABI of a
// variant using that alternative, so we can't change the value later!

// True if every alternative in _Types... can be emplaced in a variant
// without it becoming valueless. If this is true, variant<_Types...>
// can never be valueless, which enables some minor optimizations.
template <typename... _Types>
constexpr bool __never_valueless() {
    return _Traits<_Types...>::_S_move_assign && (_Never_valueless_alt<_Types>::value && ...);
}
```

`_Never_valueless_alt`为true，当：

- `sizeof _Tp`小于等于256；
- `_Tp`可以被平凡拷贝。

说明：当某个类型足够小的时候，可以被放入variant里面，它可以在栈上创建一个临时变量，并**拷贝**其到地方。

`__never_valueless`为true，当：

- `_Types...`可以移动赋值；
- `_Never_valueless_alt`为true。

说明：当某个类型足够小的时候，可以被放入variant里面，它可以在栈上创建一个临时变量，并**移动**其到地方。



### `_Variant_storage`

``` cpp
// Defines index and the dtor, possibly trivial.
template <bool __trivially_destructible, typename... _Types>
struct _Variant_storage;

template <typename... _Types>
using __select_index =
    typename __select_int::_Select_int_base<sizeof...(_Types), unsigned char, unsigned short>::type::value_type;

template <typename... _Types>
struct _Variant_storage<false, _Types...> {
    constexpr _Variant_storage() : _M_index(static_cast<__index_type>(variant_npos)) {}

    template <size_t _Np, typename... _Args>
    constexpr _Variant_storage(in_place_index_t<_Np>, _Args &&...__args)
        : _M_u(in_place_index<_Np>, std::forward<_Args>(__args)...), _M_index{_Np} {}

    void _M_reset() {
        if (!_M_valid()) [[unlikely]]
            return;

        std::__do_visit<void>([](auto &&__this_mem) mutable { std::_Destroy(std::__addressof(__this_mem)); },
                              __variant_cast<_Types...>(*this));

        _M_index = static_cast<__index_type>(variant_npos);
    }

    ~_Variant_storage() { _M_reset(); }

    void *_M_storage() const noexcept {
        return const_cast<void *>(static_cast<const void *>(std::addressof(_M_u)));
    }

    constexpr bool _M_valid() const noexcept {
        if constexpr (__variant::__never_valueless<_Types...>()) return true;
        return this->_M_index != __index_type(variant_npos);
    }

    _Variadic_union<_Types...> _M_u;
    using __index_type = __select_index<_Types...>;
    __index_type _M_index;
};

template <typename... _Types>
struct _Variant_storage<true, _Types...> {
    constexpr _Variant_storage() : _M_index(static_cast<__index_type>(variant_npos)) {}

    template <size_t _Np, typename... _Args>
    constexpr _Variant_storage(in_place_index_t<_Np>, _Args &&...__args)
        : _M_u(in_place_index<_Np>, std::forward<_Args>(__args)...), _M_index{_Np} {}

    void _M_reset() noexcept { _M_index = static_cast<__index_type>(variant_npos); }

    void *_M_storage() const noexcept {
        return const_cast<void *>(static_cast<const void *>(std::addressof(_M_u)));
    }

    constexpr bool _M_valid() const noexcept {
        if constexpr (__variant::__never_valueless<_Types...>()) return true;
        return this->_M_index != static_cast<__index_type>(variant_npos);
    }

    _Variadic_union<_Types...> _M_u;
    using __index_type = __select_index<_Types...>;
    __index_type _M_index;
};

template <typename... _Types>
using _Variant_storage_alias = _Variant_storage<_Traits<_Types...>::_S_trivial_dtor, _Types...>;
```

注意这里的`__select_index`的定义，它的意义是，当unsigned char可以保存`sizeof _Types`的值时，使用unsigned char。如果不行的话，使用unsigned short。

`_Variant_storage`的实际存储数据的地方在`_Variadic_unon`类型。

两个偏特化的版本区别在于是否可以被平凡析构。当其为false的时候，`_M_reset`需要遍历每一个元素，并调用`std::_Destroy`，并设置` _M_index`为`variant_npos`；当为true的时候，`_M_reset`只需设置`_M_index`。

特别要说明的是`_M_valid`函数，两种特化版本的这个函数实现是一样的。主要逻辑：

- 当判断`__never_valueless`为true的时候，返回true。否则，
- 返回当前的`_M_index`是否为`variant_pos`。



### `__variant_construct`

``` cpp
template <typename _Tp, typename _Up>
void __variant_construct_single(_Tp &&__lhs, _Up &&__rhs_mem) {
    void *__storage = std::addressof(__lhs._M_u);
    using _Type = remove_reference_t<decltype(__rhs_mem)>;
    if constexpr (!is_same_v<_Type, __variant_cookie>)
        ::new (__storage) _Type(std::forward<decltype(__rhs_mem)>(__rhs_mem));
}

template <typename... _Types, typename _Tp, typename _Up>
void __variant_construct(_Tp &&__lhs, _Up &&__rhs) {
    __lhs._M_index = __rhs._M_index;
    __variant::__raw_visit(
        [&__lhs](auto &&__rhs_mem) mutable {
            __variant_construct_single(std::forward<_Tp>(__lhs), std::forward<decltype(__rhs_mem)>(__rhs_mem));
        },
        __variant_cast<_Types...>(std::forward<_Up>(__rhs)));
}
```

`__variant_construct_single`的使用方法中，将rhs复制到lhs：复制`_M_index`，并在lhs的存储单元上使用rhs来初始化内存。它是创建单个对象的函数。

`__variant_construct`创建多个对象。



### 拷贝|移动 构造函数|赋值函数 base

``` cpp
// The following are (Copy|Move) (ctor|assign) layers for forwarding
// triviality and handling non-trivial SMF behaviors.

template <bool, typename... _Types>
struct _Copy_ctor_base : _Variant_storage_alias<_Types...> {
    using _Base = _Variant_storage_alias<_Types...>;
    using _Base::_Base;

    _Copy_ctor_base(const _Copy_ctor_base &__rhs) noexcept(_Traits<_Types...>::_S_nothrow_copy_ctor) {
        __variant_construct<_Types...>(*this, __rhs);
    }

    _Copy_ctor_base(_Copy_ctor_base &&) = default;
    _Copy_ctor_base &operator=(const _Copy_ctor_base &) = default;
    _Copy_ctor_base &operator=(_Copy_ctor_base &&) = default;
};

template <typename... _Types>
struct _Copy_ctor_base<true, _Types...> : _Variant_storage_alias<_Types...> {
    using _Base = _Variant_storage_alias<_Types...>;
    using _Base::_Base;
};

template <typename... _Types>
using _Copy_ctor_alias = _Copy_ctor_base<_Traits<_Types...>::_S_trivial_copy_ctor, _Types...>;

template <bool, typename... _Types>
struct _Move_ctor_base : _Copy_ctor_alias<_Types...> {
    using _Base = _Copy_ctor_alias<_Types...>;
    using _Base::_Base;

    _Move_ctor_base(_Move_ctor_base &&__rhs) noexcept(_Traits<_Types...>::_S_nothrow_move_ctor) {
        __variant_construct<_Types...>(*this, std::move(__rhs));
    }

    template <typename _Up>
    void _M_destructive_move(unsigned short __rhs_index, _Up &&__rhs) {
        this->_M_reset();
        __variant_construct_single(*this, std::forward<_Up>(__rhs));
        this->_M_index = __rhs_index;
    }

    template <typename _Up>
    void _M_destructive_copy(unsigned short __rhs_index, const _Up &__rhs) {
        this->_M_reset();
        __variant_construct_single(*this, __rhs);
        this->_M_index = __rhs_index;
    }

    _Move_ctor_base(const _Move_ctor_base &) = default;
    _Move_ctor_base &operator=(const _Move_ctor_base &) = default;
    _Move_ctor_base &operator=(_Move_ctor_base &&) = default;
};

template <typename... _Types>
struct _Move_ctor_base<true, _Types...> : _Copy_ctor_alias<_Types...> {
    using _Base = _Copy_ctor_alias<_Types...>;
    using _Base::_Base;

    template <typename _Up>
    void _M_destructive_move(unsigned short __rhs_index, _Up &&__rhs) {
        this->_M_reset();
        __variant_construct_single(*this, std::forward<_Up>(__rhs));
        this->_M_index = __rhs_index;
    }

    template <typename _Up>
    void _M_destructive_copy(unsigned short __rhs_index, const _Up &__rhs) {
        this->_M_reset();
        __variant_construct_single(*this, __rhs);
        this->_M_index = __rhs_index;
    }
};

template <typename... _Types>
using _Move_ctor_alias = _Move_ctor_base<_Traits<_Types...>::_S_trivial_move_ctor, _Types...>;

template <bool, typename... _Types>
struct _Copy_assign_base : _Move_ctor_alias<_Types...> {
    using _Base = _Move_ctor_alias<_Types...>;
    using _Base::_Base;

    _Copy_assign_base &operator=(const _Copy_assign_base &__rhs) noexcept(
        _Traits<_Types...>::_S_nothrow_copy_assign) {
        __variant::__raw_idx_visit(
            [this](auto &&__rhs_mem, auto __rhs_index) mutable {
                if constexpr (__rhs_index != variant_npos) {
                    if (this->_M_index == __rhs_index)
                        __variant::__get<__rhs_index>(*this) = __rhs_mem;
                    else {
                        using __rhs_type = __remove_cvref_t<decltype(__rhs_mem)>;
                        if constexpr (is_nothrow_copy_constructible_v<__rhs_type> ||
                                      !is_nothrow_move_constructible_v<__rhs_type>)
                            // The standard says this->emplace<__rhs_type>(__rhs_mem)
                            // should be used here, but _M_destructive_copy is
                            // equivalent in this case. Either copy construction
                            // doesn't throw, so _M_destructive_copy gives strong
                            // exception safety guarantee, or both copy construction
                            // and move construction can throw, so emplace only gives
                            // basic exception safety anyway.
                            this->_M_destructive_copy(__rhs_index, __rhs_mem);
                        else
                            __variant_cast<_Types...>(*this) =
                                variant<_Types...>(std::in_place_index<__rhs_index>, __rhs_mem);
                    }
                } else
                    this->_M_reset();
            },
            __variant_cast<_Types...>(__rhs));
        return *this;
    }

    _Copy_assign_base(const _Copy_assign_base &) = default;
    _Copy_assign_base(_Copy_assign_base &&) = default;
    _Copy_assign_base &operator=(_Copy_assign_base &&) = default;
};

template <typename... _Types>
struct _Copy_assign_base<true, _Types...> : _Move_ctor_alias<_Types...> {
    using _Base = _Move_ctor_alias<_Types...>;
    using _Base::_Base;
};

template <typename... _Types>
using _Copy_assign_alias = _Copy_assign_base<_Traits<_Types...>::_S_trivial_copy_assign, _Types...>;

template <bool, typename... _Types>
struct _Move_assign_base : _Copy_assign_alias<_Types...> {
    using _Base = _Copy_assign_alias<_Types...>;
    using _Base::_Base;

    _Move_assign_base &operator=(_Move_assign_base &&__rhs) noexcept(_Traits<_Types...>::_S_nothrow_move_assign) {
        __variant::__raw_idx_visit(
            [this](auto &&__rhs_mem, auto __rhs_index) mutable {
                if constexpr (__rhs_index != variant_npos) {
                    if (this->_M_index == __rhs_index)
                        __variant::__get<__rhs_index>(*this) = std::move(__rhs_mem);
                    else
                        __variant_cast<_Types...>(*this).template emplace<__rhs_index>(std::move(__rhs_mem));
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

拷贝构造函数base，继承自`_Variant_storage`。主要函数就是一个拷贝构造函数，它调用`__variant_construct`。

移动构造函数base，继承自拷贝构造函数base。对第一个参数进行了偏特化：

- 普通版本下，主要函数就是一个移动构造函数，它调用`__variant_construct`。还有函数`_M_destructive_move`和`_M_destructive_copy`。它先是销毁了自己持有的数据，再调用`__variant_construct_single`来创建。
- 第一个模板参数为true的时候的偏特化版本，它的移动构造函数是默认的。

拷贝赋值函数base，继承自移动构造函数base。主要函数就是一个拷贝赋值函数，它拷贝了rhs。注意`_M_destructive_copy`的使用。

移动赋值函数base，继承自拷贝赋值函数base。主要函数就是一个移动赋值函数，它将rhs移动到this。注意这里面使用了`emplace`函数。



### variant的base class：

``` cpp
template <typename... _Types>
struct _Variant_base : _Move_assign_alias<_Types...> {
    using _Base = _Move_assign_alias<_Types...>;

    constexpr _Variant_base() noexcept(_Traits<_Types...>::_S_nothrow_default_ctor)
        : _Variant_base(in_place_index<0>) {}

    template <size_t _Np, typename... _Args>
    constexpr explicit _Variant_base(in_place_index_t<_Np> __i, _Args &&...__args)
        : _Base(__i, std::forward<_Args>(__args)...) {}

    _Variant_base(const _Variant_base &) = default;
    _Variant_base(_Variant_base &&) = default;
    _Variant_base &operator=(const _Variant_base &) = default;
    _Variant_base &operator=(_Variant_base &&) = default;
};
```

它直接继承自移动赋值函数base，里面主要有两个构造函数，顺着继承图一直向上找，会发现，它调用的是`_Variant_storage`的构造函数。



### `tuple_count`

``` cpp
// For how many times does _Tp appear in _Tuple?
template <typename _Tp, typename _Tuple>
struct __tuple_count;

template <typename _Tp, typename _Tuple>
inline constexpr size_t __tuple_count_v = __tuple_count<_Tp, _Tuple>::value;

template <typename _Tp, typename... _Types>
struct __tuple_count<_Tp, tuple<_Types...>> : integral_constant<size_t, 0> {};

template <typename _Tp, typename _First, typename... _Rest>
struct __tuple_count<_Tp, tuple<_First, _Rest...>>
    : integral_constant<size_t, __tuple_count_v<_Tp, tuple<_Rest...>> + is_same_v<_Tp, _First>> {};

// TODO: Reuse this in <tuple> ?
template <typename _Tp, typename... _Types>
inline constexpr bool __exactly_once = __tuple_count_v<_Tp, tuple<_Types...>> == 1;
```

`__tuple_count`这个模板获得`_Tp`类型在tuple中出现的次数。



### `Build_FUN`

``` cpp
// Helper used to check for valid conversions that don't involve narrowing.
template <typename _Ti>
struct _Arr {
    _Ti _M_x[1];
};

// "Build an imaginary function FUN(Ti) for each alternative type Ti"
template <size_t _Ind, typename _Tp, typename _Ti, typename = void>
struct _Build_FUN {
    // This function means 'using _Build_FUN<I, T, Ti>::_S_fun;' is valid,
    // but only static functions will be considered in the call below.
    void _S_fun();
};

// "... for which Ti x[] = {std::forward<T>(t)}; is well-formed."
template <size_t _Ind, typename _Tp, typename _Ti>
struct _Build_FUN<_Ind, _Tp, _Ti, void_t<decltype(_Arr<_Ti>{{std::declval<_Tp>()}})>> {
    // This is the FUN function for type _Ti, with index _Ind
    static integral_constant<size_t, _Ind> _S_fun(_Ti);
};

template <typename _Tp, typename _Variant, typename = make_index_sequence<variant_size_v<_Variant>>>
struct _Build_FUNs;

template <typename _Tp, typename... _Ti, size_t... _Ind>
struct _Build_FUNs<_Tp, variant<_Ti...>, index_sequence<_Ind...>> : _Build_FUN<_Ind, _Tp, _Ti>... {
    using _Build_FUN<_Ind, _Tp, _Ti>::_S_fun...;
};

// The index j of the overload FUN(Tj) selected by overload resolution
// for FUN(std::forward<_Tp>(t))
template <typename _Tp, typename _Variant>
using _FUN_type = decltype(_Build_FUNs<_Tp, _Variant>::_S_fun(std::declval<_Tp>()));
template <typename _Tp, typename _Variant, typename = void>
struct __accepted_index : integral_constant<size_t, variant_npos> {};

template <typename _Tp, typename _Variant>
struct __accepted_index<_Tp, _Variant, void_t<_FUN_type<_Tp, _Variant>>> : _FUN_type<_Tp, _Variant> {};
```



### `__get_storage`

``` cpp
// Returns the raw storage for __v.
template <typename _Variant>
void *__get_storage(_Variant &&__v) noexcept {
    return __v._M_storage();
}
```

返回variant类型实际的storage。



### `_Extra_visit_slot_needed`

``` cpp
template <typename _Maybe_variant_cookie, typename _Variant>
struct _Extra_visit_slot_needed {
    template <typename>
    struct _Variant_never_valueless;

    template <typename... _Types>
    struct _Variant_never_valueless<variant<_Types...>> : bool_constant<__variant::__never_valueless<_Types...>()> {
    };

    static constexpr bool value =
        (is_same_v<_Maybe_variant_cookie, __variant_cookie> ||
         is_same_v<_Maybe_variant_cookie,
                   __variant_idx_cookie>)&&!_Variant_never_valueless<__remove_cvref_t<_Variant>>::value;
};
```



### `_Multi_array`

``` cpp
// Used for storing a multi-dimensional vtable.
template <typename _Tp, size_t... _Dimensions>
struct _Multi_array;

// Partial specialization with rank zero, stores a single _Tp element.
template <typename _Tp>
struct _Multi_array<_Tp> {
    template <typename>
    struct __untag_result : false_type {
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
    struct __untag_result<__deduce_visit_result<_Res> (*)(_Args...)> : true_type {
        using element_type = _Res (*)(_Args...);
    };

    using __result_is_deduced = __untag_result<_Tp>;

    constexpr const typename __untag_result<_Tp>::element_type &_M_access() const { return _M_data; }

    typename __untag_result<_Tp>::element_type _M_data;
};

// Partial specialization with rank >= 1.
template <typename _Ret, typename _Visitor, typename... _Variants, size_t __first, size_t... __rest>
struct _Multi_array<_Ret (*)(_Visitor, _Variants...), __first, __rest...> {
    static constexpr size_t __index = sizeof...(_Variants) - sizeof...(__rest) - 1;

    using _Variant = typename _Nth_type<__index, _Variants...>::type;

    static constexpr int __do_cookie = _Extra_visit_slot_needed<_Ret, _Variant>::value ? 1 : 0;

    using _Tp = _Ret (*)(_Visitor, _Variants...);

    template <typename... _Args>
    constexpr decltype(auto) _M_access(size_t __first_index, _Args... __rest_indices) const {
        return _M_arr[__first_index + __do_cookie]._M_access(__rest_indices...);
    }

    _Multi_array<_Tp, __rest...> _M_arr[__first + __do_cookie];
};
```



### `__gen_vtable`

``` cpp
// Creates a multi-dimensional vtable recursively.
//
// For example,
// visit([](auto, auto){},
//       variant<int, char>(),  // typedef'ed as V1
//       variant<float, double, long double>())  // typedef'ed as V2
// will trigger instantiations of:
// __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 2, 3>,
//                   tuple<V1&&, V2&&>, std::index_sequence<>>
//   __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 3>,
//                     tuple<V1&&, V2&&>, std::index_sequence<0>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<0, 0>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<0, 1>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<0, 2>>
//   __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&), 3>,
//                     tuple<V1&&, V2&&>, std::index_sequence<1>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<1, 0>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<1, 1>>
//     __gen_vtable_impl<_Multi_array<void(*)(V1&&, V2&&)>,
//                       tuple<V1&&, V2&&>, std::index_sequence<1, 2>>
// The returned multi-dimensional vtable can be fast accessed by the visitor
// using index calculation.
template <typename _Array_type, typename _Index_seq>
struct __gen_vtable_impl;

// Defines the _S_apply() member that returns a _Multi_array populated
// with function pointers that perform the visitation expressions e(m)
// for each valid pack of indexes into the variant types _Variants.
//
// This partial specialization builds up the index sequences by recursively
// calling _S_apply() on the next specialization of __gen_vtable_impl.
// The base case of the recursion defines the actual function pointers.
template <typename _Result_type, typename _Visitor, size_t... __dimensions, typename... _Variants,
          size_t... __indices>
struct __gen_vtable_impl<_Multi_array<_Result_type (*)(_Visitor, _Variants...), __dimensions...>,
                         std::index_sequence<__indices...>> {
    using _Next = remove_reference_t<typename _Nth_type<sizeof...(__indices), _Variants...>::type>;
    using _Array_type = _Multi_array<_Result_type (*)(_Visitor, _Variants...), __dimensions...>;

    static constexpr _Array_type _S_apply() {
        _Array_type __vtable{};
        _S_apply_all_alts(__vtable, make_index_sequence<variant_size_v<_Next>>());
        return __vtable;
    }

    template <size_t... __var_indices>
    static constexpr void _S_apply_all_alts(_Array_type &__vtable, std::index_sequence<__var_indices...>) {
        if constexpr (_Extra_visit_slot_needed<_Result_type, _Next>::value)
            (_S_apply_single_alt<true, __var_indices>(__vtable._M_arr[__var_indices + 1], &(__vtable._M_arr[0])),
             ...);
        else
            (_S_apply_single_alt<false, __var_indices>(__vtable._M_arr[__var_indices]), ...);
    }

    template <bool __do_cookie, size_t __index, typename _Tp>
    static constexpr void _S_apply_single_alt(_Tp &__element, _Tp *__cookie_element = nullptr) {
        if constexpr (__do_cookie) {
            __element = __gen_vtable_impl<_Tp, std::index_sequence<__indices..., __index>>::_S_apply();
            *__cookie_element = __gen_vtable_impl<_Tp, std::index_sequence<__indices..., variant_npos>>::_S_apply();
        } else {
            auto __tmp_element = __gen_vtable_impl<remove_reference_t<decltype(__element)>,
                                                   std::index_sequence<__indices..., __index>>::_S_apply();
            static_assert(is_same_v<_Tp, decltype(__tmp_element)>,
                          "std::visit requires the visitor to have the same "
                          "return type for all alternatives of a variant");
            __element = __tmp_element;
        }
    }
};

// This partial specialization is the base case for the recursion.
// It populates a _Multi_array element with the address of a function
// that invokes the visitor with the alternatives specified by __indices.
template <typename _Result_type, typename _Visitor, typename... _Variants, size_t... __indices>
struct __gen_vtable_impl<_Multi_array<_Result_type (*)(_Visitor, _Variants...)>,
                         std::index_sequence<__indices...>> {
    using _Array_type = _Multi_array<_Result_type (*)(_Visitor, _Variants...)>;

    template <size_t __index, typename _Variant>
    static constexpr decltype(auto) __element_by_index_or_cookie(_Variant &&__var) noexcept {
        if constexpr (__index != variant_npos)
            return __variant::__get<__index>(std::forward<_Variant>(__var));
        else
            return __variant_cookie{};
    }

    static constexpr decltype(auto) __visit_invoke(_Visitor &&__visitor, _Variants... __vars) {
        if constexpr (is_same_v<_Result_type, __variant_idx_cookie>)
            // For raw visitation using indices, pass the indices to the visitor
            // and discard the return value:
            std::__invoke(std::forward<_Visitor>(__visitor),
                          __element_by_index_or_cookie<__indices>(std::forward<_Variants>(__vars))...,
                          integral_constant<size_t, __indices>()...);
        else if constexpr (is_same_v<_Result_type, __variant_cookie>)
            // For raw visitation without indices, and discard the return value:
            std::__invoke(std::forward<_Visitor>(__visitor),
                          __element_by_index_or_cookie<__indices>(std::forward<_Variants>(__vars))...);
        else if constexpr (_Array_type::__result_is_deduced::value)
            // For the usual std::visit case deduce the return value:
            return std::__invoke(std::forward<_Visitor>(__visitor),
                                 __element_by_index_or_cookie<__indices>(std::forward<_Variants>(__vars))...);
        else  // for std::visit<R> use INVOKE<R>
            return std::__invoke_r<_Result_type>(std::forward<_Visitor>(__visitor),
                                                 __variant::__get<__indices>(std::forward<_Variants>(__vars))...);
    }

    static constexpr auto _S_apply() {
        if constexpr (_Array_type::__result_is_deduced::value) {
            constexpr bool __visit_ret_type_mismatch =
                !is_same_v<typename _Result_type::type,
                           decltype(__visit_invoke(std::declval<_Visitor>(), std::declval<_Variants>()...))>;
            if constexpr (__visit_ret_type_mismatch) {
                struct __cannot_match {};
                return __cannot_match{};
            } else
                return _Array_type{&__visit_invoke};
        } else
            return _Array_type{&__visit_invoke};
    }
};

template <typename _Result_type, typename _Visitor, typename... _Variants>
struct __gen_vtable {
    using _Array_type =
        _Multi_array<_Result_type (*)(_Visitor, _Variants...), variant_size_v<remove_reference_t<_Variants>>...>;

    static constexpr _Array_type _S_vtable = __gen_vtable_impl<_Array_type, std::index_sequence<>>::_S_apply();
};
```



### `_Variant_hash_base`

``` cpp
template <size_t _Np, typename _Tp>
struct _Base_dedup : public _Tp {};

template <typename _Variant, typename __indices>
struct _Variant_hash_base;

template <typename... _Types, size_t... __indices>
struct _Variant_hash_base<variant<_Types...>, std::index_sequence<__indices...>>
    : _Base_dedup<__indices, __poison_hash<remove_const_t<_Types>>>... {};
```



### `__variant_construct_by_index`

``` cpp
template <size_t _Np, typename _Variant, typename... _Args>
void __variant_construct_by_index(_Variant & __v, _Args && ...__args) {
    __v._M_index = _Np;
    auto &&__storage = __detail::__variant::__get<_Np>(__v);
    ::new ((void *)std::addressof(__storage))
        remove_reference_t<decltype(__storage)>(std::forward<_Args>(__args)...);
}
```



### `holds_alternative`

``` cpp
template <typename _Tp, typename... _Types>
constexpr bool holds_alternative(const variant<_Types...> &__v) noexcept {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    return __v.index() == __detail::__variant::__index_of_v<_Tp, _Types...>;
}
```



### `get`

``` cpp
template <typename _Tp, typename... _Types>
constexpr _Tp &get(variant<_Types...> & __v) {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(__v);
}

template <typename _Tp, typename... _Types>
constexpr _Tp &&get(variant<_Types...> && __v) {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(std::move(__v));
}

template <typename _Tp, typename... _Types>
constexpr const _Tp &get(const variant<_Types...> &__v) {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(__v);
}

template <typename _Tp, typename... _Types>
constexpr const _Tp &&get(const variant<_Types...> &&__v) {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get<__detail::__variant::__index_of_v<_Tp, _Types...>>(std::move(__v));
}
```



### `add_pointer_t`

``` cpp
template <size_t _Np, typename... _Types>
constexpr add_pointer_t<variant_alternative_t<_Np, variant<_Types...>>> get_if(variant<_Types...> *
                                                                               __ptr) noexcept {
    using _Alternative_type = variant_alternative_t<_Np, variant<_Types...>>;
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    static_assert(!is_void_v<_Alternative_type>, "_Tp must not be void");
    if (__ptr && __ptr->index() == _Np) return std::addressof(__detail::__variant::__get<_Np>(*__ptr));
    return nullptr;
}

template <size_t _Np, typename... _Types>
constexpr add_pointer_t<const variant_alternative_t<_Np, variant<_Types...>>> get_if(
    const variant<_Types...> *__ptr) noexcept {
    using _Alternative_type = variant_alternative_t<_Np, variant<_Types...>>;
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    static_assert(!is_void_v<_Alternative_type>, "_Tp must not be void");
    if (__ptr && __ptr->index() == _Np) return std::addressof(__detail::__variant::__get<_Np>(*__ptr));
    return nullptr;
}

template <typename _Tp, typename... _Types>
constexpr add_pointer_t<_Tp> get_if(variant<_Types...> * __ptr) noexcept {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get_if<__detail::__variant::__index_of_v<_Tp, _Types...>>(__ptr);
}

template <typename _Tp, typename... _Types>
constexpr add_pointer_t<const _Tp> get_if(const variant<_Types...> *__ptr) noexcept {
    static_assert(__detail::__variant::__exactly_once<_Tp, _Types...>, "T must occur exactly once in alternatives");
    static_assert(!is_void_v<_Tp>, "_Tp must not be void");
    return std::get_if<__detail::__variant::__index_of_v<_Tp, _Types...>>(__ptr);
}
```



### `monostate`

``` cpp
    struct monostate {};

#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP, __NAME)                                            \
    template <typename... _Types>                                                                    \
    constexpr bool operator __OP(const variant<_Types...> &__lhs, const variant<_Types...> &__rhs) { \
        bool __ret = true;                                                                           \
        __detail::__variant::__raw_idx_visit(                                                        \
            [&__ret, &__lhs](auto &&__rhs_mem, auto __rhs_index) mutable {                           \
                if constexpr (__rhs_index != variant_npos) {                                         \
                    if (__lhs.index() == __rhs_index) {                                              \
                        auto &__this_mem = std::get<__rhs_index>(__lhs);                             \
                        __ret = __this_mem __OP __rhs_mem;                                           \
                    } else                                                                           \
                        __ret = (__lhs.index() + 1) __OP(__rhs_index + 1);                           \
                } else                                                                               \
                    __ret = (__lhs.index() + 1) __OP(__rhs_index + 1);                               \
            },                                                                                       \
            __rhs);                                                                                  \
        return __ret;                                                                                \
    }

    _VARIANT_RELATION_FUNCTION_TEMPLATE(<, less)
    _VARIANT_RELATION_FUNCTION_TEMPLATE(<=, less_equal)
    _VARIANT_RELATION_FUNCTION_TEMPLATE(==, equal)
    _VARIANT_RELATION_FUNCTION_TEMPLATE(!=, not_equal)
    _VARIANT_RELATION_FUNCTION_TEMPLATE(>=, greater_equal)
    _VARIANT_RELATION_FUNCTION_TEMPLATE(>, greater)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE

    constexpr bool operator==(monostate, monostate) noexcept { return true; }

#ifdef __cpp_lib_three_way_comparison
    template <typename... _Types>
    requires(three_way_comparable<_Types> &&
             ...) constexpr common_comparison_category_t<compare_three_way_result_t<_Types>...>
    operator<=>(const variant<_Types...> &__v, const variant<_Types...> &__w) {
        common_comparison_category_t<compare_three_way_result_t<_Types>...> __ret = strong_ordering::equal;

        __detail::__variant::__raw_idx_visit(
            [&__ret, &__v](auto &&__w_mem, auto __w_index) mutable {
                if constexpr (__w_index != variant_npos) {
                    if (__v.index() == __w_index) {
                        auto &__this_mem = std::get<__w_index>(__v);
                        __ret = __this_mem <=> __w_mem;
                        return;
                    }
                }
                __ret = (__v.index() + 1) <=> (__w_index + 1);
            },
            __w);
        return __ret;
    }

    constexpr strong_ordering operator<=>(monostate, monostate) noexcept { return strong_ordering::equal; }
#else
    constexpr bool operator!=(monostate, monostate) noexcept { return false; }
    constexpr bool operator<(monostate, monostate) noexcept { return false; }
    constexpr bool operator>(monostate, monostate) noexcept { return false; }
    constexpr bool operator<=(monostate, monostate) noexcept { return true; }
    constexpr bool operator>=(monostate, monostate) noexcept { return true; }
#endif
```



### `swap`

``` cpp
template <typename... _Types>
inline enable_if_t<(is_move_constructible_v<_Types> && ...) && (is_swappable_v<_Types> && ...)> swap(
    variant<_Types...> & __lhs, variant<_Types...> & __rhs) noexcept(noexcept(__lhs.swap(__rhs))) {
    __lhs.swap(__rhs);
}

template <typename... _Types>
enable_if_t<!((is_move_constructible_v<_Types> && ...) && (is_swappable_v<_Types> && ...))> swap(
    variant<_Types...> &, variant<_Types...> &) = delete;
```



### `bad_variant_cast`

``` cpp
class bad_variant_access : public exception {
public:
    bad_variant_access() noexcept {}

    const char *what() const noexcept override { return _M_reason; }

private:
    bad_variant_access(const char *__reason) noexcept : _M_reason(__reason) {}

    // Must point to a string with static storage duration:
    const char *_M_reason = "bad variant access";

    friend void __throw_bad_variant_access(const char *__what);
};

// Must only be called with a string literal
inline void __throw_bad_variant_access(const char *__what) { _GLIBCXX_THROW_OR_ABORT(bad_variant_access(__what)); }

inline void __throw_bad_variant_access(bool __valueless) {
    if (__valueless) [[__unlikely__]]
        __throw_bad_variant_access("std::get: variant is valueless");
    else
        __throw_bad_variant_access("std::get: wrong index for variant");
}
```



## 2 variant

### 定义

``` cpp
template <typename... _Types>
class variant
    : private __detail::__variant::_Variant_base<_Types...>,
      private _Enable_default_constructor<__detail::__variant::_Traits<_Types...>::_S_default_ctor,
                                          variant<_Types...>>,
      private _Enable_copy_move<__detail::__variant::_Traits<_Types...>::_S_copy_ctor,
                                __detail::__variant::_Traits<_Types...>::_S_copy_assign,
                                __detail::__variant::_Traits<_Types...>::_S_move_ctor,
                                __detail::__variant::_Traits<_Types...>::_S_move_assign, variant<_Types...>>;
```



### 默认构造、拷贝构造、拷贝赋值、移动构造、移动赋值、析构函数

``` cpp
variant() = default;
variant(const variant &__rhs) = default;
variant(variant &&) = default;
variant &operator=(const variant &) = default;
variant &operator=(variant &&) = default;
~variant() = default;
```



### 带参数构造函数

``` cpp
template <typename _Tp, typename = enable_if_t<sizeof...(_Types) != 0>,
          typename = enable_if_t<__not_in_place_tag<_Tp>>, typename _Tj = __accepted_type<_Tp &&>,
          typename = enable_if_t<__exactly_once<_Tj> && is_constructible_v<_Tj, _Tp>>>
constexpr variant(_Tp &&__t) noexcept(is_nothrow_constructible_v<_Tj, _Tp>)
    : variant(in_place_index<__accepted_index<_Tp>>, std::forward<_Tp>(__t)) {}

template <typename _Tp, typename... _Args,
          typename = enable_if_t<__exactly_once<_Tp> && is_constructible_v<_Tp, _Args...>>>
constexpr explicit variant(in_place_type_t<_Tp>, _Args &&...__args)
    : variant(in_place_index<__index_of<_Tp>>, std::forward<_Args>(__args)...) {}

template <
    typename _Tp, typename _Up, typename... _Args,
    typename = enable_if_t<__exactly_once<_Tp> && is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>>>
constexpr explicit variant(in_place_type_t<_Tp>, initializer_list<_Up> __il, _Args &&...__args)
    : variant(in_place_index<__index_of<_Tp>>, __il, std::forward<_Args>(__args)...) {}

template <size_t _Np, typename... _Args, typename _Tp = __to_type<_Np>,
          typename = enable_if_t<is_constructible_v<_Tp, _Args...>>>
constexpr explicit variant(in_place_index_t<_Np>, _Args &&...__args)
    : _Base(in_place_index<_Np>, std::forward<_Args>(__args)...),
      _Default_ctor_enabler(_Enable_default_constructor_tag{}) {}

template <size_t _Np, typename _Up, typename... _Args, typename _Tp = __to_type<_Np>,
          typename = enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>>>
constexpr explicit variant(in_place_index_t<_Np>, initializer_list<_Up> __il, _Args &&...__args)
    : _Base(in_place_index<_Np>, __il, std::forward<_Args>(__args)...),
      _Default_ctor_enabler(_Enable_default_constructor_tag{}) {}
```



### 移动赋值函数

``` cpp
template <typename _Tp>
enable_if_t<__exactly_once<__accepted_type<_Tp &&>> && is_constructible_v<__accepted_type<_Tp &&>, _Tp> &&
                is_assignable_v<__accepted_type<_Tp &&> &, _Tp>,
            variant &>
operator=(_Tp &&__rhs) noexcept(is_nothrow_assignable_v<__accepted_type<_Tp &&> &, _Tp>
                                    &&is_nothrow_constructible_v<__accepted_type<_Tp &&>, _Tp>) {
    constexpr auto __index = __accepted_index<_Tp>;
    if (index() == __index)
        std::get<__index>(*this) = std::forward<_Tp>(__rhs);
    else {
        using _Tj = __accepted_type<_Tp &&>;
        if constexpr (is_nothrow_constructible_v<_Tj, _Tp> || !is_nothrow_move_constructible_v<_Tj>)
            this->emplace<__index>(std::forward<_Tp>(__rhs));
        else
            operator=(variant(std::forward<_Tp>(__rhs)));
    }
    return *this;
}
```



### `emplace`

``` cpp
template <typename _Tp, typename... _Args>
enable_if_t<is_constructible_v<_Tp, _Args...> && __exactly_once<_Tp>, _Tp &> emplace(_Args &&...__args) {
    constexpr size_t __index = __index_of<_Tp>;
    return this->emplace<__index>(std::forward<_Args>(__args)...);
}

template <typename _Tp, typename _Up, typename... _Args>
enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...> && __exactly_once<_Tp>, _Tp &> emplace(
    initializer_list<_Up> __il, _Args &&...__args) {
    constexpr size_t __index = __index_of<_Tp>;
    return this->emplace<__index>(__il, std::forward<_Args>(__args)...);
}

template <size_t _Np, typename... _Args>
enable_if_t<is_constructible_v<variant_alternative_t<_Np, variant>, _Args...>,
            variant_alternative_t<_Np, variant> &>
emplace(_Args &&...__args) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    using type = variant_alternative_t<_Np, variant>;
    // Provide the strong exception-safety guarantee when possible,
    // to avoid becoming valueless.
    if constexpr (is_nothrow_constructible_v<type, _Args...>) {
        this->_M_reset();
        __variant_construct_by_index<_Np>(*this, std::forward<_Args>(__args)...);
    } else if constexpr (is_scalar_v<type>) {
        // This might invoke a potentially-throwing conversion operator:
        const type __tmp(std::forward<_Args>(__args)...);
        // But these steps won't throw:
        this->_M_reset();
        __variant_construct_by_index<_Np>(*this, __tmp);
    } else if constexpr (__detail::__variant::_Never_valueless_alt<type>() && _Traits::_S_move_assign) {
        // This construction might throw:
        variant __tmp(in_place_index<_Np>, std::forward<_Args>(__args)...);
        // But _Never_valueless_alt<type> means this won't:
        *this = std::move(__tmp);
    } else {
        // This case only provides the basic exception-safety guarantee,
        // i.e. the variant can become valueless.
        this->_M_reset();
        __try {
            __variant_construct_by_index<_Np>(*this, std::forward<_Args>(__args)...);
        }
        __catch(...) {
            using __index_type = decltype(this->_M_index);
            this->_M_index = static_cast<__index_type>(variant_npos);
            __throw_exception_again;
        }
    }
    return std::get<_Np>(*this);
}

template <size_t _Np, typename _Up, typename... _Args>
enable_if_t<is_constructible_v<variant_alternative_t<_Np, variant>, initializer_list<_Up> &, _Args...>,
            variant_alternative_t<_Np, variant> &>
emplace(initializer_list<_Up> __il, _Args &&...__args) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    using type = variant_alternative_t<_Np, variant>;
    // Provide the strong exception-safety guarantee when possible,
    // to avoid becoming valueless.
    if constexpr (is_nothrow_constructible_v<type, initializer_list<_Up> &, _Args...>) {
        this->_M_reset();
        __variant_construct_by_index<_Np>(*this, __il, std::forward<_Args>(__args)...);
    } else if constexpr (__detail::__variant::_Never_valueless_alt<type>() && _Traits::_S_move_assign) {
        // This construction might throw:
        variant __tmp(in_place_index<_Np>, __il, std::forward<_Args>(__args)...);
        // But _Never_valueless_alt<type> means this won't:
        *this = std::move(__tmp);
    } else {
        // This case only provides the basic exception-safety guarantee,
        // i.e. the variant can become valueless.
        this->_M_reset();
        __try {
            __variant_construct_by_index<_Np>(*this, __il, std::forward<_Args>(__args)...);
        }
        __catch(...) {
            using __index_type = decltype(this->_M_index);
            this->_M_index = static_cast<__index_type>(variant_npos);
            __throw_exception_again;
        }
    }
    return std::get<_Np>(*this);
}
```



### `valueless_by_exception`

``` cpp
constexpr bool valueless_by_exception() const noexcept { return !this->_M_valid(); }
```



### `index`

``` cpp
constexpr size_t index() const noexcept {
    using __index_type = typename _Base::__index_type;
    if constexpr (__detail::__variant::__never_valueless<_Types...>())
        return this->_M_index;
    else if constexpr (sizeof...(_Types) <= __index_type(-1) / 2)
        return make_signed_t<__index_type>(this->_M_index);
    else
        return size_t(__index_type(this->_M_index + 1)) - 1;
}
```



### `swap`

``` cpp
void swap(variant &__rhs) noexcept((__is_nothrow_swappable<_Types>::value && ...) &&
                                   is_nothrow_move_constructible_v<variant>) {
    __detail::__variant::__raw_idx_visit(
        [this, &__rhs](auto &&__rhs_mem, auto __rhs_index) mutable {
            if constexpr (__rhs_index != variant_npos) {
                if (this->index() == __rhs_index) {
                    auto &__this_mem = std::get<__rhs_index>(*this);
                    using std::swap;
                    swap(__this_mem, __rhs_mem);
                } else {
                    if (!this->valueless_by_exception()) [[__likely__]] {
                        auto __tmp(std::move(__rhs_mem));
                        __rhs = std::move(*this);
                        this->_M_destructive_move(__rhs_index, std::move(__tmp));
                    } else {
                        this->_M_destructive_move(__rhs_index, std::move(__rhs_mem));
                        __rhs._M_reset();
                    }
                }
            } else {
                if (!this->valueless_by_exception()) [[__likely__]] {
                    __rhs = std::move(*this);
                    this->_M_reset();
                }
            }
        },
        __rhs);
}
```



### operators

``` cpp
#define _VARIANT_RELATION_FUNCTION_TEMPLATE(__OP) \
    template <typename... _Tp>                    \
    friend constexpr bool operator __OP(const variant<_Tp...> &__lhs, const variant<_Tp...> &__rhs);

        _VARIANT_RELATION_FUNCTION_TEMPLATE(<)
        _VARIANT_RELATION_FUNCTION_TEMPLATE(<=)
        _VARIANT_RELATION_FUNCTION_TEMPLATE(==)
        _VARIANT_RELATION_FUNCTION_TEMPLATE(!=)
        _VARIANT_RELATION_FUNCTION_TEMPLATE(>=)
        _VARIANT_RELATION_FUNCTION_TEMPLATE(>)

#undef _VARIANT_RELATION_FUNCTION_TEMPLATE
```



### `get`

``` cpp
template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> &get(variant<_Types...> & __v) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    if (__v.index() != _Np) __throw_bad_variant_access(__v.valueless_by_exception());
    return __detail::__variant::__get<_Np>(__v);
}

template <size_t _Np, typename... _Types>
constexpr variant_alternative_t<_Np, variant<_Types...>> &&get(variant<_Types...> && __v) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    if (__v.index() != _Np) __throw_bad_variant_access(__v.valueless_by_exception());
    return __detail::__variant::__get<_Np>(std::move(__v));
}

template <size_t _Np, typename... _Types>
constexpr const variant_alternative_t<_Np, variant<_Types...>> &get(const variant<_Types...> &__v) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    if (__v.index() != _Np) __throw_bad_variant_access(__v.valueless_by_exception());
    return __detail::__variant::__get<_Np>(__v);
}

template <size_t _Np, typename... _Types>
constexpr const variant_alternative_t<_Np, variant<_Types...>> &&get(const variant<_Types...> &&__v) {
    static_assert(_Np < sizeof...(_Types), "The index must be in [0, number of alternatives)");
    if (__v.index() != _Np) __throw_bad_variant_access(__v.valueless_by_exception());
    return __detail::__variant::__get<_Np>(std::move(__v));
}
```



### visit

``` cpp
template <typename _Result_type, typename _Visitor, typename... _Variants>
constexpr decltype(auto) __do_visit(_Visitor && __visitor, _Variants && ...__variants) {
    constexpr auto &__vtable =
        __detail::__variant::__gen_vtable<_Result_type, _Visitor &&, _Variants &&...>::_S_vtable;

    auto __func_ptr = __vtable._M_access(__variants.index()...);
    return (*__func_ptr)(std::forward<_Visitor>(__visitor), std::forward<_Variants>(__variants)...);
}

template <typename _Tp, typename... _Types>
constexpr inline bool __same_types = (is_same_v<_Tp, _Types> && ...);

template <size_t _Idx, typename _Visitor, typename _Variant>
decltype(auto) __check_visitor_result(_Visitor && __vis, _Variant && __variant) {
    return std::__invoke(std::forward<_Visitor>(__vis), std::get<_Idx>(std::forward<_Variant>(__variant)));
}

template <typename _Visitor, typename _Variant, size_t... _Idxs>
constexpr bool __check_visitor_results(std::index_sequence<_Idxs...>) {
    return __same_types<decltype(
        __check_visitor_result<_Idxs>(std::declval<_Visitor>(), std::declval<_Variant>()))...>;
}

template <typename _Visitor, typename... _Variants>
constexpr decltype(auto) visit(_Visitor && __visitor, _Variants && ...__variants) {
    if ((__variants.valueless_by_exception() || ...))
        __throw_bad_variant_access("std::visit: variant is valueless");

    using _Result_type = std::invoke_result_t<_Visitor, decltype(std::get<0>(std::declval<_Variants>()))...>;

    using _Tag = __detail::__variant::__deduce_visit_result<_Result_type>;

    if constexpr (sizeof...(_Variants) == 1) {
        constexpr bool __visit_rettypes_match = __check_visitor_results<_Visitor, _Variants...>(
            std::make_index_sequence<std::variant_size<remove_reference_t<_Variants>...>::value>());
        if constexpr (!__visit_rettypes_match) {
            static_assert(__visit_rettypes_match,
                          "std::visit requires the visitor to have the same "
                          "return type for all alternatives of a variant");
            return;
        } else
            return std::__do_visit<_Tag>(std::forward<_Visitor>(__visitor), std::forward<_Variants>(__variants)...);
    } else
        return std::__do_visit<_Tag>(std::forward<_Visitor>(__visitor), std::forward<_Variants>(__variants)...);
}
```



### hash

``` cpp
template <bool, typename... _Types>
struct __variant_hash_call_base_impl {
    size_t operator()(const variant<_Types...> &__t) const
        noexcept((is_nothrow_invocable_v<hash<decay_t<_Types>>, _Types> && ...)) {
        size_t __ret;
        __detail::__variant::__raw_visit(
            [&__t, &__ret](auto &&__t_mem) mutable {
                using _Type = __remove_cvref_t<decltype(__t_mem)>;
                if constexpr (!is_same_v<_Type, __detail::__variant::__variant_cookie>)
                    __ret = std::hash<size_t>{}(__t.index()) + std::hash<_Type>{}(__t_mem);
                else
                    __ret = std::hash<size_t>{}(__t.index());
            },
            __t);
        return __ret;
    }
};

template <typename... _Types>
struct __variant_hash_call_base_impl<false, _Types...> {};

template <typename... _Types>
using __variant_hash_call_base =
    __variant_hash_call_base_impl<(__poison_hash<remove_const_t<_Types>>::__enable_hash_call && ...), _Types...>;

template <typename... _Types>
struct hash<variant<_Types...>>
    : private __detail::__variant::_Variant_hash_base<variant<_Types...>, std::index_sequence_for<_Types...>>,
      public __variant_hash_call_base<_Types...> {
    using result_type [[__deprecated__]] = size_t;
    using argument_type [[__deprecated__]] = variant<_Types...>;
};

template <>
struct hash<monostate> {
    using result_type [[__deprecated__]] = size_t;
    using argument_type [[__deprecated__]] = monostate;

    size_t operator()(const monostate &) const noexcept {
        constexpr size_t __magic_monostate_hash = -7777;
        return __magic_monostate_hash;
    }
};

template <typename... _Types>
struct __is_fast_hash<hash<variant<_Types...>>> : bool_constant<(__is_fast_hash<_Types>::value && ...)> {};
```



# 问题

- 什么是SMF？
- `in_place_t`等相关的用法？
- `aligned_storage`的使用。



