# tuple

## 1主要代码

### 1.1 `__empty_not_final`

``` cpp
template <typename _Tp>
struct __is_empty_non_tuple : is_empty<_Tp> {};

// Using EBO for elements that are tuples causes ambiguous base errors.
template <typename _El0, typename... _El>
struct __is_empty_non_tuple<tuple<_El0, _El...>> : false_type {};

// Use the Empty Base-class Optimization for empty, non-final types.
template <typename _Tp>
using __empty_not_final = typename conditional<__is_final(_Tp), false_type, __is_empty_non_tuple<_Tp>>::type;
```

### 1.2 `_Head_base`

``` cpp
template <size_t _Idx, typename _Head, bool = __empty_not_final<_Head>::value>
struct _Head_base;

template <size_t _Idx, typename _Head>
struct _Head_base<_Idx, _Head, true> : public _Head {
    constexpr _Head_base() : _Head() {}

    constexpr _Head_base(const _Head &__h) : _Head(__h) {}

    constexpr _Head_base(const _Head_base &) = default;
    constexpr _Head_base(_Head_base &&) = default;

    template <typename _UHead>
    constexpr _Head_base(_UHead &&__h) : _Head(std::forward<_UHead>(__h)) {}

    // 省去allocator相关的内容

    static constexpr _Head &_M_head(_Head_base &__b) noexcept { return __b; }

    static constexpr const _Head &_M_head(const _Head_base &__b) noexcept { return __b; }
};

template <size_t _Idx, typename _Head>
struct _Head_base<_Idx, _Head, false> {
    constexpr _Head_base() : _M_head_impl() {}

    constexpr _Head_base(const _Head &__h) : _M_head_impl(__h) {}

    constexpr _Head_base(const _Head_base &) = default;
    constexpr _Head_base(_Head_base &&) = default;

    template <typename _UHead>
    constexpr _Head_base(_UHead &&__h) : _M_head_impl(std::forward<_UHead>(__h)) {}

	// 省去了一些allocator相关的内容
    
    static constexpr _Head &_M_head(_Head_base &__b) noexcept { return __b._M_head_impl; }

    static constexpr const _Head &_M_head(const _Head_base &__b) noexcept { return __b._M_head_impl; }

    _Head _M_head_impl;
};
```

这个类是为下面的impl服务的，保存的是`tuple`中模板参数类型的第一个类型数据。对于`__empty_not_final`为`false`的场景，模板参数`_Idx`为impl的当前索引，`_Head`为当前索引对应的类型，成员变量`_M_head_impl`为它的实际**值**，即`tuple`中对应索引的类型的实际值。注意到其中的`_M_head`函数，返回的就是这个值。

### 1.3 impl

#### impl0

``` cpp
/**
 * Contains the actual implementation of the @c tuple template, stored
 * as a recursive inheritance hierarchy from the first element (most
 * derived class) to the last (least derived class). The @c Idx
 * parameter gives the 0-based index of the element stored at this
 * point in the hierarchy; we use it to implement a constant-time
 * get() operation.
 */
template <size_t _Idx, typename... _Elements>
struct _Tuple_impl;
```

`_Tuple_impl`包含了对`tuple`模板的真正实现，用的是递归继承的方式。`_Idx`是元素的索引。结合下面的叙述，举个例子讲述一下这里面提到的“递归继承”。

``` cpp
// 如果我们定义了这么一个tuple
tuple<int, float, string>;

// 那么，在tuple类里面，会有这样的对impl的一个调用。
_Tuple_impl<0, int, float, string>; // 这里调用的是impl1

// 递归地，有：
_Tuple_impl<1, float, string>; // 仍然调用impl1
_Tuple_impl<2, string>;        // 这里是impl2，即这个递归的基本项（basic case）
```

#### impl1

``` cpp
/**
 * Recursive tuple implementation. Here we store the @c Head element
 * and derive from a @c Tuple_impl containing the remaining elements
 * (which contains the @c Tail).
 */
template <size_t _Idx, typename _Head, typename... _Tail>
struct _Tuple_impl<_Idx, _Head, _Tail...> : public _Tuple_impl<_Idx + 1, _Tail...>,
                                            private _Head_base<_Idx, _Head> {
    template <size_t, typename...>
    friend struct _Tuple_impl;

    typedef _Tuple_impl<_Idx + 1, _Tail...> _Inherited;
    typedef _Head_base<_Idx, _Head> _Base;

    static constexpr _Head &_M_head(_Tuple_impl &__t) noexcept { return _Base::_M_head(__t); }

    static constexpr const _Head &_M_head(const _Tuple_impl &__t) noexcept { return _Base::_M_head(__t); }

    static constexpr _Inherited &_M_tail(_Tuple_impl &__t) noexcept { return __t; }

    static constexpr const _Inherited &_M_tail(const _Tuple_impl &__t) noexcept { return __t; }

    constexpr _Tuple_impl() : _Inherited(), _Base() {}

    explicit constexpr _Tuple_impl(const _Head &__head, const _Tail &...__tail)
        : _Inherited(__tail...), _Base(__head) {}

    template <typename _UHead, typename... _UTail, typename = __enable_if_t<sizeof...(_Tail) == sizeof...(_UTail)>>
    explicit constexpr _Tuple_impl(_UHead &&__head, _UTail &&...__tail)
        : _Inherited(std::forward<_UTail>(__tail)...), _Base(std::forward<_UHead>(__head)) {}

    constexpr _Tuple_impl(const _Tuple_impl &) = default;

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 2729. Missing SFINAE on std::pair::operator=
    _Tuple_impl &operator=(const _Tuple_impl &) = delete;

    constexpr _Tuple_impl(_Tuple_impl &&__in) noexcept(
        __and_<is_nothrow_move_constructible<_Head>, is_nothrow_move_constructible<_Inherited>>::value)
        : _Inherited(std::move(_M_tail(__in))), _Base(std::forward<_Head>(_M_head(__in))) {}

    template <typename... _UElements>
    constexpr _Tuple_impl(const _Tuple_impl<_Idx, _UElements...> &__in)
        : _Inherited(_Tuple_impl<_Idx, _UElements...>::_M_tail(__in)),
          _Base(_Tuple_impl<_Idx, _UElements...>::_M_head(__in)) {}

    template <typename _UHead, typename... _UTails>
    constexpr _Tuple_impl(_Tuple_impl<_Idx, _UHead, _UTails...> &&__in)
        : _Inherited(std::move(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in))),
          _Base(std::forward<_UHead>(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in))) {}

    // 省去一些allocator相关的东西，区别主要在内存分配上，整体逻辑相似。
                                                
    template <typename... _UElements>
    _GLIBCXX20_CONSTEXPR void _M_assign(const _Tuple_impl<_Idx, _UElements...> &__in) {
        _M_head(*this) = _Tuple_impl<_Idx, _UElements...>::_M_head(__in);
        _M_tail(*this)._M_assign(_Tuple_impl<_Idx, _UElements...>::_M_tail(__in));
    }

    template <typename _UHead, typename... _UTails>
    _GLIBCXX20_CONSTEXPR void _M_assign(_Tuple_impl<_Idx, _UHead, _UTails...> &&__in) {
        _M_head(*this) = std::forward<_UHead>(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_head(__in));
        _M_tail(*this)._M_assign(std::move(_Tuple_impl<_Idx, _UHead, _UTails...>::_M_tail(__in)));
    }

protected:
    _GLIBCXX20_CONSTEXPR
    void _M_swap(_Tuple_impl &__in) {
        using std::swap;
        swap(_M_head(*this), _M_head(__in));
        _Inherited::_M_swap(_M_tail(__in));
    }
};
```

- 它定义了两个基类`_Inherited`、`_Base`，分别对应与它的两个基类。
- `_M_head`函数，调用`_Base::_M_head`函数，返回的是第一个索引存储的数据。`_M_tail`函数，相当于类型转换，将一个类型为`<_Idx, _Head, _Tail...>`的变量转换为`<_Idx+1, _Tail...>`类型。
- 对于传入`_Tuple_impl`类型变量的（拷贝、移动）构造函数来说，结合上面两个函数，区分head和tail来构造。
- `_M_assign`函数同样调用`_M_head`和`_M_tail`函数来进行赋值。

- `_M_swap`函数先调用`std::swap`来交换head，然后调用`_Inherited`的`_M_swap`函数来交换tail。*备注：这里的`_Inherited::_M_swap();`指的是在子类中调用父类的同名函数的形式，而不是因为`_Inherited`中的`swap`函数是static的。*

#### impl2

``` cpp
// Basis case of inheritance recursion.
template <size_t _Idx, typename _Head>
struct _Tuple_impl<_Idx, _Head> : private _Head_base<_Idx, _Head> {
    template <size_t, typename...>
    friend struct _Tuple_impl;

    typedef _Head_base<_Idx, _Head> _Base;

    static constexpr _Head &_M_head(_Tuple_impl &__t) noexcept { return _Base::_M_head(__t); }

    static constexpr const _Head &_M_head(const _Tuple_impl &__t) noexcept { return _Base::_M_head(__t); }

    constexpr _Tuple_impl() : _Base() {}

    explicit constexpr _Tuple_impl(const _Head &__head) : _Base(__head) {}

    template <typename _UHead>
    explicit constexpr _Tuple_impl(_UHead &&__head) : _Base(std::forward<_UHead>(__head)) {}

    constexpr _Tuple_impl(const _Tuple_impl &) = default;

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 2729. Missing SFINAE on std::pair::operator=
    _Tuple_impl &operator=(const _Tuple_impl &) = delete;

    constexpr _Tuple_impl(_Tuple_impl &&__in) noexcept(is_nothrow_move_constructible<_Head>::value)
        : _Base(std::forward<_Head>(_M_head(__in))) {}

    template <typename _UHead>
    constexpr _Tuple_impl(const _Tuple_impl<_Idx, _UHead> &__in)
        : _Base(_Tuple_impl<_Idx, _UHead>::_M_head(__in)) {}

    template <typename _UHead>
    constexpr _Tuple_impl(_Tuple_impl<_Idx, _UHead> &&__in)
        : _Base(std::forward<_UHead>(_Tuple_impl<_Idx, _UHead>::_M_head(__in))) {}

    template <typename _UHead>
    _GLIBCXX20_CONSTEXPR void _M_assign(const _Tuple_impl<_Idx, _UHead> &__in) {
        _M_head(*this) = _Tuple_impl<_Idx, _UHead>::_M_head(__in);
    }

    template <typename _UHead>
    _GLIBCXX20_CONSTEXPR void _M_assign(_Tuple_impl<_Idx, _UHead> &&__in) {
        _M_head(*this) = std::forward<_UHead>(_Tuple_impl<_Idx, _UHead>::_M_head(__in));
    }

protected:
    _GLIBCXX20_CONSTEXPR
    void _M_swap(_Tuple_impl &__in) {
        using std::swap;
        swap(_M_head(*this), _M_head(__in));
    }
};
```

正如前面所说，impl2是一个recursion的basic case，处理的是递归最后一层，只剩下最后一个元素的情况。和impl1更简单些，它的基类只有`_Base`，而没有`_Inherited`。这里不再赘述。

### 1.4 tuple

#### tuple0

``` cpp
/// Primary class template, tuple
template <typename... _Elements>
class tuple : public _Tuple_impl<0, _Elements...> {
    typedef _Tuple_impl<0, _Elements...> _Inherited;

public:
    template <typename _Dummy = void, _ImplicitDefaultCtor<is_void<_Dummy>::value> = true>
    constexpr tuple() noexcept(__and_<is_nothrow_default_constructible<_Elements>...>::value) : _Inherited() {}

    template <bool _NotEmpty = (sizeof...(_Elements) >= 1), _ImplicitCtor<_NotEmpty, const _Elements &...> = true>
    constexpr tuple(const _Elements &...__elements) noexcept(__nothrow_constructible<const _Elements &...>())
        : _Inherited(__elements...) {}

    constexpr tuple(const tuple &) = default;

    constexpr tuple(tuple &&) = default;

    template <typename... _UElements,
              bool _Valid = (sizeof...(_Elements) == sizeof...(_UElements)) &&
                            !__use_other_ctor<const tuple<_UElements...> &>(),
              _ImplicitCtor<_Valid, const _UElements &...> = true>
    constexpr tuple(const tuple<_UElements...> &__in) noexcept(__nothrow_constructible<const _UElements &...>())
        : _Inherited(static_cast<const _Tuple_impl<0, _UElements...> &>(__in)) {}

    // tuple assignment

    _GLIBCXX20_CONSTEXPR
    tuple &operator=(
        typename conditional<__assignable<const _Elements &...>(), const tuple &, const __nonesuch &>::type
            __in) noexcept(__nothrow_assignable<const _Elements &...>()) {
        this->_M_assign(__in);
        return *this;
    }

    // tuple swap
    _GLIBCXX20_CONSTEXPR
    void swap(tuple &__in) noexcept(__and_<__is_nothrow_swappable<_Elements>...>::value) {
        _Inherited::_M_swap(__in);
    }
};
```

- 构造函数、赋值函数、swap函数等，主要还是调用的`_Tuple_impl`

#### tuple1

``` cpp
// Explicit specialization, zero-element tuple.
template <>
class tuple<> {
public:
    void swap(tuple &) noexcept { /* no-op */
    }
    // We need the default since we're going to define no-op
    // allocator constructors.
    tuple() = default;
};
```

专门为无元素的tuple定制的tuple1，不继承自任何其他类，构造函数、swap函数等都是空的，即没有任何操作（no-op）。

#### tuple2

``` cpp
/// Partial specialization, 2-element tuple.
/// Includes construction and assignment from a pair.
template <typename _T1, typename _T2>
class tuple<_T1, _T2> : public _Tuple_impl<0, _T1, _T2> {
    typedef _Tuple_impl<0, _T1, _T2> _Inherited;

public:
    template <bool _Dummy = true, _ImplicitDefaultCtor<_Dummy, _T1, _T2> = true>
    constexpr tuple() noexcept(__nothrow_default_constructible()) : _Inherited() {}

    template <bool _Dummy = true, _ImplicitCtor<_Dummy, const _T1 &, const _T2 &> = true>
    constexpr tuple(const _T1 &__a1, const _T2 &__a2) noexcept(__nothrow_constructible<const _T1 &, const _T2 &>())
        : _Inherited(__a1, __a2) {}

    template <typename _U1, typename _U2, _ImplicitCtor<!__is_alloc_arg<_U1>(), _U1, _U2> = true>
    constexpr tuple(_U1 &&__a1, _U2 &&__a2) noexcept(__nothrow_constructible<_U1, _U2>())
        : _Inherited(std::forward<_U1>(__a1), std::forward<_U2>(__a2)) {}

    constexpr tuple(const tuple &) = default;

    constexpr tuple(tuple &&) = default;

    template <typename _U1, typename _U2, _ImplicitCtor<true, const _U1 &, const _U2 &> = true>
    constexpr tuple(const tuple<_U1, _U2> &__in) noexcept(__nothrow_constructible<const _U1 &, const _U2 &>())
        : _Inherited(static_cast<const _Tuple_impl<0, _U1, _U2> &>(__in)) {}

    template <typename _U1, typename _U2, _ImplicitCtor<true, _U1, _U2> = true>
    constexpr tuple(pair<_U1, _U2> &&__in) noexcept(__nothrow_constructible<_U1, _U2>())
        : _Inherited(std::forward<_U1>(__in.first), std::forward<_U2>(__in.second)) {}

    // Tuple assignment.

    _GLIBCXX20_CONSTEXPR
    tuple &operator=(
        typename conditional<__assignable<const _T1 &, const _T2 &>(), const tuple &, const __nonesuch &>::type
            __in) noexcept(__nothrow_assignable<const _T1 &, const _T2 &>()) {
        this->_M_assign(__in);
        return *this;
    }

    _GLIBCXX20_CONSTEXPR
    tuple &operator=(typename conditional<__assignable<_T1, _T2>(), tuple &&, __nonesuch &&>::type __in) noexcept(
        __nothrow_assignable<_T1, _T2>()) {
        this->_M_assign(std::move(__in));
        return *this;
    }

    template <typename _U1, typename _U2>
    _GLIBCXX20_CONSTEXPR __enable_if_t<__assignable<const _U1 &, const _U2 &>(), tuple &> operator=(
        const pair<_U1, _U2> &__in) noexcept(__nothrow_assignable<const _U1 &, const _U2 &>()) {
        this->_M_head(*this) = __in.first;
        this->_M_tail(*this)._M_head(*this) = __in.second;
        return *this;
    }

    template <typename _U1, typename _U2>
    _GLIBCXX20_CONSTEXPR __enable_if_t<__assignable<_U1, _U2>(), tuple &> operator=(pair<_U1, _U2> &&__in) noexcept(
        __nothrow_assignable<_U1, _U2>()) {
        this->_M_head(*this) = std::forward<_U1>(__in.first);
        this->_M_tail(*this)._M_head(*this) = std::forward<_U2>(__in.second);
        return *this;
    }

    _GLIBCXX20_CONSTEXPR
    void swap(tuple &__in) noexcept(__and_<__is_nothrow_swappable<_T1>, __is_nothrow_swappable<_T2>>::value) {
        _Inherited::_M_swap(__in);
    }
};
```

tuple2是专门为只有两个元素的tuple定制的，大多数的操作和tuple0没有区别。它继承自`_Tuple_impl<0, _T1, _T2>`，主要的区别是加了参数为`std::pair`的`operator=`函数，第一个元素调用`_M_head`函数构造`_Base`，第二个元素调用`_M_tail`函数构造`_Inherited`。

### 1.5 相关函数

#### `tuple_element` & `tuple_size`

``` cpp
/**
 * Recursive case for tuple_element: strip off the first element in
 * the tuple and retrieve the (i-1)th element of the remaining tuple.
 */
template <size_t __i, typename _Head, typename... _Tail>
struct tuple_element<__i, tuple<_Head, _Tail...>> : tuple_element<__i - 1, tuple<_Tail...>> {};

/**
 * Basis case for tuple_element: The first element is the one we're seeking.
 */
template <typename _Head, typename... _Tail>
struct tuple_element<0, tuple<_Head, _Tail...>> {
    typedef _Head type;
};

/**
 * Error case for tuple_element: invalid index.
 */
template <size_t __i>
struct tuple_element<__i, tuple<>> {
    static_assert(__i < tuple_size<tuple<>>::value, "tuple index is in range");
};

/// class tuple_size
template <typename... _Elements>
struct tuple_size<tuple<_Elements...>> : public integral_constant<size_t, sizeof...(_Elements)> {};

#if __cplusplus > 201402L
template <typename _Tp>
inline constexpr size_t tuple_size_v = tuple_size<_Tp>::value;
#endif
```

另外，在`<utility>`中定义了：

``` cpp
template<size_t __i, typename _Tp>
using __tuple_element_t = typename tuple_element<__i, _Tp>::type;
```

- `tuple_element`的定义仍然是递归的模板继承，通过这种方式，来确定第n个模板参数的类型。

- `tuple_size`实际上用`integral_constant`包装了`sizeof...(_Elements)`，即tuple的参数的个数，亦即tuple的size。

#### `get`

``` cpp
template <size_t __i, typename _Head, typename... _Tail>
constexpr _Head &__get_helper(_Tuple_impl<__i, _Head, _Tail...> & __t) noexcept {
    return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
}

/// Return a reference to the ith element of a tuple.
template <size_t __i, typename... _Elements>
constexpr __tuple_element_t<__i, tuple<_Elements...>> &get(tuple<_Elements...> & __t) noexcept {
    return std::__get_helper<__i>(__t);
}

/// Return an rvalue reference to the ith element of a tuple rvalue.
template <size_t __i, typename... _Elements>
constexpr __tuple_element_t<__i, tuple<_Elements...>> &&get(tuple<_Elements...> && __t) noexcept {
    typedef __tuple_element_t<__i, tuple<_Elements...>> __element_type;
    return std::forward<__element_type &&>(std::get<__i>(__t));
}
```

`get`函数的逻辑，主要还是调用`_Tuple_impl::_M_head`函数，返回tuple中特定索引`__i`的值。

#### 比较函数

``` cpp
// This class performs the comparison operations on tuples
template <typename _Tp, typename _Up, size_t __i, size_t __size>
struct __tuple_compare {
    static constexpr bool __eq(const _Tp &__t, const _Up &__u) {
        return bool(std::get<__i>(__t) == std::get<__i>(__u)) &&
               __tuple_compare<_Tp, _Up, __i + 1, __size>::__eq(__t, __u);
    }

    static constexpr bool __less(const _Tp &__t, const _Up &__u) {
        return bool(std::get<__i>(__t) < std::get<__i>(__u)) ||
               (!bool(std::get<__i>(__u) < std::get<__i>(__t)) &&
                __tuple_compare<_Tp, _Up, __i + 1, __size>::__less(__t, __u));
    }
};

template <typename _Tp, typename _Up, size_t __size>
struct __tuple_compare<_Tp, _Up, __size, __size> {
    static constexpr bool __eq(const _Tp &, const _Up &) { return true; }

    static constexpr bool __less(const _Tp &, const _Up &) { return false; }
};

template <typename... _TElements, typename... _UElements>
constexpr bool operator==(const tuple<_TElements...> &__t, const tuple<_UElements...> &__u) {
    static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                  "tuple objects can only be compared if they have equal sizes.");
    using __compare = __tuple_compare<tuple<_TElements...>, tuple<_UElements...>, 0, sizeof...(_TElements)>;
    return __compare::__eq(__t, __u);
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator<(const tuple<_TElements...> &__t, const tuple<_UElements...> &__u) {
    static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                  "tuple objects can only be compared if they have equal sizes.");
    using __compare = __tuple_compare<tuple<_TElements...>, tuple<_UElements...>, 0, sizeof...(_TElements)>;
    return __compare::__less(__t, __u);
}
```

- 对于第一个compare结构体的**等于**逻辑，先比较第`__i`个元素是否相等，如果相等的话再比较`__i+1`个元素是否相等，递归下去。它的**小于**逻辑，先比较第`__i`个元素是否小于，如果不小于的话，再比较`__i+1`个元素是否小于，递归下去。
- 对于第二个compare结构体的等于逻辑直接返回true，小于逻辑直接返回false。这就是上面这个递归函数的最终逻辑，对于等于逻辑，如果一直相等，当`__i+1==__size`的时候，意味着最终是相等的；对于小于逻辑，如果一直不小于，当`__i+1==__size`的时候，意味着最终是不小于的（小于函数返回false）。
- 另外，注意到，只能相同模板个数大小的tuple才能比较。

#### `make_tuple` & `forward_as_tuple` & `tie`

``` cpp
// NB: DR 705.
template <typename... _Elements>
constexpr tuple<typename __decay_and_strip<_Elements>::__type...> make_tuple(_Elements && ...__args) {
    typedef tuple<typename __decay_and_strip<_Elements>::__type...> __result_type;
    return __result_type(std::forward<_Elements>(__args)...);
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2275. Why is forward_as_tuple not constexpr?
/// std::forward_as_tuple
template <typename... _Elements>
constexpr tuple<_Elements &&...> forward_as_tuple(_Elements && ...__args) noexcept {
    return tuple<_Elements &&...>(std::forward<_Elements>(__args)...);
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2301. Why is tie not constexpr?
/// tie
template <typename... _Elements>
constexpr tuple<_Elements &...> tie(_Elements & ...__args) noexcept {
    return tuple<_Elements &...>(__args...);
}
```

这几个函数的特点，就是将传入的参数构造一个`tuple`。

#### `tuple_cat`

``` cpp
// Performs the actual concatenation by step-wise expanding tuple-like
// objects into the elements,  which are finally forwarded into the
// result tuple.
template <typename _Ret, typename _Indices, typename... _Tpls>
struct __tuple_concater;

template <typename _Ret, size_t... _Is, typename _Tp, typename... _Tpls>
struct __tuple_concater<_Ret, _Index_tuple<_Is...>, _Tp, _Tpls...> {
    template <typename... _Us>
    static constexpr _Ret _S_do(_Tp &&__tp, _Tpls &&...__tps, _Us &&...__us) {
        typedef typename __make_1st_indices<_Tpls...>::__type __idx;
        typedef __tuple_concater<_Ret, __idx, _Tpls...> __next;
        return __next::_S_do(std::forward<_Tpls>(__tps)..., std::forward<_Us>(__us)...,
                             std::get<_Is>(std::forward<_Tp>(__tp))...);
    }
};

template <typename _Ret>
struct __tuple_concater<_Ret, _Index_tuple<>> {
    template <typename... _Us>
    static constexpr _Ret _S_do(_Us &&...__us) {
        return _Ret(std::forward<_Us>(__us)...);
    }
};

/// tuple_cat
template <typename... _Tpls, typename = typename enable_if<__and_<__is_tuple_like<_Tpls>...>::value>::type>
constexpr auto tuple_cat(_Tpls && ...__tpls)->typename __tuple_cat_result<_Tpls...>::__type {
    typedef typename __tuple_cat_result<_Tpls...>::__type __ret;
    typedef typename __make_1st_indices<_Tpls...>::__type __idx;
    typedef __tuple_concater<__ret, __idx, _Tpls...> __concater;
    return __concater::_S_do(std::forward<_Tpls>(__tpls)...);
}
```

#### `pair`

``` cpp
// See stl_pair.h...
/** "piecewise construction" using a tuple of arguments for each member.
 *
 * @param __first Arguments for the first member of the pair.
 * @param __second Arguments for the second member of the pair.
 *
 * The elements of each tuple will be used as the constructor arguments
 * for the data members of the pair.
 */
template <class _T1, class _T2>
template <typename... _Args1, typename... _Args2>
_GLIBCXX20_CONSTEXPR inline pair<_T1, _T2>::pair(piecewise_construct_t, tuple<_Args1...> __first,
                                                 tuple<_Args2...> __second)
    : pair(__first, __second, typename _Build_index_tuple<sizeof...(_Args1)>::__type(),
           typename _Build_index_tuple<sizeof...(_Args2)>::__type()) {}

template <class _T1, class _T2>
template <typename... _Args1, size_t... _Indexes1, typename... _Args2, size_t... _Indexes2>
_GLIBCXX20_CONSTEXPR inline pair<_T1, _T2>::pair(tuple<_Args1...> & __tuple1, tuple<_Args2...> & __tuple2,
                                                 _Index_tuple<_Indexes1...>, _Index_tuple<_Indexes2...>)
    : first(std::forward<_Args1>(std::get<_Indexes1>(__tuple1))...),
      second(std::forward<_Args2>(std::get<_Indexes2>(__tuple2))...) {}
```

`pair`函数的逻辑，将tuple`__tuple1`的`Indexes1`上的元素作为这个`pair`的first数据，将tuple`__tuple2`的`Indexes2`上的元素作为这个`pair`的second数据。

#### `apply` & `make_from_tuple`

``` cpp
#define __cpp_lib_apply 201603

template <typename _Fn, typename _Tuple, size_t... _Idx>
constexpr decltype(auto) __apply_impl(_Fn && __f, _Tuple && __t, index_sequence<_Idx...>) {
    return std::__invoke(std::forward<_Fn>(__f), std::get<_Idx>(std::forward<_Tuple>(__t))...);
}

template <typename _Fn, typename _Tuple>
constexpr decltype(auto) apply(_Fn && __f,
                               _Tuple && __t) noexcept(__unpack_std_tuple<is_nothrow_invocable, _Fn, _Tuple>) {
    using _Indices = make_index_sequence<tuple_size_v<remove_reference_t<_Tuple>>>;
    return std::__apply_impl(std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
}

#define __cpp_lib_make_from_tuple 201606

template <typename _Tp, typename _Tuple, size_t... _Idx>
constexpr _Tp __make_from_tuple_impl(_Tuple && __t, index_sequence<_Idx...>) {
    return _Tp(std::get<_Idx>(std::forward<_Tuple>(__t))...);
}

template <typename _Tp, typename _Tuple>
constexpr _Tp make_from_tuple(_Tuple && __t) noexcept(__unpack_std_tuple<is_nothrow_constructible, _Tp, _Tuple>) {
    return __make_from_tuple_impl<_Tp>(std::forward<_Tuple>(__t),
                                       make_index_sequence<tuple_size_v<remove_reference_t<_Tuple>>>{});
}
```

- `apply`函数，将函数`__f`应用于tuple`__t`上。即将tuple的每一个参数作为函数的入参。
- `make_from_tuple`函数，类似于`apply`函数，以输入的tuple`__t`作为类型`_Tp`的构造函数的参数，返回`_Tp`的实例。

## 2 备注

### 2.1 `is_empty`

``` cpp
template <class T> is_empty;
```

以下摘自[网页](https://en.cppreference.com/w/cpp/types/is_empty)：

> If `T` is an empty type (that is, a non-union class type with no non-static data members other than bit-fields of size 0, no virtual functions, no virtual base classes, and no non-empty base classes), provides the member constant `value` equal to true. For any other type, `value` is false.

简单翻译如下：

> 如果`T`是一个空类型（除了size为0的数据类型没有非静态数据成员的非union类型，没有虚函数，没有虚基类，并且没有非空基类），那么`is_empty`类的常量`value`为true；否则，为false。

### 2.2 `no_unique_address`

以下摘自[网页](https://en.cppreference.com/w/cpp/language/attributes/no_unique_address)：

> Allows this data member to be overlapped with other non-static data members or base class subobjects of its class.
>
> Makes this member subobject potentially-overlapping, i.e., allows this member to be overlapped with other non-static data members or base class subobjects of its class. This means that if the member has an empty class type (e.g. stateless allocator), the compiler may optimise it to occupy no space, just like if it were an empty base. If the member is not empty, any tail padding in it may be also reused to store other data members.

简单翻译如下：

> 允许特定的数据成员可以和其他非静态数据成员或者基类子对象重叠。
>
> 使这个子对象有可能重叠的，即允许这个成员可以和其他非静态成员变量或基类子对象重叠。这意味着如果成员有一个空的类型（如无状态分配器），编译器可以优化它到不占用任何空间，就像它是一个空基类。如果不是空的，任何尾部的padding可能会重用来存储其他数据成员。

### 2.3 EBO：Empty Base-class Optimisation

以下摘自[网页](https://en.cppreference.com/w/cpp/language/ebo)：

> Allows the size of an empty base subobject to be zero.
>
> The size of any object or member subobject is required to be at least 1 even if the type is an empty class type(that is, a class or struct that has no non-static data members), (unless with `[[no_unique_address]]`, see below) (since C++20) in order to be able to guarantee that the addresses of distinct objects of the same type are always distinct.
>
> However, base class subobjects are not so constrained, and can be completely optimized out from the object layout.

简单翻译如下：

> EBO是一种允许空基类子对象的大小可以为0的技术。
>
> 一般地，任何一个对象或成员子对象必须至少是1，就算是空类型（即一个class或struct没有非静态成员变量）的时候。（除非有`[[no_unique_address]]`的标识）。这是为了能保证相同类型的不同对象的地址是不同的。
>
> 但是空基类子对象是不受到这个影响的。

参看一个例子（来源同上）：

``` cpp
struct Base {}; // empty class
 
struct Derived1 : Base {
    int i;
};
 
int main()
{
    // the size of any object of empty class type is at least 1
    static_assert(sizeof(Base) >= 1);
 
    // empty base optimization applies
    static_assert(sizeof(Derived1) == sizeof(int));
}
```

以下内容来源同上：

> Empty base optimization is prohibited if one of the empty base classes is also the type or the base of the type of the first non-static data member, since the two base subobjects of the same type are required to have different addresses within the object representation of the most derived type.

简单翻译如下：

> 当一个空基类是第一个非静态数据成员的类型或者类型的基类型时，EBO不能适用。因为两个同样类型的基类子对象需要有不同的地址。

参见例子（来源同上）：

``` cpp
struct Base {}; // empty class
 
struct Derived1 : Base {
    int i;
};
 
struct Derived2 : Base {
    Base c; // Base, occupies 1 byte, followed by padding for i
    int i;
};
 
struct Derived3 : Base {
    Derived1 c; // derived from Base, occupies sizeof(int) bytes
    int i;
};
 
int main()
{
    // empty base optimization does not apply,
    // base occupies 1 byte, Base member occupies 1 byte
    // followed by 2 bytes of padding to satisfy int alignment requirements
    static_assert(sizeof(Derived2) == 2*sizeof(int));
 
    // empty base optimization does not apply,
    // base takes up at least 1 byte plus the padding
    // to satisfy alignment requirement of the first member (whose
    // alignment is the same as int)
    static_assert(sizeof(Derived3) == 3*sizeof(int));
}
```

以下来源同上：

> If multiple inheritance occurs, then the specific optimizations are compiler specific. In MSVC, the null base class optimization is applied with and only with the last null base class, the rest of the null base classes are not applied with the null base optimization and one byte is allocated. In GCC, no matter how many empty base classes exist, the empty base class applies the empty base class optimization without allocating any space and the empty base class address is the same as the first address of the derived class object.
>
> The empty member subobjects are permitted to be optimized out just like the empty bases if they use the attribute `[[no_unique_address]]`. Taking the address of such member results in an address that may equal the address of some other member of the same object. (C++20)

简单翻译如下：

> 如果存在多重继承时，不同的编译器有不同的优化。对于MSVC来说……。对于GCC来说，不管有多少空基类存在，EBO适用于说有的空基类，不分配任何的内存。空基类的地址和派生类对象的第一个地址相同。
>
> 如果空成员子对象使用了`[[no_unique_address]]`，就可以被像空基类一样优化掉。对这样的成员取地址可能会等于同对象的其他成员的地址。

参见例子（来源同上）：

``` cpp
struct Empty {}; // empty class
 
struct X {
    int i;
    [[no_unique_address]] Empty e;
};
 
int main()
{
    // the size of any object of empty class type is at least 1
    static_assert(sizeof(Empty) >= 1);
 
    // empty member optimized out:
    static_assert(sizeof(X) == sizeof(int));
}
```

## 3 问题

- 对`tuple_cat`的实现仍然有点儿疑问？