# unique ptr

## 1 主要代码

### 1.1 `default_delete`

``` cpp
/// Primary template of default_delete, used by unique_ptr for single objects
template <typename _Tp>
struct default_delete {
    /// Default constructor
    constexpr default_delete() noexcept = default;

    /** @brief Converting constructor.
     *
     * Allows conversion from a deleter for objects of another type, `_Up`,
     * only if `_Up*` is convertible to `_Tp*`.
     */
    template <typename _Up, typename = _Require<is_convertible<_Up*, _Tp*>>>
    default_delete(const default_delete<_Up>&) noexcept {}

    /// Calls `delete __ptr`
    void operator()(_Tp* __ptr) const {
        static_assert(!is_void<_Tp>::value, "can't delete pointer to incomplete type");
        static_assert(sizeof(_Tp) > 0, "can't delete pointer to incomplete type");
        delete __ptr;
    }
};

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// DR 740 - omit specialization for array objects with a compile time length

/// Specialization of default_delete for arrays, used by `unique_ptr<T[]>`
template <typename _Tp>
struct default_delete<_Tp[]> {
public:
    /// Default constructor
    constexpr default_delete() noexcept = default;

    /** @brief Converting constructor.
     *
     * Allows conversion from a deleter for arrays of another type, such as
     * a const-qualified version of `_Tp`.
     *
     * Conversions from types derived from `_Tp` are not allowed because
     * it is undefined to `delete[]` an array of derived types through a
     * pointer to the base type.
     */
    template <typename _Up, typename = _Require<is_convertible<_Up (*)[], _Tp (*)[]>>>
    default_delete(const default_delete<_Up[]>&) noexcept {}

    /// Calls `delete[] __ptr`
    template <typename _Up>
    typename enable_if<is_convertible<_Up (*)[], _Tp (*)[]>::value>::type operator()(_Up* __ptr) const {
        static_assert(sizeof(_Tp) > 0, "can't delete pointer to incomplete type");
        delete[] __ptr;
    }
};
```

类`default_delete`是为`unique_ptr`设计的默认的deleter，即当使用者不提供定制的时候，就用这个。对单个对象和数组进行了实现，核心实现就是`delete ptr`、`delete[] ptr`。

### 1.2 `__uniq_ptr_impl`

``` cpp
// Manages the pointer and deleter of a unique_ptr
template <typename _Tp, typename _Dp>
class __uniq_ptr_impl {
    template <typename _Up, typename _Ep, typename = void>
    struct _Ptr {
        using type = _Up*;
    };

    template <typename _Up, typename _Ep>
    struct _Ptr<_Up, _Ep, __void_t<typename remove_reference<_Ep>::type::pointer>> {
        using type = typename remove_reference<_Ep>::type::pointer;
    };

public:
    using _DeleterConstraint = enable_if<__and_<__not_<is_pointer<_Dp>>, is_default_constructible<_Dp>>::value>;

    using pointer = typename _Ptr<_Tp, _Dp>::type;

    static_assert(!is_rvalue_reference<_Dp>::value,
                  "unique_ptr's deleter type must be a function object type"
                  " or an lvalue reference type");

    __uniq_ptr_impl() = default;
    __uniq_ptr_impl(pointer __p) : _M_t() { _M_ptr() = __p; }

    template <typename _Del>
    __uniq_ptr_impl(pointer __p, _Del&& __d) : _M_t(__p, std::forward<_Del>(__d)) {}

    __uniq_ptr_impl(__uniq_ptr_impl&& __u) noexcept : _M_t(std::move(__u._M_t)) { __u._M_ptr() = nullptr; }

    __uniq_ptr_impl& operator=(__uniq_ptr_impl&& __u) noexcept {
        reset(__u.release());
        _M_deleter() = std::forward<_Dp>(__u._M_deleter());
        return *this;
    }

    pointer& _M_ptr() { return std::get<0>(_M_t); }
    pointer _M_ptr() const { return std::get<0>(_M_t); }
    _Dp& _M_deleter() { return std::get<1>(_M_t); }
    const _Dp& _M_deleter() const { return std::get<1>(_M_t); }

    void reset(pointer __p) noexcept {
        const pointer __old_p = _M_ptr();
        _M_ptr() = __p;
        if (__old_p) _M_deleter()(__old_p);
    }

    pointer release() noexcept {
        pointer __p = _M_ptr();
        _M_ptr() = nullptr;
        return __p;
    }

    void swap(__uniq_ptr_impl& __rhs) noexcept {
        using std::swap;
        swap(this->_M_ptr(), __rhs._M_ptr());
        swap(this->_M_deleter(), __rhs._M_deleter());
    }

private:
    tuple<pointer, _Dp> _M_t;
};
```

- 唯一的成员变量是tuple，第一个元素类型是持有的指针的类型，`_Dp`是deleter的类型。
- `_M_ptr`、`_M_deleter`函数：访问上述tuple的两个元素。
- 移动构造函数：用另一个的`_M_t`来构造this的`_M_t`，并将另一个的pointer置为nullptr。
- `release`函数：将this的pointer置为nullptr，并返回此pointer。
- `reset`函数：设置this的pointer为`__p`，并将`__old_p`使用deleter来销毁。
- `swap`函数：对`_M_t`的每一个元素调用`std::swap`。
- 移动赋值函数：将this的pointer销毁，并设置为另一个pointer，然后将另一个的pointer设为nullptr，

### 1.3 `__uniq_ptr_data`

``` cpp
// Defines move construction + assignment as either defaulted or deleted.
template <typename _Tp, typename _Dp, bool = is_move_constructible<_Dp>::value,
          bool = is_move_assignable<_Dp>::value>
struct __uniq_ptr_data : __uniq_ptr_impl<_Tp, _Dp> {
    using __uniq_ptr_impl<_Tp, _Dp>::__uniq_ptr_impl;
    __uniq_ptr_data(__uniq_ptr_data&&) = default;
    __uniq_ptr_data& operator=(__uniq_ptr_data&&) = default;
};

template <typename _Tp, typename _Dp>
struct __uniq_ptr_data<_Tp, _Dp, true, false> : __uniq_ptr_impl<_Tp, _Dp> {
    using __uniq_ptr_impl<_Tp, _Dp>::__uniq_ptr_impl;
    __uniq_ptr_data(__uniq_ptr_data&&) = default;
    __uniq_ptr_data& operator=(__uniq_ptr_data&&) = delete;
};

template <typename _Tp, typename _Dp>
struct __uniq_ptr_data<_Tp, _Dp, false, true> : __uniq_ptr_impl<_Tp, _Dp> {
    using __uniq_ptr_impl<_Tp, _Dp>::__uniq_ptr_impl;
    __uniq_ptr_data(__uniq_ptr_data&&) = delete;
    __uniq_ptr_data& operator=(__uniq_ptr_data&&) = default;
};

template <typename _Tp, typename _Dp>
struct __uniq_ptr_data<_Tp, _Dp, false, false> : __uniq_ptr_impl<_Tp, _Dp> {
    using __uniq_ptr_impl<_Tp, _Dp>::__uniq_ptr_impl;
    __uniq_ptr_data(__uniq_ptr_data&&) = delete;
    __uniq_ptr_data& operator=(__uniq_ptr_data&&) = delete;
};
```

`__uniq_ptr_data`继承自`__uniq_ptr_impl`，对于是否可以移动构造、移动赋值有分别不同的特化形式。

### 1.4 `unique_ptr`

#### 一般形式的`unique_ptr`

``` cpp
/// 20.7.1.2 unique_ptr for single objects.
template <typename _Tp, typename _Dp = default_delete<_Tp>>
class unique_ptr {
    template <typename _Up>
    using _DeleterConstraint = typename __uniq_ptr_impl<_Tp, _Up>::_DeleterConstraint::type;

    __uniq_ptr_data<_Tp, _Dp> _M_t;

public:
    using pointer = typename __uniq_ptr_impl<_Tp, _Dp>::pointer;
    using element_type = _Tp;
    using deleter_type = _Dp;

private:
    // helper template for detecting a safe conversion from another
    // unique_ptr
    template <typename _Up, typename _Ep>
    using __safe_conversion_up =
        __and_<is_convertible<typename unique_ptr<_Up, _Ep>::pointer, pointer>, __not_<is_array<_Up>>>;

public:
    // Constructors.

    /// Default constructor, creates a unique_ptr that owns nothing.
    template <typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
    constexpr unique_ptr() noexcept : _M_t() {}

    /** Takes ownership of a pointer.
     *
     * @param __p  A pointer to an object of @c element_type
     *
     * The deleter will be value-initialized.
     */
    template <typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
    explicit unique_ptr(pointer __p) noexcept : _M_t(__p) {}

    /** Takes ownership of a pointer.
     *
     * @param __p  A pointer to an object of @c element_type
     * @param __d  A reference to a deleter.
     *
     * The deleter will be initialized with @p __d
     */
    template <typename _Del = deleter_type, typename = _Require<is_copy_constructible<_Del>>>
    unique_ptr(pointer __p, const deleter_type& __d) noexcept : _M_t(__p, __d) {}

    /** Takes ownership of a pointer.
     *
     * @param __p  A pointer to an object of @c element_type
     * @param __d  An rvalue reference to a (non-reference) deleter.
     *
     * The deleter will be initialized with @p std::move(__d)
     */
    template <typename _Del = deleter_type, typename = _Require<is_move_constructible<_Del>>>
    unique_ptr(pointer __p, __enable_if_t<!is_lvalue_reference<_Del>::value, _Del&&> __d) noexcept
        : _M_t(__p, std::move(__d)) {}

    template <typename _Del = deleter_type, typename _DelUnref = typename remove_reference<_Del>::type>
    unique_ptr(pointer, __enable_if_t<is_lvalue_reference<_Del>::value, _DelUnref&&>) = delete;

    /// Creates a unique_ptr that owns nothing.
    template <typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
    constexpr unique_ptr(nullptr_t) noexcept : _M_t() {}

    // Move constructors.

    /// Move constructor.
    unique_ptr(unique_ptr&&) = default;

    /** @brief Converting constructor from another type
     *
     * Requires that the pointer owned by @p __u is convertible to the
     * type of pointer owned by this object, @p __u does not own an array,
     * and @p __u has a compatible deleter type.
     */
    template <typename _Up, typename _Ep,
              typename = _Require<__safe_conversion_up<_Up, _Ep>,
                                  typename conditional<is_reference<_Dp>::value, is_same<_Ep, _Dp>,
                                                       is_convertible<_Ep, _Dp>>::type>>
    unique_ptr(unique_ptr<_Up, _Ep>&& __u) noexcept : _M_t(__u.release(), std::forward<_Ep>(__u.get_deleter())) {}

    /// Destructor, invokes the deleter if the stored pointer is not null.
    ~unique_ptr() noexcept {
        static_assert(__is_invocable<deleter_type&, pointer>::value,
                      "unique_ptr's deleter must be invocable with a pointer");
        auto& __ptr = _M_t._M_ptr();
        if (__ptr != nullptr) get_deleter()(std::move(__ptr));
        __ptr = pointer();
    }

    // Assignment.

    /** @brief Move assignment operator.
     *
     * Invokes the deleter if this object owns a pointer.
     */
    unique_ptr& operator=(unique_ptr&&) = default;

    /** @brief Assignment from another type.
     *
     * @param __u  The object to transfer ownership from, which owns a
     *             convertible pointer to a non-array object.
     *
     * Invokes the deleter if this object owns a pointer.
     */
    template <typename _Up, typename _Ep>
    typename enable_if<__and_<__safe_conversion_up<_Up, _Ep>, is_assignable<deleter_type&, _Ep&&>>::value,
                       unique_ptr&>::type
    operator=(unique_ptr<_Up, _Ep>&& __u) noexcept {
        reset(__u.release());
        get_deleter() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

    /// Reset the %unique_ptr to empty, invoking the deleter if necessary.
    unique_ptr& operator=(nullptr_t) noexcept {
        reset();
        return *this;
    }

    // Observers.

    /// Dereference the stored pointer.
    typename add_lvalue_reference<element_type>::type operator*() const {
        __glibcxx_assert(get() != pointer());
        return *get();
    }

    /// Return the stored pointer.
    pointer operator->() const noexcept {
        _GLIBCXX_DEBUG_PEDASSERT(get() != pointer());
        return get();
    }

    /// Return the stored pointer.
    pointer get() const noexcept { return _M_t._M_ptr(); }

    /// Return a reference to the stored deleter.
    deleter_type& get_deleter() noexcept { return _M_t._M_deleter(); }

    /// Return a reference to the stored deleter.
    const deleter_type& get_deleter() const noexcept { return _M_t._M_deleter(); }

    /// Return @c true if the stored pointer is not null.
    explicit operator bool() const noexcept { return get() == pointer() ? false : true; }

    // Modifiers.

    /// Release ownership of any stored pointer.
    pointer release() noexcept { return _M_t.release(); }

    /** @brief Replace the stored pointer.
     *
     * @param __p  The new pointer to store.
     *
     * The deleter will be invoked if a pointer is already owned.
     */
    void reset(pointer __p = pointer()) noexcept {
        static_assert(__is_invocable<deleter_type&, pointer>::value,
                      "unique_ptr's deleter must be invocable with a pointer");
        _M_t.reset(std::move(__p));
    }

    /// Exchange the pointer and deleter with another object.
    void swap(unique_ptr& __u) noexcept {
        static_assert(__is_swappable<_Dp>::value, "deleter must be swappable");
        _M_t.swap(__u._M_t);
    }

    // Disable copy from lvalue.
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;
};
```

- 唯一成员变量：`__uniq_ptr_data<_Tp, _Dp> _M_t;`。*大家看代码的时候注意：这里虽然叫`_M_t`，但是之前说的`__uniq_ptr_impl`的`_M_t`的上一层，不是一个东西。*
- 构造函数、移动构造函数、移动赋值函数：用传入的pointer和deleter构造`_M_t`。
- 不允许拷贝构造或拷贝赋值。
- `get`、`get_deleter`函数：分别返回相应的pointer和deleter。
- `reset`、`release`函数：都是对`_M_t`调用相应的函数，即调用的是`__uniq_ptr_impl`相应的函数。
- `pointer()`：相当于是nullptr。
- 析构函数：用自己的deleter销毁pointer，并将pointer置为nullptr。
- 用nullptr的赋值函数：调用`reset`函数。即将自己的pointer用deleter销毁，并赋值为nullptr。

#### 为数组特化的`unique_ptr`

这个在设计上和一般形式的`unique_ptr`没在代码设计上有特别明显的区别，因此这里不再过多赘述。

#### 比较函数

`unique_ptr`相关的比较函数，实际上都是其背后指针的比较。这里仅摘录部分如下。关于指针比较请参见备注2.1.

``` cpp
/// Equality operator for unique_ptr objects, compares the owned pointers
template <typename _Tp, typename _Dp, typename _Up, typename _Ep>
_GLIBCXX_NODISCARD inline bool operator==(const unique_ptr<_Tp, _Dp>& __x, const unique_ptr<_Up, _Ep>& __y) {
    return __x.get() == __y.get();
}

/// Inequality operator for unique_ptr objects, compares the owned pointers
template <typename _Tp, typename _Dp, typename _Up, typename _Ep>
_GLIBCXX_NODISCARD inline bool operator!=(const unique_ptr<_Tp, _Dp>& __x, const unique_ptr<_Up, _Ep>& __y) {
    return __x.get() != __y.get();
}

/// unique_ptr comparison with nullptr
template <typename _Tp, typename _Dp>
_GLIBCXX_NODISCARD inline bool operator!=(nullptr_t, const unique_ptr<_Tp, _Dp>& __x) noexcept {
    return (bool)__x;
}
#endif  // three way comparison

/// Relational operator for unique_ptr objects, compares the owned pointers
template <typename _Tp, typename _Dp, typename _Up, typename _Ep>
_GLIBCXX_NODISCARD inline bool operator<(const unique_ptr<_Tp, _Dp>& __x, const unique_ptr<_Up, _Ep>& __y) {
    typedef typename std::common_type<typename unique_ptr<_Tp, _Dp>::pointer,
                                      typename unique_ptr<_Up, _Ep>::pointer>::type _CT;
    return std::less<_CT>()(__x.get(), __y.get());
}

/// unique_ptr comparison with nullptr
template <typename _Tp, typename _Dp>
_GLIBCXX_NODISCARD inline bool operator<(const unique_ptr<_Tp, _Dp>& __x, nullptr_t) {
    return std::less<typename unique_ptr<_Tp, _Dp>::pointer>()(__x.get(), nullptr);
}
```

### 1.5 其他

``` cpp
/// @cond undocumented
template <typename _Up, typename _Ptr = typename _Up::pointer, bool = __poison_hash<_Ptr>::__enable_hash_call>
struct __uniq_ptr_hash
#if !_GLIBCXX_INLINE_VERSION
    : private __poison_hash<_Ptr>
#endif
{
    size_t operator()(const _Up& __u) const noexcept(noexcept(std::declval<hash<_Ptr>>()(std::declval<_Ptr>()))) {
        return hash<_Ptr>()(__u.get());
    }
};

template <typename _Up, typename _Ptr>
struct __uniq_ptr_hash<_Up, _Ptr, false> : private __poison_hash<_Ptr> {};
/// @endcond

/// std::hash specialization for unique_ptr.
template <typename _Tp, typename _Dp>
struct hash<unique_ptr<_Tp, _Dp>> : public __hash_base<size_t, unique_ptr<_Tp, _Dp>>,
                                    public __uniq_ptr_hash<unique_ptr<_Tp, _Dp>> {};

#if __cplusplus >= 201402L
/// @relates unique_ptr @{
#define __cpp_lib_make_unique 201304

/// @cond undocumented

template <typename _Tp>
struct _MakeUniq {
    typedef unique_ptr<_Tp> __single_object;
};

template <typename _Tp>
struct _MakeUniq<_Tp[]> {
    typedef unique_ptr<_Tp[]> __array;
};

template <typename _Tp, size_t _Bound>
struct _MakeUniq<_Tp[_Bound]> {
    struct __invalid_type {};
};

/// @endcond

/// std::make_unique for single objects
template <typename _Tp, typename... _Args>
inline typename _MakeUniq<_Tp>::__single_object make_unique(_Args && ... __args) {
    return unique_ptr<_Tp>(new _Tp(std::forward<_Args>(__args)...));
}

/// std::make_unique for arrays of unknown bound
template <typename _Tp>
inline typename _MakeUniq<_Tp>::__array make_unique(size_t __num) {
    return unique_ptr<_Tp>(new remove_extent_t<_Tp>[__num]());
}

/// Disable std::make_unique for arrays of known bound
template <typename _Tp, typename... _Args>
typename _MakeUniq<_Tp>::__invalid_type make_unique(_Args && ...) = delete;

#if __cplusplus > 201703L
/// std::make_unique_for_overwrite for single objects
template <typename _Tp>
inline typename _MakeUniq<_Tp>::__single_object make_unique_for_overwrite() {
    return unique_ptr<_Tp>(new _Tp);
}

/// std::make_unique_for_overwrite for arrays of unknown bound
template <typename _Tp>
inline typename _MakeUniq<_Tp>::__array make_unique_for_overwrite(size_t __n) {
    return unique_ptr<_Tp>(new remove_extent_t<_Tp>[__n]);
}

/// Disable std::make_unique_for_overwrite for arrays of known bound
template <typename _Tp, typename... _Args>
typename _MakeUniq<_Tp>::__invalid_type make_unique_for_overwrite(_Args && ...) = delete;
#endif  // C++20
```

- 对`unique_ptr`的hash特化形式，继承自`posion_hash`。
- `make_unique`函数，主要是利用传入的参数调用`unique_ptr`的构造函数。

## 2 备注

### 2.1 指针比较

以下参考自[网页](https://stackoverflow.com/a/9086675/8023037)：

两个相同类型指针可以比较是否相等：

1. 两者是**相等**的当且仅当，同为null、指向同一函数、或同一地址。

两个相同类型指针可以比较大小：

1. 如果两个指针p、q是同一个类型，指向同一个对象、函数、数组的尾、或null，则<=、>=返回true，>、<返回false。
2. 如果两个指针p、q是同一个类型，指向不同的对象（不是同一对象的成员或相同数组的元素）、不同函数、或只有一个是null，则>、<、<=、>=的结果unspecified。
3. 如果两个指针指向同一对象的非静态数据成员、此类成员的数组的子对象，则指向后声明的成员的比较起来大。（在有相同访问控制、非union的情况下）
4. 如果两个指针指向同一个类的有着不同访问控制的非静态数据成员，则结果unspecified。
5. 如果两个指针指向同一union的非静态数据成员，则比较结果为相同。如果指向同一数组的不同元素、或一个指向数组的尾之后，则有更大下标的比较起来更大。
6. 其他类型的指针比较是unspecified。
