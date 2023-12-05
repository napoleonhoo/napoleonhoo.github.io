# shared ptr & weak ptr

## 1 主要代码

### 1.1 count

#### `_Mutex_base`

``` cpp
using __gnu_cxx::__default_lock_policy;
using __gnu_cxx::_Lock_policy;
using __gnu_cxx::_S_atomic;
using __gnu_cxx::_S_mutex;
using __gnu_cxx::_S_single;

// Empty helper class except when the template argument is _S_mutex.
template <_Lock_policy _Lp>
class _Mutex_base {
protected:
    // The atomic policy uses fully-fenced builtins, single doesn't care.
    enum { _S_need_barriers = 0 };
};

template <>
class _Mutex_base<_S_mutex> : public __gnu_cxx::__mutex {
protected:
    // This policy is used when atomic builtins are not available.
    // The replacement atomic operations might not have the necessary
    // memory barriers.
    enum { _S_need_barriers = 1 };
};
```

`_Lock_policy`相关参见备注2.3.

#### `_Sp_counted_base`

``` cpp
template <_Lock_policy _Lp = __default_lock_policy>
class _Sp_counted_base : public _Mutex_base<_Lp> {
public:
    _Sp_counted_base() noexcept : _M_use_count(1), _M_weak_count(1) {}

    virtual ~_Sp_counted_base() noexcept {}

    // Called when _M_use_count drops to zero, to release the resources
    // managed by *this.
    virtual void _M_dispose() noexcept = 0;

    // Called when _M_weak_count drops to zero.
    virtual void _M_destroy() noexcept { delete this; }

    virtual void* _M_get_deleter(const std::type_info&) noexcept = 0;

    void _M_add_ref_copy() { __gnu_cxx::__atomic_add_dispatch(&_M_use_count, 1); }

    void _M_add_ref_lock() {
        if (!_M_add_ref_lock_nothrow()) __throw_bad_weak_ptr();
    }

    bool _M_add_ref_lock_nothrow() noexcept;

    void _M_release() noexcept {
        // Be race-detector-friendly.  For more info see bits/c++config.
        _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_use_count);
        if (__gnu_cxx::__exchange_and_add_dispatch(&_M_use_count, -1) == 1) {
            _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_use_count);
            _M_dispose();
            // There must be a memory barrier between dispose() and destroy()
            // to ensure that the effects of dispose() are observed in the
            // thread that runs destroy().
            // See http://gcc.gnu.org/ml/libstdc++/2005-11/msg00136.html
            if (_Mutex_base<_Lp>::_S_need_barriers) {
                __atomic_thread_fence(__ATOMIC_ACQ_REL);
            }

            // Be race-detector-friendly.  For more info see bits/c++config.
            _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_weak_count);
            if (__gnu_cxx::__exchange_and_add_dispatch(&_M_weak_count, -1) == 1) {
                _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_weak_count);
                _M_destroy();
            }
        }
    }

    void _M_weak_add_ref() noexcept { __gnu_cxx::__atomic_add_dispatch(&_M_weak_count, 1); }

    void _M_weak_release() noexcept {
        // Be race-detector-friendly. For more info see bits/c++config.
        _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_weak_count);
        if (__gnu_cxx::__exchange_and_add_dispatch(&_M_weak_count, -1) == 1) {
            _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_weak_count);
            if (_Mutex_base<_Lp>::_S_need_barriers) {
                // See _M_release(),
                // destroy() must observe results of dispose()
                __atomic_thread_fence(__ATOMIC_ACQ_REL);
            }
            _M_destroy();
        }
    }

    long _M_get_use_count() const noexcept {
        // No memory barrier is used here so there is no synchronization
        // with other threads.
        return __atomic_load_n(&_M_use_count, __ATOMIC_RELAXED);
    }

private:
    _Sp_counted_base(_Sp_counted_base const&) = delete;
    _Sp_counted_base& operator=(_Sp_counted_base const&) = delete;

    _Atomic_word _M_use_count;   // #shared
    _Atomic_word _M_weak_count;  // #weak + (#shared != 0)
};

template <>
inline bool _Sp_counted_base<_S_single>::_M_add_ref_lock_nothrow() noexcept {
    if (_M_use_count == 0) return false;
    ++_M_use_count;
    return true;
}

template <>
inline bool _Sp_counted_base<_S_mutex>::_M_add_ref_lock_nothrow() noexcept {
    __gnu_cxx::__scoped_lock sentry(*this);
    if (__gnu_cxx::__exchange_and_add_dispatch(&_M_use_count, 1) == 0) {
        _M_use_count = 0;
        return false;
    }
    return true;
}

template <>
inline bool _Sp_counted_base<_S_atomic>::_M_add_ref_lock_nothrow() noexcept {
    // Perform lock-free add-if-not-zero operation.
    _Atomic_word __count = _M_get_use_count();
    do {
        if (__count == 0) return false;
        // Replace the current counter value with the old value + 1, as
        // long as it's not changed meanwhile.
    } while (!__atomic_compare_exchange_n(&_M_use_count, &__count, __count + 1, true, __ATOMIC_ACQ_REL,
                                          __ATOMIC_RELAXED));
    return true;
}

template <>
inline void _Sp_counted_base<_S_single>::_M_add_ref_copy() {
    ++_M_use_count;
}

template <>
inline void _Sp_counted_base<_S_single>::_M_release() noexcept {
    if (--_M_use_count == 0) {
        _M_dispose();
        if (--_M_weak_count == 0) _M_destroy();
    }
}

template <>
inline void _Sp_counted_base<_S_single>::_M_weak_add_ref() noexcept {
    ++_M_weak_count;
}

template <>
inline void _Sp_counted_base<_S_single>::_M_weak_release() noexcept {
    if (--_M_weak_count == 0) _M_destroy();
}

template <>
inline long _Sp_counted_base<_S_single>::_M_get_use_count() const noexcept {
    return _M_use_count;
}
```

- `__exchange_and_add_dispatch`函数参见备注2.2.
- 构造函数将两个成员变量`_M_use_count`、`_M_weak_count`初始化为1.这里的逻辑是，当它创建的时候，自己本身就是一个引用计数。
- `_M_dispose`函数是纯虚函数，当`_M_use_count`为0时，释放this持有的资源。
- `_M_destroy`函数默认删除this，当`_M_weak_count`为0时调用。
- `_M_add_ref_copy`函数，对`_M_use_count + 1`，是原子操作。
- `_M_add_ref_lock`函数，主要逻辑仍然是`_M_use_count + 1`。和`_M_add_ref_copy`的区别是对不同`_Lock_policy`有不同的实现，包含直接加、原子操作加、加锁。
- `_M_release`函数，当`_M_use_count-1=0`时，即`_M_use_count`为0时，调用`_M_dispose`。并当`_M_use_count-1=0`，即`_M_weak_count`为0时，调用`_M_destroy`。
- `_M_weak_add_ref`：对`_M_weak_count + 1`，是原子操作。
- `_M_weak_release`：只对`_M_weak_count - 1`。
- 以上两个函数都有`_Lock_policy=_M_single`时的重载形式。

#### `_Sp_counted_ptr`

``` cpp
// Counted ptr with no deleter or allocator support
template <typename _Ptr, _Lock_policy _Lp>
class _Sp_counted_ptr final : public _Sp_counted_base<_Lp> {
public:
    explicit _Sp_counted_ptr(_Ptr __p) noexcept : _M_ptr(__p) {}

    virtual void _M_dispose() noexcept { delete _M_ptr; }

    virtual void _M_destroy() noexcept { delete this; }

    virtual void* _M_get_deleter(const std::type_info&) noexcept { return nullptr; }

    _Sp_counted_ptr(const _Sp_counted_ptr&) = delete;
    _Sp_counted_ptr& operator=(const _Sp_counted_ptr&) = delete;

private:
    _Ptr _M_ptr;
};

template <>
inline void _Sp_counted_ptr<nullptr_t, _S_single>::_M_dispose() noexcept {}

template <>
inline void _Sp_counted_ptr<nullptr_t, _S_mutex>::_M_dispose() noexcept {}

template <>
inline void _Sp_counted_ptr<nullptr_t, _S_atomic>::_M_dispose() noexcept {}

```

不支持deleter和allocator的counted ptr。

- `_M_dispose`默认行为是delete `_M_ptr`，`_M_destroy`默认行为是delete this。
- 对于`_Ptr=nullptr_t`时，不同`_Lock_policy`的`_M_dispose`的行为都是空的。

#### `_Sp_ebo_helper`

``` cpp
template <int _Nm, typename _Tp, bool __use_ebo = !__is_final(_Tp) && __is_empty(_Tp)>
struct _Sp_ebo_helper;

/// Specialization using EBO.
template <int _Nm, typename _Tp>
struct _Sp_ebo_helper<_Nm, _Tp, true> : private _Tp {
    explicit _Sp_ebo_helper(const _Tp& __tp) : _Tp(__tp) {}
    explicit _Sp_ebo_helper(_Tp&& __tp) : _Tp(std::move(__tp)) {}

    static _Tp& _S_get(_Sp_ebo_helper& __eboh) { return static_cast<_Tp&>(__eboh); }
};

/// Specialization not using EBO.
template <int _Nm, typename _Tp>
struct _Sp_ebo_helper<_Nm, _Tp, false> {
    explicit _Sp_ebo_helper(const _Tp& __tp) : _M_tp(__tp) {}
    explicit _Sp_ebo_helper(_Tp&& __tp) : _M_tp(std::move(__tp)) {}

    static _Tp& _S_get(_Sp_ebo_helper& __eboh) { return __eboh._M_tp; }

private:
    _Tp _M_tp;
};
```

在类型`_Tp`为final且empty的时候，使用ebo优化的类，即模板参数`__use_ebo`为true的时候。

- 对于“空”类型，不存在私有变量，`_S_get`返回强转为`_Tp`类型的入参`_Sp_ebo_helper`。
- 对于“非空”类型，有一个`_Tp`类型的私有变量：`_M_tp`。`_S_get`返回这个变量。

#### `_Sp_counted_deleter`

``` cpp
// Support for custom deleter and/or allocator
template <typename _Ptr, typename _Deleter, typename _Alloc, _Lock_policy _Lp>
class _Sp_counted_deleter final : public _Sp_counted_base<_Lp> {
    class _Impl : _Sp_ebo_helper<0, _Deleter>, _Sp_ebo_helper<1, _Alloc> {
        typedef _Sp_ebo_helper<0, _Deleter> _Del_base;
        typedef _Sp_ebo_helper<1, _Alloc> _Alloc_base;

    public:
        _Impl(_Ptr __p, _Deleter __d, const _Alloc& __a) noexcept
            : _Del_base(std::move(__d)), _Alloc_base(__a), _M_ptr(__p) {}

        _Deleter& _M_del() noexcept { return _Del_base::_S_get(*this); }
        _Alloc& _M_alloc() noexcept { return _Alloc_base::_S_get(*this); }

        _Ptr _M_ptr;
    };

public:
    using __allocator_type = __alloc_rebind<_Alloc, _Sp_counted_deleter>;

    // __d(__p) must not throw.
    _Sp_counted_deleter(_Ptr __p, _Deleter __d) noexcept : _M_impl(__p, std::move(__d), _Alloc()) {}

    // __d(__p) must not throw.
    _Sp_counted_deleter(_Ptr __p, _Deleter __d, const _Alloc& __a) noexcept : _M_impl(__p, std::move(__d), __a) {}

    ~_Sp_counted_deleter() noexcept {}

    virtual void _M_dispose() noexcept { _M_impl._M_del()(_M_impl._M_ptr); }

    virtual void _M_destroy() noexcept {
        __allocator_type __a(_M_impl._M_alloc());
        __allocated_ptr<__allocator_type> __guard_ptr{__a, this};
        this->~_Sp_counted_deleter();
    }

    virtual void* _M_get_deleter(const type_info& __ti [[__gnu__::__unused__]]) noexcept {
#if __cpp_rtti
        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2400. shared_ptr's get_deleter() should use addressof()
        return __ti == typeid(_Deleter) ? std::__addressof(_M_impl._M_del()) : nullptr;
#else
        return nullptr;
#endif
    }

private:
    _Impl _M_impl;
};
```

`_Impl`继承自`_Sp_ebo_helper<0, _Deleter>`作为`_Del_base`、`_Sp_ebo_helper<1, _Alloc>`作为`_Alloc_base`。

- `_M_ptr`作为私有变量是被管理的对象的指针。
- `_M_del`从`_Del_base`获取一个`_Deleter`变量。
- `_M_alloc`从`_Alloc_base`获取一个`_Alloc`类型变量。

`_Sp_counted_deleter`继承自`_Sp_counted_base`。

- 唯一的私有变量是`_Impl`类型的。
- 构造函数主要是将传入的数据构造一个`_Impl`。
- `_M_dispose`函数对`_Ptr`调用`_M_del()`。
- `_M_destroy`函数对this调用析构函数。
- `_M_get_delter`，当传入的`type_info`类型的`__ti`和`_Deleter`是同样的类型的时候，返回`_M_del()`的地址，否则返回nullptr。

#### `_Sp_counted_ptr_inplace`

``` cpp
struct _Sp_make_shared_tag {
private:
    template <typename _Tp, typename _Alloc, _Lock_policy _Lp>
    friend class _Sp_counted_ptr_inplace;

    static const type_info& _S_ti() noexcept _GLIBCXX_VISIBILITY(default) {
        alignas(type_info) static constexpr char __tag[sizeof(type_info)] = {};
        return reinterpret_cast<const type_info&>(__tag);
    }

    static bool _S_eq(const type_info&) noexcept;
};

template <typename _Alloc>
struct _Sp_alloc_shared_tag {
    const _Alloc& _M_a;
};

template <typename _Tp, typename _Alloc, _Lock_policy _Lp>
class _Sp_counted_ptr_inplace final : public _Sp_counted_base<_Lp> {
    class _Impl : _Sp_ebo_helper<0, _Alloc> {
        typedef _Sp_ebo_helper<0, _Alloc> _A_base;

    public:
        explicit _Impl(_Alloc __a) noexcept : _A_base(__a) {}

        _Alloc& _M_alloc() noexcept { return _A_base::_S_get(*this); }

        __gnu_cxx::__aligned_buffer<_Tp> _M_storage;
    };

public:
    using __allocator_type = __alloc_rebind<_Alloc, _Sp_counted_ptr_inplace>;

    // Alloc parameter is not a reference so doesn't alias anything in __args
    template <typename... _Args>
    _Sp_counted_ptr_inplace(_Alloc __a, _Args&&... __args) : _M_impl(__a) {
        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2070.  allocate_shared should use allocator_traits<A>::construct
        allocator_traits<_Alloc>::construct(__a, _M_ptr(),
                                            std::forward<_Args>(__args)...);  // might throw
    }

    ~_Sp_counted_ptr_inplace() noexcept {}

    virtual void _M_dispose() noexcept { allocator_traits<_Alloc>::destroy(_M_impl._M_alloc(), _M_ptr()); }

    // Override because the allocator needs to know the dynamic type
    virtual void _M_destroy() noexcept {
        __allocator_type __a(_M_impl._M_alloc());
        __allocated_ptr<__allocator_type> __guard_ptr{__a, this};
        this->~_Sp_counted_ptr_inplace();
    }

private:
    friend class __shared_count<_Lp>;  // To be able to call _M_ptr().

    // No longer used, but code compiled against old libstdc++ headers
    // might still call it from __shared_ptr ctor to get the pointer out.
    virtual void* _M_get_deleter(const std::type_info& __ti) noexcept override {
        auto __ptr = const_cast<typename remove_cv<_Tp>::type*>(_M_ptr());
        // Check for the fake type_info first, so we don't try to access it
        // as a real type_info object. Otherwise, check if it's the real
        // type_info for this class. With RTTI enabled we can check directly,
        // or call a library function to do it.
        if (&__ti == &_Sp_make_shared_tag::_S_ti() ||
#if __cpp_rtti
            __ti == typeid(_Sp_make_shared_tag)
#else
            _Sp_make_shared_tag::_S_eq(__ti)
#endif
        )
            return __ptr;
        return nullptr;
    }

    _Tp* _M_ptr() noexcept { return _M_impl._M_storage._M_ptr(); }

    _Impl _M_impl;
};
```

这里的“inplace”指的是inplace构造，看到`_Impl`类中为一个成员变量`_M_storage`，并且`_Sp_counted_ptr_inplace`唯一的变量`_M_impl`。构造函数就是利用传入的allocator和参数在`_M_storage`上构建对象。`_M_dispose`函数就是删除`_M_storage`上的对象。

#### `_shared_count`

``` cpp
// The default deleter for shared_ptr<T[]> and shared_ptr<T[N]>.
struct __sp_array_delete {
    template <typename _Yp>
    void operator()(_Yp* __p) const {
        delete[] __p;
    }
};

template <_Lock_policy _Lp>
class __shared_count {
    template <typename _Tp>
    struct __not_alloc_shared_tag {
        using type = void;
    };

    template <typename _Tp>
    struct __not_alloc_shared_tag<_Sp_alloc_shared_tag<_Tp>> {};

public:
    constexpr __shared_count() noexcept : _M_pi(0) {}
	// 构造函数1，最基础的
    template <typename _Ptr>
    explicit __shared_count(_Ptr __p) : _M_pi(0) {
        __try {
            _M_pi = new _Sp_counted_ptr<_Ptr, _Lp>(__p);
        }
        __catch(...) {
            delete __p;
            __throw_exception_again;
        }
    }

    template <typename _Ptr>
    __shared_count(_Ptr __p, /* is_array = */ false_type) : __shared_count(__p) {}

    template <typename _Ptr>
    __shared_count(_Ptr __p, /* is_array = */ true_type)
        : __shared_count(__p, __sp_array_delete{}, allocator<void>()) {}

    template <typename _Ptr, typename _Deleter, typename = typename __not_alloc_shared_tag<_Deleter>::type>
    __shared_count(_Ptr __p, _Deleter __d) : __shared_count(__p, std::move(__d), allocator<void>()) {}
	// 构造函数2，带有delter和alloctor的
    template <typename _Ptr, typename _Deleter, typename _Alloc,
              typename = typename __not_alloc_shared_tag<_Deleter>::type>
    __shared_count(_Ptr __p, _Deleter __d, _Alloc __a) : _M_pi(0) {
        typedef _Sp_counted_deleter<_Ptr, _Deleter, _Alloc, _Lp> _Sp_cd_type;
        __try {
            typename _Sp_cd_type::__allocator_type __a2(__a);
            auto __guard = std::__allocate_guarded(__a2);
            _Sp_cd_type* __mem = __guard.get();
            ::new (__mem) _Sp_cd_type(__p, std::move(__d), std::move(__a));
            _M_pi = __mem;
            __guard = nullptr;
        }
        __catch(...) {
            __d(__p);  // Call _Deleter on __p.
            __throw_exception_again;
        }
    }

    template <typename _Tp, typename _Alloc, typename... _Args>
    __shared_count(_Tp*& __p, _Sp_alloc_shared_tag<_Alloc> __a, _Args&&... __args) {
        typedef _Sp_counted_ptr_inplace<_Tp, _Alloc, _Lp> _Sp_cp_type;
        typename _Sp_cp_type::__allocator_type __a2(__a._M_a);
        auto __guard = std::__allocate_guarded(__a2);
        _Sp_cp_type* __mem = __guard.get();
        auto __pi = ::new (__mem) _Sp_cp_type(__a._M_a, std::forward<_Args>(__args)...);
        __guard = nullptr;
        _M_pi = __pi;
        __p = __pi->_M_ptr();
    }

    // Special case for unique_ptr<_Tp,_Del> to provide the strong guarantee.
    template <typename _Tp, typename _Del>
    explicit __shared_count(std::unique_ptr<_Tp, _Del>&& __r) : _M_pi(0) {
        // _GLIBCXX_RESOLVE_LIB_DEFECTS
        // 2415. Inconsistency between unique_ptr and shared_ptr
        if (__r.get() == nullptr) return;

        using _Ptr = typename unique_ptr<_Tp, _Del>::pointer;
        using _Del2 = typename conditional<is_reference<_Del>::value,
                                           reference_wrapper<typename remove_reference<_Del>::type>, _Del>::type;
        using _Sp_cd_type = _Sp_counted_deleter<_Ptr, _Del2, allocator<void>, _Lp>;
        using _Alloc = allocator<_Sp_cd_type>;
        using _Alloc_traits = allocator_traits<_Alloc>;
        _Alloc __a;
        _Sp_cd_type* __mem = _Alloc_traits::allocate(__a, 1);
        _Alloc_traits::construct(__a, __mem, __r.release(),
                                 __r.get_deleter());  // non-throwing
        _M_pi = __mem;
    }

    // Throw bad_weak_ptr when __r._M_get_use_count() == 0.
    explicit __shared_count(const __weak_count<_Lp>& __r);

    // Does not throw if __r._M_get_use_count() == 0, caller must check.
    explicit __shared_count(const __weak_count<_Lp>& __r, std::nothrow_t) noexcept;

    ~__shared_count() noexcept {
        if (_M_pi != nullptr) _M_pi->_M_release();
    }

    __shared_count(const __shared_count& __r) noexcept : _M_pi(__r._M_pi) {
        if (_M_pi != nullptr) _M_pi->_M_add_ref_copy();
    }

    __shared_count& operator=(const __shared_count& __r) noexcept {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != _M_pi) {
            if (__tmp != nullptr) __tmp->_M_add_ref_copy();
            if (_M_pi != nullptr) _M_pi->_M_release();
            _M_pi = __tmp;
        }
        return *this;
    }

    void _M_swap(__shared_count& __r) noexcept {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        __r._M_pi = _M_pi;
        _M_pi = __tmp;
    }

    long _M_get_use_count() const noexcept { return _M_pi ? _M_pi->_M_get_use_count() : 0; }

    bool _M_unique() const noexcept { return this->_M_get_use_count() == 1; }

    void* _M_get_deleter(const std::type_info& __ti) const noexcept {
        return _M_pi ? _M_pi->_M_get_deleter(__ti) : nullptr;
    }

    bool _M_less(const __shared_count& __rhs) const noexcept {
        return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi);
    }

    bool _M_less(const __weak_count<_Lp>& __rhs) const noexcept {
        return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi);
    }

    // Friend function injected into enclosing namespace and found by ADL
    friend inline bool operator==(const __shared_count& __a, const __shared_count& __b) noexcept {
        return __a._M_pi == __b._M_pi;
    }

private:
    friend class __weak_count<_Lp>;

    _Sp_counted_base<_Lp>* _M_pi;
};

// Now that __weak_count is defined we can define this constructor:
template <_Lock_policy _Lp>
inline __shared_count<_Lp>::__shared_count(const __weak_count<_Lp>& __r) : _M_pi(__r._M_pi) {
    if (_M_pi == nullptr || !_M_pi->_M_add_ref_lock_nothrow()) __throw_bad_weak_ptr();
}

// Now that __weak_count is defined we can define this constructor:
template <_Lock_policy _Lp>
inline __shared_count<_Lp>::__shared_count(const __weak_count<_Lp>& __r, std::nothrow_t) noexcept
    : _M_pi(__r._M_pi) {
    if (_M_pi && !_M_pi->_M_add_ref_lock_nothrow()) _M_pi = nullptr;
}
```

- 唯一的私有变量是`_Sp_counted_base<_Lp>* _M_pi;`，后面会看到这个变量会通过构造函数的入参不同而*动态*绑定到不同的类型上。

- 最基础的构造函数1，传入一个`_Ptr`类型的`__p`，用他来构造一个`_Sp_counted_ptr`对象，将`_M_pi`指向它。
- 构造函数2（入参中含有deleter和allocator的）：用传入的参数构造一个`_Sp_counted_deleter`对象，将`_M_pi`指向它。
- 对于指向array的构造函数、传入了deleter的构造函数，都是调用的构造函数2.
- 对于传入了`_Sp_counted_ptr_inplace`的构造函数，用传入的参数构造一个`_Sp_counted_ptr_inplace`对象，将`_M_pi`指向它。
- 对于传入了`unique_ptr`的构造函数，逻辑没有很特别的地方，类似上面。
- 拷贝构造函数：`this->_M_pi = __r._M_pi`，并增加一个ref。
- 拷贝赋值函数：`__r._M_pi`增加一个ref，`this->_M_pi`释放，并`this->_M_pi = __r._M_pi`。
- `_M_less`函数：对两个指针比较了大小。这确实是可以比较的，见备注2.4.
- 参数为`__weak_count`的拷贝构造函数：`this->_M_pi = __r._M_pi`，并加一个ref。（`__weak_count`定义见下）

#### `__weak_count`

``` cpp
template <_Lock_policy _Lp>
class __weak_count {
public:
    constexpr __weak_count() noexcept : _M_pi(nullptr) {}

    __weak_count(const __shared_count<_Lp>& __r) noexcept : _M_pi(__r._M_pi) {
        if (_M_pi != nullptr) _M_pi->_M_weak_add_ref();
    }

    __weak_count(const __weak_count& __r) noexcept : _M_pi(__r._M_pi) {
        if (_M_pi != nullptr) _M_pi->_M_weak_add_ref();
    }

    __weak_count(__weak_count&& __r) noexcept : _M_pi(__r._M_pi) { __r._M_pi = nullptr; }

    ~__weak_count() noexcept {
        if (_M_pi != nullptr) _M_pi->_M_weak_release();
    }

    __weak_count& operator=(const __shared_count<_Lp>& __r) noexcept {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != nullptr) __tmp->_M_weak_add_ref();
        if (_M_pi != nullptr) _M_pi->_M_weak_release();
        _M_pi = __tmp;
        return *this;
    }

    __weak_count& operator=(const __weak_count& __r) noexcept {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != nullptr) __tmp->_M_weak_add_ref();
        if (_M_pi != nullptr) _M_pi->_M_weak_release();
        _M_pi = __tmp;
        return *this;
    }

    __weak_count& operator=(__weak_count&& __r) noexcept {
        if (_M_pi != nullptr) _M_pi->_M_weak_release();
        _M_pi = __r._M_pi;
        __r._M_pi = nullptr;
        return *this;
    }

    void _M_swap(__weak_count& __r) noexcept {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        __r._M_pi = _M_pi;
        _M_pi = __tmp;
    }

    long _M_get_use_count() const noexcept { return _M_pi != nullptr ? _M_pi->_M_get_use_count() : 0; }

    bool _M_less(const __weak_count& __rhs) const noexcept {
        return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi);
    }

    bool _M_less(const __shared_count<_Lp>& __rhs) const noexcept {
        return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi);
    }

    // Friend function injected into enclosing namespace and found by ADL
    friend inline bool operator==(const __weak_count& __a, const __weak_count& __b) noexcept {
        return __a._M_pi == __b._M_pi;
    }

private:
    friend class __shared_count<_Lp>;

    _Sp_counted_base<_Lp>* _M_pi;
};
```

- 拷贝构造函数：`this->_M_pi = __r._M_pi`，并加一个weak ref。
- 参数为`__shared_count`的拷贝构造函数：逻辑与上相同。
- 移动构造函数：`this->_M_pi = __r._M_pi`，再将`__r._M_pi`置为nullptr。
- 拷贝赋值函数：`__r._M_pi`加一个weak ref，`this->_M_pi`释放，并将`this->_M_pi = __r._M_pi`。
- 参数为`__shared_count`拷贝复制函数：逻辑与上相同。
- 移动赋值函数：`this->_M_pi`释放，`this->_M_pi = __r._M_pi`，并将`__r._M_pi`置为nullptr。
- `_M_less`函数，调用`std::less`比较两个类的`_M_pi`。

#### `__shared_count`和`__weak_count`小结：各类构造时的一些计数问题

- 拷贝构造时，意味着lhs是空的，再增加一个ref。
- 赋值构造时，lhs不是空的，需要释放原来的，并获得rhs的指针，再加一个ref。
- 移动构造时，lhs是空的，lhs指向rhs，rhs要销毁。
- 移动赋值时，lhs不是空的，要释放lhs，再将lhs指向rhs，rhs销毁。

### 1.2 access

#### `__shared_ptr_access`

``` cpp
template <typename _Tp, _Lock_policy _Lp, bool = is_array<_Tp>::value, bool = is_void<_Tp>::value>
class __shared_ptr_access {
public:
    using element_type = _Tp;

    element_type& operator*() const noexcept {
        __glibcxx_assert(_M_get() != nullptr);
        return *_M_get();
    }

    element_type* operator->() const noexcept {
        _GLIBCXX_DEBUG_PEDASSERT(_M_get() != nullptr);
        return _M_get();
    }

private:
    element_type* _M_get() const noexcept { return static_cast<const __shared_ptr<_Tp, _Lp>*>(this)->get(); }
};

// Define operator-> for shared_ptr<cv void>.
template <typename _Tp, _Lock_policy _Lp>
class __shared_ptr_access<_Tp, _Lp, false, true> {
public:
    using element_type = _Tp;

    element_type* operator->() const noexcept {
        auto __ptr = static_cast<const __shared_ptr<_Tp, _Lp>*>(this)->get();
        _GLIBCXX_DEBUG_PEDASSERT(__ptr != nullptr);
        return __ptr;
    }
};

// Define operator[] for shared_ptr<T[]> and shared_ptr<T[N]>.
template <typename _Tp, _Lock_policy _Lp>
class __shared_ptr_access<_Tp, _Lp, true, false> {
public:
    using element_type = typename remove_extent<_Tp>::type;

#if __cplusplus <= 201402L
    [[__deprecated__("shared_ptr<T[]>::operator* is absent from C++17")]] element_type& operator*() const noexcept {
        __glibcxx_assert(_M_get() != nullptr);
        return *_M_get();
    }

    [[__deprecated__("shared_ptr<T[]>::operator-> is absent from C++17")]] element_type* operator->()
        const noexcept {
        _GLIBCXX_DEBUG_PEDASSERT(_M_get() != nullptr);
        return _M_get();
    }
#endif

    element_type& operator[](ptrdiff_t __i) const {
        __glibcxx_assert(_M_get() != nullptr);
        __glibcxx_assert(!extent<_Tp>::value || __i < extent<_Tp>::value);
        return _M_get()[__i];
    }

private:
    element_type* _M_get() const noexcept { return static_cast<const __shared_ptr<_Tp, _Lp>*>(this)->get(); }
};
```

这个类`__shared_ptr_access`没有太多的内容，主要是提供了对后面`__shared_ptr`的`operator*`和`operator->`的实现。

### 1.3 `__shared_ptr`

#### 主体代码

``` cpp
template <typename _Tp, _Lock_policy _Lp>
class __shared_ptr : public __shared_ptr_access<_Tp, _Lp> {
public:
    using element_type = typename remove_extent<_Tp>::type;

private:
    // Constraint for taking ownership of a pointer of type _Yp*:
    template <typename _Yp>
    using _SafeConv = typename enable_if<__sp_is_constructible<_Tp, _Yp>::value>::type;

    // Constraint for construction from shared_ptr and weak_ptr:
    template <typename _Yp, typename _Res = void>
    using _Compatible = typename enable_if<__sp_compatible_with<_Yp*, _Tp*>::value, _Res>::type;

    // Constraint for assignment from shared_ptr and weak_ptr:
    template <typename _Yp>
    using _Assignable = _Compatible<_Yp, __shared_ptr&>;

    // Constraint for construction from unique_ptr:
    template <typename _Yp, typename _Del, typename _Res = void,
              typename _Ptr = typename unique_ptr<_Yp, _Del>::pointer>
    using _UniqCompatible =
        typename enable_if<__and_<__sp_compatible_with<_Yp*, _Tp*>, is_convertible<_Ptr, element_type*>>::value,
                           _Res>::type;

    // Constraint for assignment from unique_ptr:
    template <typename _Yp, typename _Del>
    using _UniqAssignable = _UniqCompatible<_Yp, _Del, __shared_ptr&>;

public:
#if __cplusplus > 201402L
    using weak_type = __weak_ptr<_Tp, _Lp>;
#endif

    constexpr __shared_ptr() noexcept : _M_ptr(0), _M_refcount() {}

    template <typename _Yp, typename = _SafeConv<_Yp>>
    explicit __shared_ptr(_Yp* __p) : _M_ptr(__p), _M_refcount(__p, typename is_array<_Tp>::type()) {
        static_assert(!is_void<_Yp>::value, "incomplete type");
        static_assert(sizeof(_Yp) > 0, "incomplete type");
        _M_enable_shared_from_this_with(__p);
    }

    template <typename _Yp, typename _Deleter, typename = _SafeConv<_Yp>>
    __shared_ptr(_Yp* __p, _Deleter __d) : _M_ptr(__p), _M_refcount(__p, std::move(__d)) {
        static_assert(__is_invocable<_Deleter&, _Yp*&>::value, "deleter expression d(p) is well-formed");
        _M_enable_shared_from_this_with(__p);
    }

    template <typename _Yp, typename _Deleter, typename _Alloc, typename = _SafeConv<_Yp>>
    __shared_ptr(_Yp* __p, _Deleter __d, _Alloc __a)
        : _M_ptr(__p), _M_refcount(__p, std::move(__d), std::move(__a)) {
        static_assert(__is_invocable<_Deleter&, _Yp*&>::value, "deleter expression d(p) is well-formed");
        _M_enable_shared_from_this_with(__p);
    }

    template <typename _Deleter>
    __shared_ptr(nullptr_t __p, _Deleter __d) : _M_ptr(0), _M_refcount(__p, std::move(__d)) {}

    template <typename _Deleter, typename _Alloc>
    __shared_ptr(nullptr_t __p, _Deleter __d, _Alloc __a)
        : _M_ptr(0), _M_refcount(__p, std::move(__d), std::move(__a)) {}

    // Aliasing constructor
    template <typename _Yp>
    __shared_ptr(const __shared_ptr<_Yp, _Lp>& __r, element_type* __p) noexcept
        : _M_ptr(__p),
          _M_refcount(__r._M_refcount)  // never throws
    {}

    // Aliasing constructor
    template <typename _Yp>
    __shared_ptr(__shared_ptr<_Yp, _Lp>&& __r, element_type* __p) noexcept : _M_ptr(__p), _M_refcount() {
        _M_refcount._M_swap(__r._M_refcount);
        __r._M_ptr = nullptr;
    }

    __shared_ptr(const __shared_ptr&) noexcept = default;
    __shared_ptr& operator=(const __shared_ptr&) noexcept = default;
    ~__shared_ptr() = default;

    template <typename _Yp, typename = _Compatible<_Yp>>
    __shared_ptr(const __shared_ptr<_Yp, _Lp>& __r) noexcept : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) {}

    __shared_ptr(__shared_ptr&& __r) noexcept : _M_ptr(__r._M_ptr), _M_refcount() {
        _M_refcount._M_swap(__r._M_refcount);
        __r._M_ptr = nullptr;
    }

    template <typename _Yp, typename = _Compatible<_Yp>>
    __shared_ptr(__shared_ptr<_Yp, _Lp>&& __r) noexcept : _M_ptr(__r._M_ptr), _M_refcount() {
        _M_refcount._M_swap(__r._M_refcount);
        __r._M_ptr = nullptr;
    }

    template <typename _Yp, typename = _Compatible<_Yp>>
    explicit __shared_ptr(const __weak_ptr<_Yp, _Lp>& __r)
        : _M_refcount(__r._M_refcount)  // may throw
    {
        // It is now safe to copy __r._M_ptr, as
        // _M_refcount(__r._M_refcount) did not throw.
        _M_ptr = __r._M_ptr;
    }

    // If an exception is thrown this constructor has no effect.
    template <typename _Yp, typename _Del, typename = _UniqCompatible<_Yp, _Del>>
    __shared_ptr(unique_ptr<_Yp, _Del>&& __r) : _M_ptr(__r.get()), _M_refcount() {
        auto __raw = __to_address(__r.get());
        _M_refcount = __shared_count<_Lp>(std::move(__r));
        _M_enable_shared_from_this_with(__raw);
    }

#if __cplusplus <= 201402L && _GLIBCXX_USE_DEPRECATED
protected:
    // If an exception is thrown this constructor has no effect.
    template <typename _Tp1, typename _Del,
              typename enable_if<__and_<__not_<is_array<_Tp>>, is_array<_Tp1>,
                                        is_convertible<typename unique_ptr<_Tp1, _Del>::pointer, _Tp*>>::value,
                                 bool>::type = true>
    __shared_ptr(unique_ptr<_Tp1, _Del>&& __r, __sp_array_delete) : _M_ptr(__r.get()), _M_refcount() {
        auto __raw = __to_address(__r.get());
        _M_refcount = __shared_count<_Lp>(std::move(__r));
        _M_enable_shared_from_this_with(__raw);
    }

public:
#endif

    constexpr __shared_ptr(nullptr_t) noexcept : __shared_ptr() {}

    template <typename _Yp>
    _Assignable<_Yp> operator=(const __shared_ptr<_Yp, _Lp>& __r) noexcept {
        _M_ptr = __r._M_ptr;
        _M_refcount = __r._M_refcount;  // __shared_count::op= doesn't throw
        return *this;
    }

    __shared_ptr& operator=(__shared_ptr&& __r) noexcept {
        __shared_ptr(std::move(__r)).swap(*this);
        return *this;
    }

    template <class _Yp>
    _Assignable<_Yp> operator=(__shared_ptr<_Yp, _Lp>&& __r) noexcept {
        __shared_ptr(std::move(__r)).swap(*this);
        return *this;
    }

    template <typename _Yp, typename _Del>
    _UniqAssignable<_Yp, _Del> operator=(unique_ptr<_Yp, _Del>&& __r) {
        __shared_ptr(std::move(__r)).swap(*this);
        return *this;
    }

    void reset() noexcept { __shared_ptr().swap(*this); }

    template <typename _Yp>
    _SafeConv<_Yp> reset(_Yp* __p)  // _Yp must be complete.
    {
        // Catch self-reset errors.
        __glibcxx_assert(__p == nullptr || __p != _M_ptr);
        __shared_ptr(__p).swap(*this);
    }

    template <typename _Yp, typename _Deleter>
    _SafeConv<_Yp> reset(_Yp* __p, _Deleter __d) {
        __shared_ptr(__p, std::move(__d)).swap(*this);
    }

    template <typename _Yp, typename _Deleter, typename _Alloc>
    _SafeConv<_Yp> reset(_Yp* __p, _Deleter __d, _Alloc __a) {
        __shared_ptr(__p, std::move(__d), std::move(__a)).swap(*this);
    }

    /// Return the stored pointer.
    element_type* get() const noexcept { return _M_ptr; }

    /// Return true if the stored pointer is not null.
    explicit operator bool() const noexcept { return _M_ptr != nullptr; }

    /// Return true if use_count() == 1.
    bool unique() const noexcept { return _M_refcount._M_unique(); }

    /// If *this owns a pointer, return the number of owners, otherwise zero.
    long use_count() const noexcept { return _M_refcount._M_get_use_count(); }

    /// Exchange both the owned pointer and the stored pointer.
    void swap(__shared_ptr<_Tp, _Lp>& __other) noexcept {
        std::swap(_M_ptr, __other._M_ptr);
        _M_refcount._M_swap(__other._M_refcount);
    }

    /** @brief Define an ordering based on ownership.
     *
     * This function defines a strict weak ordering between two shared_ptr
     * or weak_ptr objects, such that one object is less than the other
     * unless they share ownership of the same pointer, or are both empty.
     * @{
     */
    template <typename _Tp1>
    bool owner_before(__shared_ptr<_Tp1, _Lp> const& __rhs) const noexcept {
        return _M_refcount._M_less(__rhs._M_refcount);
    }

    template <typename _Tp1>
    bool owner_before(__weak_ptr<_Tp1, _Lp> const& __rhs) const noexcept {
        return _M_refcount._M_less(__rhs._M_refcount);
    }
    /// @}

protected:
    // This constructor is non-standard, it is used by allocate_shared.
    template <typename _Alloc, typename... _Args>
    __shared_ptr(_Sp_alloc_shared_tag<_Alloc> __tag, _Args&&... __args)
        : _M_ptr(), _M_refcount(_M_ptr, __tag, std::forward<_Args>(__args)...) {
        _M_enable_shared_from_this_with(_M_ptr);
    }

    template <typename _Tp1, _Lock_policy _Lp1, typename _Alloc, typename... _Args>
    friend __shared_ptr<_Tp1, _Lp1> __allocate_shared(const _Alloc& __a, _Args&&... __args);

    // This constructor is used by __weak_ptr::lock() and
    // shared_ptr::shared_ptr(const weak_ptr&, std::nothrow_t).
    __shared_ptr(const __weak_ptr<_Tp, _Lp>& __r, std::nothrow_t) noexcept
        : _M_refcount(__r._M_refcount, std::nothrow) {
        _M_ptr = _M_refcount._M_get_use_count() ? __r._M_ptr : nullptr;
    }

    friend class __weak_ptr<_Tp, _Lp>;

private:
    template <typename _Yp>
    using __esft_base_t =
        decltype(__enable_shared_from_this_base(std::declval<const __shared_count<_Lp>&>(), std::declval<_Yp*>()));

    // Detect an accessible and unambiguous enable_shared_from_this base.
    template <typename _Yp, typename = void>
    struct __has_esft_base : false_type {};

    template <typename _Yp>
    struct __has_esft_base<_Yp, __void_t<__esft_base_t<_Yp>>> : __not_<is_array<_Tp>> {
    };  // No enable shared_from_this for arrays

    template <typename _Yp, typename _Yp2 = typename remove_cv<_Yp>::type>
    typename enable_if<__has_esft_base<_Yp2>::value>::type _M_enable_shared_from_this_with(_Yp* __p) noexcept {
        if (auto __base = __enable_shared_from_this_base(_M_refcount, __p))
            __base->_M_weak_assign(const_cast<_Yp2*>(__p), _M_refcount);
    }

    template <typename _Yp, typename _Yp2 = typename remove_cv<_Yp>::type>
    typename enable_if<!__has_esft_base<_Yp2>::value>::type _M_enable_shared_from_this_with(_Yp*) noexcept {}

    void* _M_get_deleter(const std::type_info& __ti) const noexcept { return _M_refcount._M_get_deleter(__ti); }

    template <typename _Tp1, _Lock_policy _Lp1>
    friend class __shared_ptr;
    template <typename _Tp1, _Lock_policy _Lp1>
    friend class __weak_ptr;

    template <typename _Del, typename _Tp1, _Lock_policy _Lp1>
    friend _Del* get_deleter(const __shared_ptr<_Tp1, _Lp1>&) noexcept;

    template <typename _Del, typename _Tp1>
    friend _Del* get_deleter(const shared_ptr<_Tp1>&) noexcept;

    element_type* _M_ptr;             // Contained pointer.
    __shared_count<_Lp> _M_refcount;  // Reference counter.
};
```

两个成员变量：`_M_ptr`指向实际的数据，`_M_refcount`是它的引用计数。

- 构造函数：输入的数据主要是`_Yp* __p`，直接将赋值给`_M_ptr`，并用其构造一个`_M_refcount`。并对齐调用`_M_enable_shared_from_this_with`。
- 拷贝构造函数：用`__r._M_ptr`和`__r._M_refcount`直接初始化this的`_M_ptr`、`_M_refcount`。
- 移动构造函数：用`__r._M_ptr`直接初始化this的`_M_ptr`，`_M_refcount`对`__r._M_refcount`调用`swap`函数，并将`__r._M_ptr`置为nullptr。
- 输入为`__weak_ptr`的拷贝构造函数：用`__r._M_refcount`直接初始化this的`_M_refcount`，并将`_M_ptr`赋值给this的`_M_ptr`。
- 输入为`unique_ptr`的拷贝构造函数：用`__r.get()`直接初始化this的`_M_ptr`，并且调用`__shared_count`的对`unique_ptr`的特化形式进行初始化。并对齐调用`_M_enable_shared_from_this_with`。
- 拷贝赋值函数：用`__r._M_ptr`和`__r._M_refcount`直接赋值给this的`_M_ptr`、`_M_refcount`。
- 移动赋值函数：用`__r`移动构造一个`__shared_ptr`并和this进行`swap`。
- `reset`函数：用输入参数（或不带参数）构造一个`__shared_ptr`，并与this进行`swap`。
- `swap`函数：对`_M_ptr`调用`std::swap`，对`_M_refcount`调用类的`swap`函数。
- `owner_before`函数：对对`_M_refcount`调用类的`_M_less`函数。
- `_M_enable_shared_from_this_with`函数：

#### 相关函数

``` cpp
// 20.7.2.2.7 shared_ptr comparisons
template <typename _Tp1, typename _Tp2, _Lock_policy _Lp>
inline bool operator==(const __shared_ptr<_Tp1, _Lp>& __a, const __shared_ptr<_Tp2, _Lp>& __b) noexcept {
    return __a.get() == __b.get();
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator==(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    return !__a;
}

#ifdef __cpp_lib_three_way_comparison
template <typename _Tp, typename _Up, _Lock_policy _Lp>
inline strong_ordering operator<=>(const __shared_ptr<_Tp, _Lp>& __a, const __shared_ptr<_Up, _Lp>& __b) noexcept {
    return compare_three_way()(__a.get(), __b.get());
}

template <typename _Tp, _Lock_policy _Lp>
inline strong_ordering operator<=>(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    using pointer = typename __shared_ptr<_Tp, _Lp>::element_type*;
    return compare_three_way()(__a.get(), static_cast<pointer>(nullptr));
}
#else
template <typename _Tp, _Lock_policy _Lp>
inline bool operator==(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    return !__a;
}

template <typename _Tp1, typename _Tp2, _Lock_policy _Lp>
inline bool operator!=(const __shared_ptr<_Tp1, _Lp>& __a, const __shared_ptr<_Tp2, _Lp>& __b) noexcept {
    return __a.get() != __b.get();
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator!=(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    return (bool)__a;
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator!=(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    return (bool)__a;
}

template <typename _Tp, typename _Up, _Lock_policy _Lp>
inline bool operator<(const __shared_ptr<_Tp, _Lp>& __a, const __shared_ptr<_Up, _Lp>& __b) noexcept {
    using _Tp_elt = typename __shared_ptr<_Tp, _Lp>::element_type;
    using _Up_elt = typename __shared_ptr<_Up, _Lp>::element_type;
    using _Vp = typename common_type<_Tp_elt*, _Up_elt*>::type;
    return less<_Vp>()(__a.get(), __b.get());
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator<(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    using _Tp_elt = typename __shared_ptr<_Tp, _Lp>::element_type;
    return less<_Tp_elt*>()(__a.get(), nullptr);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator<(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    using _Tp_elt = typename __shared_ptr<_Tp, _Lp>::element_type;
    return less<_Tp_elt*>()(nullptr, __a.get());
}

template <typename _Tp1, typename _Tp2, _Lock_policy _Lp>
inline bool operator<=(const __shared_ptr<_Tp1, _Lp>& __a, const __shared_ptr<_Tp2, _Lp>& __b) noexcept {
    return !(__b < __a);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator<=(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    return !(nullptr < __a);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator<=(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    return !(__a < nullptr);
}

template <typename _Tp1, typename _Tp2, _Lock_policy _Lp>
inline bool operator>(const __shared_ptr<_Tp1, _Lp>& __a, const __shared_ptr<_Tp2, _Lp>& __b) noexcept {
    return (__b < __a);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator>(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    return nullptr < __a;
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator>(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    return __a < nullptr;
}

template <typename _Tp1, typename _Tp2, _Lock_policy _Lp>
inline bool operator>=(const __shared_ptr<_Tp1, _Lp>& __a, const __shared_ptr<_Tp2, _Lp>& __b) noexcept {
    return !(__a < __b);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator>=(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t) noexcept {
    return !(__a < nullptr);
}

template <typename _Tp, _Lock_policy _Lp>
inline bool operator>=(nullptr_t, const __shared_ptr<_Tp, _Lp>& __a) noexcept {
    return !(nullptr < __a);
}
#endif  // three-way comparison

// 20.7.2.2.8 shared_ptr specialized algorithms.
template <typename _Tp, _Lock_policy _Lp>
inline void swap(__shared_ptr<_Tp, _Lp> & __a, __shared_ptr<_Tp, _Lp> & __b) noexcept {
    __a.swap(__b);
}

// 20.7.2.2.9 shared_ptr casts

// The seemingly equivalent code:
// shared_ptr<_Tp, _Lp>(static_cast<_Tp*>(__r.get()))
// will eventually result in undefined behaviour, attempting to
// delete the same object twice.
/// static_pointer_cast
template <typename _Tp, typename _Tp1, _Lock_policy _Lp>
inline __shared_ptr<_Tp, _Lp> static_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r) noexcept {
    using _Sp = __shared_ptr<_Tp, _Lp>;
    return _Sp(__r, static_cast<typename _Sp::element_type*>(__r.get()));
}

// The seemingly equivalent code:
// shared_ptr<_Tp, _Lp>(const_cast<_Tp*>(__r.get()))
// will eventually result in undefined behaviour, attempting to
// delete the same object twice.
/// const_pointer_cast
template <typename _Tp, typename _Tp1, _Lock_policy _Lp>
inline __shared_ptr<_Tp, _Lp> const_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r) noexcept {
    using _Sp = __shared_ptr<_Tp, _Lp>;
    return _Sp(__r, const_cast<typename _Sp::element_type*>(__r.get()));
}

// The seemingly equivalent code:
// shared_ptr<_Tp, _Lp>(dynamic_cast<_Tp*>(__r.get()))
// will eventually result in undefined behaviour, attempting to
// delete the same object twice.
/// dynamic_pointer_cast
template <typename _Tp, typename _Tp1, _Lock_policy _Lp>
inline __shared_ptr<_Tp, _Lp> dynamic_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r) noexcept {
    using _Sp = __shared_ptr<_Tp, _Lp>;
    if (auto* __p = dynamic_cast<typename _Sp::element_type*>(__r.get())) return _Sp(__r, __p);
    return _Sp();
}

#if __cplusplus > 201402L
template <typename _Tp, typename _Tp1, _Lock_policy _Lp>
inline __shared_ptr<_Tp, _Lp> reinterpret_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r) noexcept {
    using _Sp = __shared_ptr<_Tp, _Lp>;
    return _Sp(__r, reinterpret_cast<typename _Sp::element_type*>(__r.get()));
}
#endif
```

- 比较类函数：实际上都是背后指针的比较。
- `swap`函数：调用`__shard_ptr::swap`函数。
- 各种`*_pointer_cast`函数：实际上对实际的指针（`__shared_ptr::get()`）调用了相应的`*_cast`函数，并返回相应类型的`__shared_ptr`。

### 1.4 `__weak_ptr`

#### 主体代码

``` cpp
template <typename _Tp, _Lock_policy _Lp>
class __weak_ptr {
    template <typename _Yp, typename _Res = void>
    using _Compatible = typename enable_if<__sp_compatible_with<_Yp*, _Tp*>::value, _Res>::type;

    // Constraint for assignment from shared_ptr and weak_ptr:
    template <typename _Yp>
    using _Assignable = _Compatible<_Yp, __weak_ptr&>;

public:
    using element_type = typename remove_extent<_Tp>::type;

    constexpr __weak_ptr() noexcept : _M_ptr(nullptr), _M_refcount() {}

    __weak_ptr(const __weak_ptr&) noexcept = default;

    ~__weak_ptr() = default;

    // The "obvious" converting constructor implementation:
    //
    //  template<typename _Tp1>
    //    __weak_ptr(const __weak_ptr<_Tp1, _Lp>& __r)
    //    : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) // never throws
    //    { }
    //
    // has a serious problem.
    //
    //  __r._M_ptr may already have been invalidated. The _M_ptr(__r._M_ptr)
    //  conversion may require access to *__r._M_ptr (virtual inheritance).
    //
    // It is not possible to avoid spurious access violations since
    // in multithreaded programs __r._M_ptr may be invalidated at any point.
    template <typename _Yp, typename = _Compatible<_Yp>>
    __weak_ptr(const __weak_ptr<_Yp, _Lp>& __r) noexcept : _M_refcount(__r._M_refcount) {
        _M_ptr = __r.lock().get();
    }

    template <typename _Yp, typename = _Compatible<_Yp>>
    __weak_ptr(const __shared_ptr<_Yp, _Lp>& __r) noexcept : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) {}

    __weak_ptr(__weak_ptr&& __r) noexcept : _M_ptr(__r._M_ptr), _M_refcount(std::move(__r._M_refcount)) {
        __r._M_ptr = nullptr;
    }

    template <typename _Yp, typename = _Compatible<_Yp>>
    __weak_ptr(__weak_ptr<_Yp, _Lp>&& __r) noexcept
        : _M_ptr(__r.lock().get()), _M_refcount(std::move(__r._M_refcount)) {
        __r._M_ptr = nullptr;
    }

    __weak_ptr& operator=(const __weak_ptr& __r) noexcept = default;

    template <typename _Yp>
    _Assignable<_Yp> operator=(const __weak_ptr<_Yp, _Lp>& __r) noexcept {
        _M_ptr = __r.lock().get();
        _M_refcount = __r._M_refcount;
        return *this;
    }

    template <typename _Yp>
    _Assignable<_Yp> operator=(const __shared_ptr<_Yp, _Lp>& __r) noexcept {
        _M_ptr = __r._M_ptr;
        _M_refcount = __r._M_refcount;
        return *this;
    }

    __weak_ptr& operator=(__weak_ptr&& __r) noexcept {
        _M_ptr = __r._M_ptr;
        _M_refcount = std::move(__r._M_refcount);
        __r._M_ptr = nullptr;
        return *this;
    }

    template <typename _Yp>
    _Assignable<_Yp> operator=(__weak_ptr<_Yp, _Lp>&& __r) noexcept {
        _M_ptr = __r.lock().get();
        _M_refcount = std::move(__r._M_refcount);
        __r._M_ptr = nullptr;
        return *this;
    }

    __shared_ptr<_Tp, _Lp> lock() const noexcept { return __shared_ptr<element_type, _Lp>(*this, std::nothrow); }

    long use_count() const noexcept { return _M_refcount._M_get_use_count(); }

    bool expired() const noexcept { return _M_refcount._M_get_use_count() == 0; }

    template <typename _Tp1>
    bool owner_before(const __shared_ptr<_Tp1, _Lp>& __rhs) const noexcept {
        return _M_refcount._M_less(__rhs._M_refcount);
    }

    template <typename _Tp1>
    bool owner_before(const __weak_ptr<_Tp1, _Lp>& __rhs) const noexcept {
        return _M_refcount._M_less(__rhs._M_refcount);
    }

    void reset() noexcept { __weak_ptr().swap(*this); }

    void swap(__weak_ptr& __s) noexcept {
        std::swap(_M_ptr, __s._M_ptr);
        _M_refcount._M_swap(__s._M_refcount);
    }

private:
    // Used by __enable_shared_from_this.
    void _M_assign(_Tp* __ptr, const __shared_count<_Lp>& __refcount) noexcept {
        if (use_count() == 0) {
            _M_ptr = __ptr;
            _M_refcount = __refcount;
        }
    }

    template <typename _Tp1, _Lock_policy _Lp1>
    friend class __shared_ptr;
    template <typename _Tp1, _Lock_policy _Lp1>
    friend class __weak_ptr;
    friend class __enable_shared_from_this<_Tp, _Lp>;
    friend class enable_shared_from_this<_Tp>;

    element_type* _M_ptr;           // Contained pointer.
    __weak_count<_Lp> _M_refcount;  // Reference counter.
};

// 20.7.2.3.6 weak_ptr specialized algorithms.
template <typename _Tp, _Lock_policy _Lp>
inline void swap(__weak_ptr<_Tp, _Lp> & __a, __weak_ptr<_Tp, _Lp> & __b) noexcept {
    __a.swap(__b);
}
```

- 数据成员`_M_ptr`实际的指针，`_M_refcount`引用计数。
- 同指针类型的拷贝构造函数：默认行为。直接用另一个的对象的`_M_ptr`、`_M_refcount`来初始化自己的数据。
- 非同指针类型的拷贝构造函数：用另一个对象的`_M_refcount`来初始化自己的`_M_refcount`，但`_M_ptr`的初始化需要获得另一个对象的`_M_ptr`的时候加锁。
- 输入类型为`_shared_ptr`的拷贝构造函数，直接用这个`_shared_ptr`的`_M_ptr`、`_M_refcount`来初始化自己的数据。
- 同指针类型的移动构造函数：直接用另一个的对象的`_M_ptr`、`_M_refcount`来初始化自己的数据。最后，需要将另一个对象的`_M_ptr`置为nullptr。
- 非同指针类型的移动构造函数：`_M_ptr`的初始化需要获得另一个对象的`_M_ptr`的时候加锁，`_M_refcount`移动构造，将另一个对象的`_M_ptr`置为nullptr。
- 拷贝赋值、移动赋值函数与对应的拷贝构造、移动构造函数逻辑基本一致。
- `lock`函数：用此类的元素构造一个`__shared_ptr`。
- `owner_before`函数：`_M_refcount`对`__r._M_refcount`调用`_M_less`函数。

#### 为什么获取不同指针类型`weak_ptr`的实际指针时需要加锁，以及为何加锁函数是这么设计的？

- 因为`weak_ptr`只是一个*弱*引用，并不真正的**持有**这个指针，所以在不同类型指针复制的时候需要加锁，防止因多线程问题，这个指针被销毁。
- 这里的加锁，并不是常见的所谓如`mutex`等的加锁，而是将其转为一个`shared_ptr`，因为它是**持有**这个指针的，正常情况下复制的过程会得到符合预期的结果。

### 1.5 其他

#### `_enable_shared_from_this`

``` cpp
template <typename _Tp, _Lock_policy _Lp>
class __enable_shared_from_this {
protected:
    constexpr __enable_shared_from_this() noexcept {}

    __enable_shared_from_this(const __enable_shared_from_this&) noexcept {}

    __enable_shared_from_this& operator=(const __enable_shared_from_this&) noexcept { return *this; }

    ~__enable_shared_from_this() {}

public:
    __shared_ptr<_Tp, _Lp> shared_from_this() { return __shared_ptr<_Tp, _Lp>(this->_M_weak_this); }

    __shared_ptr<const _Tp, _Lp> shared_from_this() const {
        return __shared_ptr<const _Tp, _Lp>(this->_M_weak_this);
    }

#if __cplusplus > 201402L || !defined(__STRICT_ANSI__)  // c++1z or gnu++11
    __weak_ptr<_Tp, _Lp> weak_from_this() noexcept { return this->_M_weak_this; }

    __weak_ptr<const _Tp, _Lp> weak_from_this() const noexcept { return this->_M_weak_this; }
#endif

private:
    template <typename _Tp1>
    void _M_weak_assign(_Tp1* __p, const __shared_count<_Lp>& __n) const noexcept {
        _M_weak_this._M_assign(__p, __n);
    }

    friend const __enable_shared_from_this* __enable_shared_from_this_base(const __shared_count<_Lp>&,
                                                                           const __enable_shared_from_this* __p) {
        return __p;
    }

    template <typename, _Lock_policy>
    friend class __shared_ptr;

    mutable __weak_ptr<_Tp, _Lp> _M_weak_this;
};
```

- 这个类的使用方法一般是作为某个类的基类来使用的，使得可以从这个类里直接获得一个智能指针。
- 主要的成员变量是一个`__weak_ptr`。
- 当调用`shared_from_this`就将这个成员变构造一个`__shared_ptr`，`weak_from_this`。

#### `__make_shared`

``` cpp
template <typename _Tp, _Lock_policy _Lp = __default_lock_policy, typename _Alloc, typename... _Args>
inline __shared_ptr<_Tp, _Lp> __allocate_shared(const _Alloc& __a, _Args&&... __args) {
    return __shared_ptr<_Tp, _Lp>(_Sp_alloc_shared_tag<_Alloc>{__a}, std::forward<_Args>(__args)...);
}

template <typename _Tp, _Lock_policy _Lp = __default_lock_policy, typename... _Args>
inline __shared_ptr<_Tp, _Lp> __make_shared(_Args && ... __args) {
    typedef typename std::remove_const<_Tp>::type _Tp_nc;
    return std::__allocate_shared<_Tp, _Lp>(std::allocator<_Tp_nc>(), std::forward<_Args>(__args)...);
}
```

利用传入的参数构建一个`__shared_ptr`。

#### `std::hash`对`__shared_ptr`的特化形式

``` cpp
/// std::hash specialization for __shared_ptr.
template <typename _Tp, _Lock_policy _Lp>
struct hash<__shared_ptr<_Tp, _Lp>> : public __hash_base<size_t, __shared_ptr<_Tp, _Lp>> {
    size_t operator()(const __shared_ptr<_Tp, _Lp>& __s) const noexcept {
        return hash<typename __shared_ptr<_Tp, _Lp>::element_type*>()(__s.get());
    }
};
```

主要是退化为`std::hash`对ptr的特化形式。

#### `__owner_less`

``` cpp
template <typename _Tp, typename _Tp1>
struct _Sp_owner_less : public binary_function<_Tp, _Tp, bool> {
    bool operator()(const _Tp& __lhs, const _Tp& __rhs) const noexcept { return __lhs.owner_before(__rhs); }

    bool operator()(const _Tp& __lhs, const _Tp1& __rhs) const noexcept { return __lhs.owner_before(__rhs); }

    bool operator()(const _Tp1& __lhs, const _Tp& __rhs) const noexcept { return __lhs.owner_before(__rhs); }
};

template <>
struct _Sp_owner_less<void, void> {
    template <typename _Tp, typename _Up>
    auto operator()(const _Tp& __lhs, const _Up& __rhs) const noexcept -> decltype(__lhs.owner_before(__rhs)) {
        return __lhs.owner_before(__rhs);
    }

    using is_transparent = void;
};

template <typename _Tp, _Lock_policy _Lp>
struct owner_less<__shared_ptr<_Tp, _Lp>> : public _Sp_owner_less<__shared_ptr<_Tp, _Lp>, __weak_ptr<_Tp, _Lp>> {};

template <typename _Tp, _Lock_policy _Lp>
struct owner_less<__weak_ptr<_Tp, _Lp>> : public _Sp_owner_less<__weak_ptr<_Tp, _Lp>, __shared_ptr<_Tp, _Lp>> {};
```

主要调用`owner_before`函数，没有过于特别的逻辑。

### 1.5 `<shared_ptr.h>`文件的说明

这个文件里面是真正对外的接口的说明，实际上`shared_ptr`、`weak_ptr`、`enable_shared_from_this`等等都是对`<shared_ptr_base.h>`中的相应的类的调用，没有特别的逻辑。

## 2 备注

### 2.1 关于`_GLIBCXX_SYNCHRONIZATION_HAPPENS_*`

摘自`<bits/c++config>`

``` cpp
// Macros for race detectors.
// _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(A) and
// _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(A) should be used to explain
// atomic (lock-free) synchronization to race detectors:
// the race detector will infer a happens-before arc from the former to the
// latter when they share the same argument pointer.
//
// The most frequent use case for these macros (and the only case in the
// current implementation of the library) is atomic reference counting:
//   void _M_remove_reference()
//   {
//     _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&this->_M_refcount);
//     if (__gnu_cxx::__exchange_and_add_dispatch(&this->_M_refcount, -1) <= 0)
//       {
//         _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&this->_M_refcount);
//         _M_destroy(__a);
//       }
//   }
// The annotations in this example tell the race detector that all memory
// accesses occurred when the refcount was positive do not race with
// memory accesses which occurred after the refcount became zero.
#ifndef _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE
# define  _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(A)
#endif
#ifndef _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER
# define  _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(A)
#endif
```

翻译整理如下：

`_GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(A)`和`_GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(A)`是用来竞争检测的宏，用在atomic（无锁）的同步。

上述代码中的例子，这两个宏的使用会告诉race detector，当refcount是正数的时候，所有内存读取不会和refcount变成0之后的内存读取相竞争。

### 2.2 `__exchange_and_add_dispatch`

参见文件`<ext/atomicity.h>`和[网页](https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_concurrency.html)，它的函数原型：

``` cpp
_Atomic_word
__exchange_and_add_dispatch(volatile _Atomic_word*, int);
```

功能：第一个参数与第二个参数相加，并返回第一个参数的旧值。

### 2.3 `_Lock_policy`

参见`<ext/concurrence.h>`

``` cpp
// Available locking policies:
// _S_single    single-threaded code that doesn't need to be locked.
// _S_mutex     multi-threaded code that requires additional support
//              from gthr.h or abstraction layers in concurrence.h.
// _S_atomic    multi-threaded code using atomic operations.
enum _Lock_policy { _S_single, _S_mutex, _S_atomic }; 

// Compile time constant that indicates prefered locking policy in
// the current configuration.
static const _Lock_policy __default_lock_policy = 
#ifndef __GTHREADS
_S_single;
#elif defined _GLIBCXX_HAVE_ATOMIC_LOCK_POLICY
_S_atomic;
#else
_S_mutex;
#endif
```

`_Lock_policy`是个枚举类型，有三个变量，解释如下：

- `_S_single`：单线程，不需要加锁
- `_S_mutex`：多线程，需要加锁的
- `_S_atomic`：多线程，使用原子操作

### 2.4 对指针类型的`std::less`

以下内容摘抄自[网页](https://en.cppreference.com/w/cpp/utility/functional/less)：

> A specialization of `std::less` for any pointer type yields the implementation-defined strict total order

即对指针来讲，`std::less`会有一个视实现而定的严格完全顺序。也就是说，用`std::less`比较指针是可行的。

## 3 问题

- `__shared_count`的`/* is_array = */ false_type`是啥意思？什么用法？
