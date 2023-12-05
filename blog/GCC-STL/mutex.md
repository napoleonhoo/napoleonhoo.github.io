# mutex

## 1 主要代码

### 1.1 各种lock

#### lock tag

``` cpp
/// Do not acquire ownership of the mutex.
struct defer_lock_t {
    explicit defer_lock_t() = default;
};

/// Try to acquire ownership of the mutex without blocking.
struct try_to_lock_t {
    explicit try_to_lock_t() = default;
};

/// Assume the calling thread has already obtained mutex ownership
/// and manage it.
struct adopt_lock_t {
    explicit adopt_lock_t() = default;
};

/// Tag used to prevent a scoped lock from acquiring ownership of a mutex.
_GLIBCXX17_INLINE constexpr defer_lock_t defer_lock{};

/// Tag used to prevent a scoped lock from blocking if a mutex is locked.
_GLIBCXX17_INLINE constexpr try_to_lock_t try_to_lock{};

/// Tag used to make a scoped lock take ownership of a locked mutex.
_GLIBCXX17_INLINE constexpr adopt_lock_t adopt_lock{};
```

注意这三种struct和tag的用法：

- `defer_lock_t`&`defer_lock`：不要取得mutex的所有权（ownership）。
- `try_to__lock_t`&`try_to_lock`：尝试不阻塞地获取mutex的所有权。
- `adopt_lock_t`&`adopt_lock`：假设已经获得了mutex的所有权。

#### `lock_guard`

``` cpp
/** @brief A simple scoped lock type.
 *
 * A lock_guard controls mutex ownership within a scope, releasing
 * ownership in the destructor.
 */
template <typename _Mutex>
class lock_guard {
public:
    typedef _Mutex mutex_type;

    explicit lock_guard(mutex_type& __m) : _M_device(__m) { _M_device.lock(); }

    lock_guard(mutex_type& __m, adopt_lock_t) noexcept : _M_device(__m) {}  // calling thread owns mutex

    ~lock_guard() { _M_device.unlock(); }

    lock_guard(const lock_guard&) = delete;
    lock_guard& operator=(const lock_guard&) = delete;

private:
    mutex_type& _M_device;
};
```

- `lock_guard`是一种简单的scoped lock，采用RAII的实现，即在构造的时候加锁（`lock`），析构的释放锁（`unlock`）。 
- 不允许拷贝、移动。
- 对于带有`adopt_lock_t`参数的构造函数，不加锁，因为这个tag意味着mutex已经加过锁了。

#### `unique_lock`

``` cpp
/** @brief A movable scoped lock type.
 *
 * A unique_lock controls mutex ownership within a scope. Ownership of the
 * mutex can be delayed until after construction and can be transferred
 * to another unique_lock by move construction or move assignment. If a
 * mutex lock is owned when the destructor runs ownership will be released.
 *
 * @ingroup mutexes
 */
template <typename _Mutex>
class unique_lock {
public:
    typedef _Mutex mutex_type;

    unique_lock() noexcept : _M_device(0), _M_owns(false) {}

    explicit unique_lock(mutex_type& __m) : _M_device(std::__addressof(__m)), _M_owns(false) {
        lock();
        _M_owns = true;
    }

    unique_lock(mutex_type& __m, defer_lock_t) noexcept : _M_device(std::__addressof(__m)), _M_owns(false) {}

    unique_lock(mutex_type& __m, try_to_lock_t)
        : _M_device(std::__addressof(__m)), _M_owns(_M_device->try_lock()) {}

    unique_lock(mutex_type& __m, adopt_lock_t) noexcept : _M_device(std::__addressof(__m)), _M_owns(true) {
        // XXX calling thread owns mutex
    }

    template <typename _Clock, typename _Duration>
    unique_lock(mutex_type& __m, const chrono::time_point<_Clock, _Duration>& __atime)
        : _M_device(std::__addressof(__m)), _M_owns(_M_device->try_lock_until(__atime)) {}

    template <typename _Rep, typename _Period>
    unique_lock(mutex_type& __m, const chrono::duration<_Rep, _Period>& __rtime)
        : _M_device(std::__addressof(__m)), _M_owns(_M_device->try_lock_for(__rtime)) {}

    ~unique_lock() {
        if (_M_owns) unlock();
    }

    unique_lock(const unique_lock&) = delete;
    unique_lock& operator=(const unique_lock&) = delete;

    unique_lock(unique_lock&& __u) noexcept : _M_device(__u._M_device), _M_owns(__u._M_owns) {
        __u._M_device = 0;
        __u._M_owns = false;
    }

    unique_lock& operator=(unique_lock&& __u) noexcept {
        if (_M_owns) unlock();

        unique_lock(std::move(__u)).swap(*this);

        __u._M_device = 0;
        __u._M_owns = false;

        return *this;
    }

    void lock() {
        if (!_M_device)
            __throw_system_error(int(errc::operation_not_permitted));
        else if (_M_owns)
            __throw_system_error(int(errc::resource_deadlock_would_occur));
        else {
            _M_device->lock();
            _M_owns = true;
        }
    }

    bool try_lock() {
        if (!_M_device)
            __throw_system_error(int(errc::operation_not_permitted));
        else if (_M_owns)
            __throw_system_error(int(errc::resource_deadlock_would_occur));
        else {
            _M_owns = _M_device->try_lock();
            return _M_owns;
        }
    }

    template <typename _Clock, typename _Duration>
    bool try_lock_until(const chrono::time_point<_Clock, _Duration>& __atime) {
        if (!_M_device)
            __throw_system_error(int(errc::operation_not_permitted));
        else if (_M_owns)
            __throw_system_error(int(errc::resource_deadlock_would_occur));
        else {
            _M_owns = _M_device->try_lock_until(__atime);
            return _M_owns;
        }
    }

    template <typename _Rep, typename _Period>
    bool try_lock_for(const chrono::duration<_Rep, _Period>& __rtime) {
        if (!_M_device)
            __throw_system_error(int(errc::operation_not_permitted));
        else if (_M_owns)
            __throw_system_error(int(errc::resource_deadlock_would_occur));
        else {
            _M_owns = _M_device->try_lock_for(__rtime);
            return _M_owns;
        }
    }

    void unlock() {
        if (!_M_owns)
            __throw_system_error(int(errc::operation_not_permitted));
        else if (_M_device) {
            _M_device->unlock();
            _M_owns = false;
        }
    }

    void swap(unique_lock& __u) noexcept {
        std::swap(_M_device, __u._M_device);
        std::swap(_M_owns, __u._M_owns);
    }

    mutex_type* release() noexcept {
        mutex_type* __ret = _M_device;
        _M_device = 0;
        _M_owns = false;
        return __ret;
    }

    bool owns_lock() const noexcept { return _M_owns; }

    explicit operator bool() const noexcept { return owns_lock(); }

    mutex_type* mutex() const noexcept { return _M_device; }

private:
    mutex_type* _M_device;
    bool _M_owns;
};

/// Swap overload for unique_lock objects.
/// @relates unique_lock
template <typename _Mutex>
inline void swap(unique_lock<_Mutex> & __x, unique_lock<_Mutex> & __y) noexcept {
    __x.swap(__y);
}
```

比`lock_guard`更复杂一些，设计方式还是RAII，可以move，增加了更多操作。

- 基本的仍然是构造的时候加锁，析构的时候解锁。
- 带有前述三种tag的，不同的tag分别有相应的操作。
- 不允许拷贝构造、赋值。
- 移动构造、赋值函数，将输入参数的`_M_device`设为0，`_M_owns`设为false。
- `lock`相关函数，实现还是调用mutex相关的操作。
- `release`函数返回`_M_device`，并将this的`_M_device`设为0，`_M_owns`设为false。
- `bool`函数返回是否对`_M_device`有所有权。
- `mutex`函数返回`_M_device`。

#### `scoped_lock`

``` cpp
/** @brief A scoped lock type for multiple lockable objects.
 *
 * A scoped_lock controls mutex ownership within a scope, releasing
 * ownership in the destructor.
 */
template <typename... _MutexTypes>
class scoped_lock {
public:
    explicit scoped_lock(_MutexTypes&... __m) : _M_devices(std::tie(__m...)) { std::lock(__m...); }

    explicit scoped_lock(adopt_lock_t, _MutexTypes&... __m) noexcept
        : _M_devices(std::tie(__m...)) {}  // calling thread owns mutex

    ~scoped_lock() {
        std::apply([](auto&... __m) { (__m.unlock(), ...); }, _M_devices);
    }

    scoped_lock(const scoped_lock&) = delete;
    scoped_lock& operator=(const scoped_lock&) = delete;

private:
    tuple<_MutexTypes&...> _M_devices;
};

// 不吃又任何mutex的scoped_lock特化形式
template <>
class scoped_lock<> {
public:
    explicit scoped_lock() = default;
    explicit scoped_lock(adopt_lock_t) noexcept {}
    ~scoped_lock() = default;

    scoped_lock(const scoped_lock&) = delete;
    scoped_lock& operator=(const scoped_lock&) = delete;
};

// 持有单个mutex的scoped_lock特化形式
template <typename _Mutex>
class scoped_lock<_Mutex> {
public:
    using mutex_type = _Mutex;

    explicit scoped_lock(mutex_type& __m) : _M_device(__m) { _M_device.lock(); }

    explicit scoped_lock(adopt_lock_t, mutex_type& __m) noexcept : _M_device(__m) {}  // calling thread owns mutex

    ~scoped_lock() { _M_device.unlock(); }

    scoped_lock(const scoped_lock&) = delete;
    scoped_lock& operator=(const scoped_lock&) = delete;

private:
    mutex_type& _M_device;
};
```

`scoped_lock`类也是一种RAII方式，和`lock_guard`的实现基本类似，区别只在于`scoped_lock`可以持有多个mutex。所以看到可以持有多个mutex的`scoped_lock`最普通的形式，构造函数对所有的mutex加锁，析构函数对所有的mutex解锁。

#### 各种lock函数

``` cpp
/// @cond undocumented
template <typename _Lock>
inline unique_lock<_Lock> __try_to_lock(_Lock & __l) {
    return unique_lock<_Lock>{__l, try_to_lock};
}

template <int _Idx, bool _Continue = true>
struct __try_lock_impl {
    template <typename... _Lock>
    static void __do_try_lock(tuple<_Lock&...>& __locks, int& __idx) {
        __idx = _Idx;
        auto __lock = std::__try_to_lock(std::get<_Idx>(__locks));
        if (__lock.owns_lock()) {
            constexpr bool __cont = _Idx + 2 < sizeof...(_Lock);
            using __try_locker = __try_lock_impl<_Idx + 1, __cont>;
            __try_locker::__do_try_lock(__locks, __idx);
            if (__idx == -1) __lock.release();
        }
    }
};

template <int _Idx>
struct __try_lock_impl<_Idx, false> {
    template <typename... _Lock>
    static void __do_try_lock(tuple<_Lock&...>& __locks, int& __idx) {
        __idx = _Idx;
        auto __lock = std::__try_to_lock(std::get<_Idx>(__locks));
        if (__lock.owns_lock()) {
            __idx = -1;
            __lock.release();
        }
    }
};
/// @endcond

/** @brief Generic try_lock.
 *  @param __l1 Meets Lockable requirements (try_lock() may throw).
 *  @param __l2 Meets Lockable requirements (try_lock() may throw).
 *  @param __l3 Meets Lockable requirements (try_lock() may throw).
 *  @return Returns -1 if all try_lock() calls return true. Otherwise returns
 *          a 0-based index corresponding to the argument that returned false.
 *  @post Either all arguments are locked, or none will be.
 *
 *  Sequentially calls try_lock() on each argument.
 */
template <typename _Lock1, typename _Lock2, typename... _Lock3>
int try_lock(_Lock1 & __l1, _Lock2 & __l2, _Lock3 & ... __l3) {
    int __idx;
    auto __locks = std::tie(__l1, __l2, __l3...);
    __try_lock_impl<0>::__do_try_lock(__locks, __idx);
    return __idx;
}

/** @brief Generic lock.
 *  @param __l1 Meets Lockable requirements (try_lock() may throw).
 *  @param __l2 Meets Lockable requirements (try_lock() may throw).
 *  @param __l3 Meets Lockable requirements (try_lock() may throw).
 *  @throw An exception thrown by an argument's lock() or try_lock() member.
 *  @post All arguments are locked.
 *
 *  All arguments are locked via a sequence of calls to lock(), try_lock()
 *  and unlock().  If the call exits via an exception any locks that were
 *  obtained will be released.
 */
template <typename _L1, typename _L2, typename... _L3>
void lock(_L1 & __l1, _L2 & __l2, _L3 & ... __l3) {
    while (true) {
        using __try_locker = __try_lock_impl<0, sizeof...(_L3) != 0>;
        unique_lock<_L1> __first(__l1);
        int __idx;
        auto __locks = std::tie(__l2, __l3...);
        __try_locker::__do_try_lock(__locks, __idx);
        if (__idx == -1) {
            __first.release();
            return;
        }
    }
}
```

### 1.2 各种mutex

#### `__mutex_base`

``` cpp
// #ifdef _GLIBCXX_HAS_GTHREADS
// Common base class for std::mutex and std::timed_mutex
class __mutex_base {
protected:
    typedef __gthread_mutex_t __native_type;

#ifdef __GTHREAD_MUTEX_INIT
    __native_type _M_mutex = __GTHREAD_MUTEX_INIT;

    constexpr __mutex_base() noexcept = default;
#else
    __native_type _M_mutex;

    __mutex_base() noexcept {
        // XXX EAGAIN, ENOMEM, EPERM, EBUSY(may), EINVAL(may)
        __GTHREAD_MUTEX_INIT_FUNCTION(&_M_mutex);
    }

    ~__mutex_base() noexcept { __gthread_mutex_destroy(&_M_mutex); }
#endif

    __mutex_base(const __mutex_base&) = delete;
    __mutex_base& operator=(const __mutex_base&) = delete;
};
```

这个类是`std::mutex`和`std::timed_mutex`的基类。主要的实现了构造和析构函数。调用的是`__GTHREAD_MUTEX_INIT_FUNCTION`和`__gthread_mutex_destroy`。核心成员变量为`__gthread_mutex_t`类型的`_M_mutex`。

#### `mutex`

``` cpp
/// The standard mutex type.
class mutex : private __mutex_base {
public:
    typedef __native_type* native_handle_type;

#ifdef __GTHREAD_MUTEX_INIT
    constexpr
#endif
        mutex() noexcept = default;
    ~mutex() = default;

    mutex(const mutex&) = delete;
    mutex& operator=(const mutex&) = delete;

    void lock() {
        int __e = __gthread_mutex_lock(&_M_mutex);

        // EINVAL, EAGAIN, EBUSY, EINVAL, EDEADLK(may)
        if (__e) __throw_system_error(__e);
    }

    bool try_lock() noexcept {
        // XXX EINVAL, EAGAIN, EBUSY
        return !__gthread_mutex_trylock(&_M_mutex);
    }

    void unlock() {
        // XXX EINVAL, EAGAIN, EPERM
        __gthread_mutex_unlock(&_M_mutex);
    }

    native_handle_type native_handle() noexcept { return &_M_mutex; }
};
```

`mutex`继承自`__mutex_base`，`lock`相关函数即为调用gthread的相关操作。注意有一个`native_handle`函数，返回的是它的成员变量，即实际的`_M_mutex`。

#### `recursive_mutex`

``` cpp
// Common base class for std::recursive_mutex and std::recursive_timed_mutex
class __recursive_mutex_base {
protected:
    typedef __gthread_recursive_mutex_t __native_type;

    __recursive_mutex_base(const __recursive_mutex_base&) = delete;
    __recursive_mutex_base& operator=(const __recursive_mutex_base&) = delete;

#ifdef __GTHREAD_RECURSIVE_MUTEX_INIT
    __native_type _M_mutex = __GTHREAD_RECURSIVE_MUTEX_INIT;

    __recursive_mutex_base() = default;
#else
    __native_type _M_mutex;

    __recursive_mutex_base() {
        // XXX EAGAIN, ENOMEM, EPERM, EBUSY(may), EINVAL(may)
        __GTHREAD_RECURSIVE_MUTEX_INIT_FUNCTION(&_M_mutex);
    }

    ~__recursive_mutex_base() { __gthread_recursive_mutex_destroy(&_M_mutex); }
#endif
};

/// The standard recursive mutex type.
class recursive_mutex : private __recursive_mutex_base {
public:
    typedef __native_type* native_handle_type;

    recursive_mutex() = default;
    ~recursive_mutex() = default;

    recursive_mutex(const recursive_mutex&) = delete;
    recursive_mutex& operator=(const recursive_mutex&) = delete;

    void lock() {
        int __e = __gthread_recursive_mutex_lock(&_M_mutex);

        // EINVAL, EAGAIN, EBUSY, EINVAL, EDEADLK(may)
        if (__e) __throw_system_error(__e);
    }

    bool try_lock() noexcept {
        // XXX EINVAL, EAGAIN, EBUSY
        return !__gthread_recursive_mutex_trylock(&_M_mutex);
    }

    void unlock() {
        // XXX EINVAL, EAGAIN, EBUSY
        __gthread_recursive_mutex_unlock(&_M_mutex);
    }

    native_handle_type native_handle() noexcept { return &_M_mutex; }
};
```

#### `timed_mutex`



## 2 备注

- 最近的libstdc++已经用gthread统称thread了，对于Unix/Linux系统，这个就是对pthread的封装。
- gthread在Unix/Linux系统的实现，即支持pthread的系统，可以参看libstdc++中的文件`<gthr-posix.h>`。

## 3 问题

- 可以再进一步关注下pthread内部的实现。
