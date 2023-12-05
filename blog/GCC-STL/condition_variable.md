# condition variable

## 1 主要代码

### 1.1 `__condvar`

``` cpp
// Implementation details for std::condition_variable
class __condvar {
    using timespec = __gthread_time_t;

public:
    __condvar() noexcept {
#ifndef __GTHREAD_COND_INIT
        __GTHREAD_COND_INIT_FUNCTION(&_M_cond);
#endif
    }

    ~__condvar() {
        int __e __attribute__((__unused__)) = __gthread_cond_destroy(&_M_cond);
        __glibcxx_assert(__e != EBUSY);  // threads are still blocked
    }

    __condvar(const __condvar&) = delete;
    __condvar& operator=(const __condvar&) = delete;

    __gthread_cond_t* native_handle() noexcept { return &_M_cond; }

    // Expects: Calling thread has locked __m.
    void wait(mutex& __m) noexcept {
        int __e __attribute__((__unused__)) = __gthread_cond_wait(&_M_cond, __m.native_handle());
        __glibcxx_assert(__e == 0);
    }

    void wait_until(mutex& __m, timespec& __abs_time) noexcept {
        __gthread_cond_timedwait(&_M_cond, __m.native_handle(), &__abs_time);
    }

#ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    void wait_until(mutex& __m, clockid_t __clock, timespec& __abs_time) noexcept {
        pthread_cond_clockwait(&_M_cond, __m.native_handle(), __clock, &__abs_time);
    }
#endif

    void notify_one() noexcept {
        int __e __attribute__((__unused__)) = __gthread_cond_signal(&_M_cond);
        __glibcxx_assert(__e == 0);
    }

    void notify_all() noexcept {
        int __e __attribute__((__unused__)) = __gthread_cond_broadcast(&_M_cond);
        __glibcxx_assert(__e == 0);
    }

protected:
#ifdef __GTHREAD_COND_INIT
    __gthread_cond_t _M_cond = __GTHREAD_COND_INIT;
#else
    __gthread_cond_t _M_cond;
#endif
};
```

这个类基础的成员变量是`__gthread_cond_t _M_cond;`。

## 2 备注

## 3 问题

