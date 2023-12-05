# `ConcurrentBoundedQueue`

## 1 简介

基于RingBuffer实现的MPMC队列，核心特性：
- 当前队列未满时，push操作是wait-free的
- 当前队列非空时，pop操作是wiat-free的
- 当push或pop从阻塞中被唤醒后，后续操作是wait-free的

## 2 主要成员类

### `SlotFutex`

``` cpp
    // 采用{uint16_t waiters, uint16_t version}打包存储到一个futex中
    class SlotFutex {
    public:
        // 重置到初始状态
        inline void reset() noexcept;
        // 读取版本号部分
        inline uint16_t version() const noexcept;
        // 等待版本就绪
        template <bool USE_FUTEX_WAIT>
        inline void wait_until_reach_expected_version(
                uint16_t expected_version) noexcept;
        void wait_until_reach_expected_version_slow(
                uint16_t expected_version,
                const struct ::timespec* timeout) noexcept;
        // 推进版本号
        inline void advance_version(uint16_t version) noexcept;
        // 唤醒等待者，和推进版本号之间需要建立seq_cst关系
        inline void wakeup_waiters(uint16_t current_version) noexcept;
        void wakeup_waiters_slow(uint32_t current_version_and_waiters) noexcept;
        // 推进版本号 && 唤醒等待者
        inline void advance_version_and_wakeup_waiters(
                uint16_t next_version) noexcept;

    private:
        Futex<S> _futex {0};
    };

    struct alignas(BABYLON_CACHELINE_SIZE) Slot {
        T value;
        SlotFutex futex;
    };
```

### `SlotVector`

``` cpp
class SlotVector {
    public:
        SlotVector() noexcept = default;
        SlotVector(SlotVector&& other) noexcept;
        SlotVector(const SlotVector&) = delete;
        SlotVector& operator=(SlotVector&&) noexcept;
        SlotVector& operator=(const SlotVector&) = delete;
        ~SlotVector() noexcept;

        SlotVector(size_t size) noexcept;

        void resize(size_t size) noexcept;

        inline size_t size() const noexcept;

        inline T& value(size_t index) noexcept;
        inline Iterator value_iterator(size_t index) noexcept;
        inline SlotFutex& futex(size_t index) noexcept;

        void swap(SlotVector& other) noexcept;

    private:
        Slot* _slots {nullptr};
        size_t _size {0};
    };
```

## 3 主要成员变量

## 4 主要成员函数