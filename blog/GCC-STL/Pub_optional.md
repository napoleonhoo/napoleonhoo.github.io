# optional

## 1 `_Optional_payload_base`结构体

这个模板管理`std::optional`的实际数据。

### 定义

``` cpp
template <typename _Tp> struct _Optional_payload_base {
  using _Stored_type = remove_const_t<_Tp>;
};
```

有一个存储类型的定义，就是`_Tp`去掉const修饰符后的类型。

### `_Storage`共用体

``` cpp
struct _Empty_byte {};

template <typename _Up, bool = is_trivially_destructible_v<_Up>>
union _Storage {
  constexpr _Storage() noexcept : _M_empty() {}

  template <typename... _Args>
  constexpr _Storage(in_place_t, _Args &&...__args)
      : _M_value(std::forward<_Args>(__args)...) {}

  template <typename _Vp, typename... _Args>
  constexpr _Storage(std::initializer_list<_Vp> __il, _Args &&...__args)
      : _M_value(__il, std::forward<_Args>(__args)...) {}

  _Empty_byte _M_empty;
  _Up _M_value;
};

template <typename _Up> union _Storage<_Up, false> {
  constexpr _Storage() noexcept : _M_empty() {}

  template <typename... _Args>
  constexpr _Storage(in_place_t, _Args &&...__args)
      : _M_value(std::forward<_Args>(__args)...) {}

  template <typename _Vp, typename... _Args>
  constexpr _Storage(std::initializer_list<_Vp> __il, _Args &&...__args)
      : _M_value(__il, std::forward<_Args>(__args)...) {}

  // User-provided destructor is needed when _Up has non-trivial dtor.
  ~_Storage() {}

  _Empty_byte _M_empty;
  _Up _M_value;
};

_Storage<_Stored_type> _M_payload;
```

`_Storage`模板的第一个模板参数为`_Up`，代表了实际存储的对象的类型。第二个参数为`is_trivially_destructible_v<_Up>`。当其为false的时候，有一个特化的版本，里面定义了一个析构函数。一般地，这个共用体的默认构造函数初始化了一个`_Empty_byte`对象，是一个空的结构体对象。另外存在两个构造函数，会根据传入的参数初始化一个`_up`类型的变量。

`_Storage`类型的变量`_M_payload`构成了`_Optional_payload_base`的核心成员变量，是此结构体的数据实际存储的地方，也是`std::optional`的实际存储数据的地方。



### 另一数据成员：`_M_engaged`

``` cpp
bool _M_engaged = false;
```

这个变量表明了这个结构体是否真实承载了数据，也可以理解为`_M_payload`为空还是有真实数据的标志。



### 构造、析构、拷贝、赋值等函数

``` cpp
_Optional_payload_base() = default;
~_Optional_payload_base() = default;

template <typename... _Args>
constexpr _Optional_payload_base(in_place_t __tag, _Args &&...__args)
    : _M_payload(__tag, std::forward<_Args>(__args)...), _M_engaged(true) {}

template <typename _Up, typename... _Args>
constexpr _Optional_payload_base(std::initializer_list<_Up> __il,
                                 _Args &&...__args)
    : _M_payload(__il, std::forward<_Args>(__args)...), _M_engaged(true) {}

// Constructor used by _Optional_base copy constructor when the
// contained value is not trivially copy constructible.
constexpr _Optional_payload_base(bool __engaged,
                                 const _Optional_payload_base &__other) {
  if (__other._M_engaged)
    this->_M_construct(__other._M_get());
}

// Constructor used by _Optional_base move constructor when the
// contained value is not trivially move constructible.
constexpr _Optional_payload_base(bool __engaged,
                                 _Optional_payload_base &&__other) {
  if (__other._M_engaged)
    this->_M_construct(std::move(__other._M_get()));
}

// Copy constructor is only used to when the contained value is
// trivially copy constructible.
_Optional_payload_base(const _Optional_payload_base &) = default;

// Move constructor is only used to when the contained value is
// trivially copy constructible.
_Optional_payload_base(_Optional_payload_base &&) = default;

_Optional_payload_base &operator=(const _Optional_payload_base &) = default;

_Optional_payload_base &operator=(_Optional_payload_base &&) = default;
```

注意参数中有`__engaged`参数的拷贝、移动构造函数和不带这个参数的拷贝、移动构造函数的区别。

- 带有这个参数的拷贝、移动构造函数，当other的engaged参数为true时，调用`_M_construct`函数。当包含的具体数据类型**不是trivially copy constructible**的，`_Optional_base`会使用这种函数。
- 不带这个参数的，为默认实现。当包含的具体数据类型是trivially copy constructible的，`_Optional_base`会使用这种函数。



### 其他成员函数

``` cpp
// used to perform non-trivial copy assignment.
constexpr void _M_copy_assign(const _Optional_payload_base &__other) {
  if (this->_M_engaged && __other._M_engaged)
    this->_M_get() = __other._M_get();
  else {
    if (__other._M_engaged)
      this->_M_construct(__other._M_get());
    else
      this->_M_reset();
  }
}

// used to perform non-trivial move assignment.
constexpr void _M_move_assign(_Optional_payload_base &&__other) noexcept(
    __and_v<is_nothrow_move_constructible<_Tp>,
            is_nothrow_move_assignable<_Tp>>) {
  if (this->_M_engaged && __other._M_engaged)
    this->_M_get() = std::move(__other._M_get());
  else {
    if (__other._M_engaged)
      this->_M_construct(std::move(__other._M_get()));
    else
      this->_M_reset();
  }
}
template <typename... _Args>
void _M_construct(_Args &&...__args) noexcept(
    is_nothrow_constructible_v<_Stored_type, _Args...>) {
  ::new ((void *)std::__addressof(this->_M_payload))
      _Stored_type(std::forward<_Args>(__args)...);
  this->_M_engaged = true;
}

constexpr void _M_destroy() noexcept {
  _M_engaged = false;
  _M_payload._M_value.~_Stored_type();
}

// The _M_get() operations have _M_engaged as a precondition.
// They exist to access the contained value with the appropriate
// const-qualification, because _M_payload has had the const removed.

constexpr _Tp &_M_get() noexcept { return this->_M_payload._M_value; }

constexpr const _Tp &_M_get() const noexcept {
  return this->_M_payload._M_value;
}

// _M_reset is a 'safe' operation with no precondition.
constexpr void _M_reset() noexcept {
  if (this->_M_engaged)
    _M_destroy();
}
```



## 2 `_Optional_payload`结构体

### 基本定义

``` cpp
// Class template that manages the payload for optionals.
template <typename _Tp,
          bool /*_HasTrivialDestructor*/ = is_trivially_destructible_v<_Tp>,
          bool /*_HasTrivialCopy */ = is_trivially_copy_assignable_v<_Tp>
              &&is_trivially_copy_constructible_v<_Tp>,
          bool /*_HasTrivialMove */ = is_trivially_move_assignable_v<_Tp>
              &&is_trivially_move_constructible_v<_Tp>>
struct _Optional_payload;
```

它的模板参数第一个是`_Tp`，是实际的数据类型，除此之外，还有三个模板参数：

1. 析构是trivial的吗？
2. 拷贝构造、赋值是trivial的吗？
3. 移动构造、赋值是trivial的吗？

根据这三个的取值不同，有下面几种不同的特化形式：

``` cpp
// Payload for potentially-constexpr optionals (trivial copy/move/destroy).
template <typename _Tp>
struct _Optional_payload<_Tp, true, true, true>
    : _Optional_payload_base<_Tp> {
  using _Optional_payload_base<_Tp>::_Optional_payload_base;

  _Optional_payload() = default;
};

// Payload for optionals with non-trivial copy construction/assignment.
template <typename _Tp>
struct _Optional_payload<_Tp, true, false, true>
    : _Optional_payload_base<_Tp> {
  using _Optional_payload_base<_Tp>::_Optional_payload_base;

  _Optional_payload() = default;
  ~_Optional_payload() = default;
  _Optional_payload(const _Optional_payload &) = default;
  _Optional_payload(_Optional_payload &&) = default;
  _Optional_payload &operator=(_Optional_payload &&) = default;

  // Non-trivial copy assignment.
  constexpr _Optional_payload &operator=(const _Optional_payload &__other) {
    this->_M_copy_assign(__other);
    return *this;
  }
};

// Payload for optionals with non-trivial move construction/assignment.
template <typename _Tp>
struct _Optional_payload<_Tp, true, true, false>
    : _Optional_payload_base<_Tp> {
  using _Optional_payload_base<_Tp>::_Optional_payload_base;

  _Optional_payload() = default;
  ~_Optional_payload() = default;
  _Optional_payload(const _Optional_payload &) = default;
  _Optional_payload(_Optional_payload &&) = default;
  _Optional_payload &operator=(const _Optional_payload &) = default;

  // Non-trivial move assignment.
  constexpr _Optional_payload &
  operator=(_Optional_payload &&__other) noexcept(
      __and_v<is_nothrow_move_constructible<_Tp>,
              is_nothrow_move_assignable<_Tp>>) {
    this->_M_move_assign(std::move(__other));
    return *this;
  }
};

// Payload for optionals with non-trivial copy and move assignment.
template <typename _Tp>
struct _Optional_payload<_Tp, true, false, false>
    : _Optional_payload_base<_Tp> {
  using _Optional_payload_base<_Tp>::_Optional_payload_base;

  _Optional_payload() = default;
  ~_Optional_payload() = default;
  _Optional_payload(const _Optional_payload &) = default;
  _Optional_payload(_Optional_payload &&) = default;

  // Non-trivial copy assignment.
  constexpr _Optional_payload &operator=(const _Optional_payload &__other) {
    this->_M_copy_assign(__other);
    return *this;
  }

  // Non-trivial move assignment.
  constexpr _Optional_payload &
  operator=(_Optional_payload &&__other) noexcept(
      __and_v<is_nothrow_move_constructible<_Tp>,
              is_nothrow_move_assignable<_Tp>>) {
    this->_M_move_assign(std::move(__other));
    return *this;
  }
};
// Payload for optionals with non-trivial destructors.
template <typename _Tp, bool _Copy, bool _Move>
struct _Optional_payload<_Tp, false, _Copy, _Move>
    : _Optional_payload<_Tp, true, false, false> {
  // Base class implements all the constructors and assignment operators:
  using _Optional_payload<_Tp, true, false, false>::_Optional_payload;
  _Optional_payload() = default;
  _Optional_payload(const _Optional_payload &) = default;
  _Optional_payload(_Optional_payload &&) = default;
  _Optional_payload &operator=(const _Optional_payload &) = default;
  _Optional_payload &operator=(_Optional_payload &&) = default;

  // Destructor needs to destroy the contained value:
  ~_Optional_payload() { this->_M_reset(); }
};
```

由上可知，

- 对于拷贝构造、赋值非trivial的，定义拷贝赋值函数，调用`_Optional_payload_base::_M_copy_assign`。
- 对于移动构造、赋值非trivial的，定义移动赋值函数，调用`_Optional_payload_base::_M_move_assign`。
- 对于析构非trivial的，定义了析构函数，调用`_Optional_payload_base::_M_reset`。



## 3 `_Optional_base_impl`类

这个类存在的意义主要是为了给后面的`_Optional_base`类提供一个基类，避免在每个特化版本中写同样的函数。

主要代码：

``` cpp
// Common base class for _Optional_base<T> to avoid repeating these
// member functions in each specialization.
template <typename _Tp, typename _Dp> class _Optional_base_impl {
protected:
  using _Stored_type = remove_const_t<_Tp>;

  // The _M_construct operation has !_M_engaged as a precondition
  // while _M_destruct has _M_engaged as a precondition.
  template <typename... _Args>
  void _M_construct(_Args &&...__args) noexcept(
      is_nothrow_constructible_v<_Stored_type, _Args...>) {
    ::new (std::__addressof(static_cast<_Dp *>(this)->_M_payload._M_payload))
        _Stored_type(std::forward<_Args>(__args)...);
    static_cast<_Dp *>(this)->_M_payload._M_engaged = true;
  }

  void _M_destruct() noexcept {
    static_cast<_Dp *>(this)->_M_payload._M_destroy();
  }

  // _M_reset is a 'safe' operation with no precondition.
  constexpr void _M_reset() noexcept {
    static_cast<_Dp *>(this)->_M_payload._M_reset();
  }

  constexpr bool _M_is_engaged() const noexcept {
    return static_cast<const _Dp *>(this)->_M_payload._M_engaged;
  }

  // The _M_get operations have _M_engaged as a precondition.
  constexpr _Tp &_M_get() noexcept {
    __glibcxx_assert(this->_M_is_engaged());
    return static_cast<_Dp *>(this)->_M_payload._M_get();
  }

  constexpr const _Tp &_M_get() const noexcept {
    __glibcxx_assert(this->_M_is_engaged());
    return static_cast<const _Dp *>(this)->_M_payload._M_get();
  }
};
```

它有两个模板参数，从后文可知，`_Tp`指的是optional存储的实际数据类型，`_Dp=_Optional_base<_Tp>`。从上面的代码基本可以看出，这个类的函数主要提供了对`_Optional_payload`类型变量的操作。



## 4 `_Optional_base`类

这个类提供了`optional`的拷贝、移动构造函数。

``` cpp
template <typename _Tp, bool = is_trivially_copy_constructible_v<_Tp>,
          bool = is_trivially_move_constructible_v<_Tp>>
struct _Optional_base : _Optional_base_impl<_Tp, _Optional_base<_Tp>> {
  // Constructors for disengaged optionals.
  constexpr _Optional_base() = default;

  // Constructors for engaged optionals.
  template <typename... _Args,
            enable_if_t<is_constructible_v<_Tp, _Args...>, bool> = false>
  constexpr explicit _Optional_base(in_place_t, _Args &&...__args)
      : _M_payload(in_place, std::forward<_Args>(__args)...) {}

  template <
      typename _Up, typename... _Args,
      enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>,
                  bool> = false>
  constexpr explicit _Optional_base(in_place_t, initializer_list<_Up> __il,
                                    _Args &&...__args)
      : _M_payload(in_place, __il, std::forward<_Args>(__args)...) {}

  // Copy and move constructors.
  constexpr _Optional_base(const _Optional_base &__other)
      : _M_payload(__other._M_payload._M_engaged, __other._M_payload) {}

  constexpr _Optional_base(_Optional_base &&__other) noexcept(
      is_nothrow_move_constructible_v<_Tp>)
      : _M_payload(__other._M_payload._M_engaged,
                   std::move(__other._M_payload)) {}

  // Assignment operators.
  _Optional_base &operator=(const _Optional_base &) = default;
  _Optional_base &operator=(_Optional_base &&) = default;

  _Optional_payload<_Tp> _M_payload;
};
```

同样的，其最后两个bool类型的参数为：是否可以平凡的拷贝构造、是否可以平凡的移动构造。针对两个bool值的不同，有以下的偏特化版本。

``` cpp
template <typename _Tp>
struct _Optional_base<_Tp, false, true>
    : _Optional_base_impl<_Tp, _Optional_base<_Tp>> {
  // Constructors for disengaged optionals.
  constexpr _Optional_base() = default;

  // Constructors for engaged optionals.
  template <typename... _Args,
            enable_if_t<is_constructible_v<_Tp, _Args...>, bool> = false>
  constexpr explicit _Optional_base(in_place_t, _Args &&...__args)
      : _M_payload(in_place, std::forward<_Args>(__args)...) {}

  template <
      typename _Up, typename... _Args,
      enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>,
                  bool> = false>
  constexpr explicit _Optional_base(in_place_t, initializer_list<_Up> __il,
                                    _Args... __args)
      : _M_payload(in_place, __il, std::forward<_Args>(__args)...) {}

  // Copy and move constructors.
  constexpr _Optional_base(const _Optional_base &__other)
      : _M_payload(__other._M_payload._M_engaged, __other._M_payload) {}

  constexpr _Optional_base(_Optional_base &&__other) = default;

  // Assignment operators.
  _Optional_base &operator=(const _Optional_base &) = default;
  _Optional_base &operator=(_Optional_base &&) = default;

  _Optional_payload<_Tp> _M_payload;
};

template <typename _Tp>
struct _Optional_base<_Tp, true, false>
    : _Optional_base_impl<_Tp, _Optional_base<_Tp>> {
  // Constructors for disengaged optionals.
  constexpr _Optional_base() = default;

  // Constructors for engaged optionals.
  template <typename... _Args,
            enable_if_t<is_constructible_v<_Tp, _Args...>, bool> = false>
  constexpr explicit _Optional_base(in_place_t, _Args &&...__args)
      : _M_payload(in_place, std::forward<_Args>(__args)...) {}

  template <
      typename _Up, typename... _Args,
      enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>,
                  bool> = false>
  constexpr explicit _Optional_base(in_place_t, initializer_list<_Up> __il,
                                    _Args &&...__args)
      : _M_payload(in_place, __il, std::forward<_Args>(__args)...) {}

  // Copy and move constructors.
  constexpr _Optional_base(const _Optional_base &__other) = default;

  constexpr _Optional_base(_Optional_base &&__other) noexcept(
      is_nothrow_move_constructible_v<_Tp>)
      : _M_payload(__other._M_payload._M_engaged,
                   std::move(__other._M_payload)) {}

  // Assignment operators.
  _Optional_base &operator=(const _Optional_base &) = default;
  _Optional_base &operator=(_Optional_base &&) = default;

  _Optional_payload<_Tp> _M_payload;
};

template <typename _Tp>
struct _Optional_base<_Tp, true, true>
    : _Optional_base_impl<_Tp, _Optional_base<_Tp>> {
  // Constructors for disengaged optionals.
  constexpr _Optional_base() = default;

  // Constructors for engaged optionals.
  template <typename... _Args,
            enable_if_t<is_constructible_v<_Tp, _Args...>, bool> = false>
  constexpr explicit _Optional_base(in_place_t, _Args &&...__args)
      : _M_payload(in_place, std::forward<_Args>(__args)...) {}

  template <
      typename _Up, typename... _Args,
      enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>,
                  bool> = false>
  constexpr explicit _Optional_base(in_place_t, initializer_list<_Up> __il,
                                    _Args &&...__args)
      : _M_payload(in_place, __il, std::forward<_Args>(__args)...) {}

  // Copy and move constructors.
  constexpr _Optional_base(const _Optional_base &__other) = default;
  constexpr _Optional_base(_Optional_base &&__other) = default;

  // Assignment operators.
  _Optional_base &operator=(const _Optional_base &) = default;
  _Optional_base &operator=(_Optional_base &&) = default;

  _Optional_payload<_Tp> _M_payload;
};
```

由上面可以看到，不可以平凡拷贝构造的，调用`_Optional_payload`的非默认形式的拷贝构造函数；不可以平凡移动构造的，调用`_Optional_payload`的非默认形式的移动构造函数；



## 5 `optional`类

### 定义

``` cpp
template <typename _Tp>
class optional
    : private _Optional_base<_Tp>,
      private _Enable_copy_move<
          // Copy constructor.
          is_copy_constructible_v<_Tp>,
          // Copy assignment.
          __and_v<is_copy_constructible<_Tp>, is_copy_assignable<_Tp>>,
          // Move constructor.
          is_move_constructible_v<_Tp>,
          // Move assignment.
          __and_v<is_move_constructible<_Tp>, is_move_assignable<_Tp>>,
          // Unique tag type.
          optional<_Tp>>;
```



### using定义

``` cpp
using _Base = _Optional_base<_Tp>;

// SFINAE helpers
template <typename _Up>
using __not_self = __not_<is_same<optional, __remove_cvref_t<_Up>>>;
template <typename _Up>
using __not_tag = __not_<is_same<in_place_t, __remove_cvref_t<_Up>>>;
template <typename... _Cond>
using _Requires = enable_if_t<__and_v<_Cond...>, bool>;

using value_type = _Tp;
```

`__not_self`：要求去掉cvref之后的`_Up`不能是`optional`类型。

`__not_tag`：要求去掉cvref之后的`_Up`不能是不是`in_place_t`这个tag。

`_Requires`：要求后面`_Cond...`代表的条件皆为true。



### `nullopt_t`结构体

``` cpp
/// Tag type to disengage optional objects.
struct nullopt_t {
  // Do not user-declare default constructor at all for
  // optional_value = {} syntax to work.
  // nullopt_t() = delete;

  // Used for constructing nullopt.
  enum class _Construct { _Token };

  // Must be constexpr for nullopt_t to be literal.
  explicit constexpr nullopt_t(_Construct) {}
};
```



### 构造、拷贝构造、移动构造函数

``` cpp
constexpr optional() noexcept {}

constexpr optional(nullopt_t) noexcept {}

// 这个移动构造函数的调用条件，以下皆为true：
// 		1. _Up不是optional的；
//		2. _Up不是in_place_t这个tag；
//		3. _Tp可以用_Up构造；
// 		4. _Tp可以转换为_Up。
template <typename _Up = _Tp, _Requires<__not_self<_Up>, __not_tag<_Up>,
                                        is_constructible<_Tp, _Up>,
                                        is_convertible<_Up, _Tp>> = true>
constexpr optional(_Up &&__t) noexcept(is_nothrow_constructible_v<_Tp, _Up>)
    : _Base(std::in_place, std::forward<_Up>(__t)) {}

// 这个移动构造函数的调用条件，以下至少有一个为false：
// 		1. _Up不是optional的；
//		2. _Up不是in_place_t这个tag；
//		3. _Tp可以用_Up构造；
// 		4. _Tp不可以转换为_Up。
template <
    typename _Up = _Tp,
    _Requires<__not_self<_Up>, __not_tag<_Up>, is_constructible<_Tp, _Up>,
              __not_<is_convertible<_Up, _Tp>>> = false>
explicit constexpr optional(_Up &&__t) noexcept(
    is_nothrow_constructible_v<_Tp, _Up>)
    : _Base(std::in_place, std::forward<_Up>(__t)) {}

// 这个移动构造函数的调用条件，以下皆为true：
// 		1. _Up不是optional的；
//		2. _Tp可以用_Up构造；
// 		3. _Tp不可以转换为_Up；
//		4. _Tp可以用optional<_Up>的某一种形式构造。
template <
    typename _Up,
    _Requires<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, const _Up &>,
              is_convertible<const _Up &, _Tp>,
              __not_<__converts_from_optional<_Tp, _Up>>> = true>
constexpr optional(const optional<_Up> &__t) noexcept(
    is_nothrow_constructible_v<_Tp, const _Up &>) {
  if (__t)
    emplace(*__t);
}

// 这个移动构造函数的调用条件，以下至少有一个为false：
// 		1. _Up不是optional的；
//		2. _Tp可以用_Up构造；
// 		3. _Tp可以转换为_Up；
//		4. _Tp可以用optional<_Up>的某一种形式构造。
template <
    typename _Up,
    _Requires<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, const _Up &>,
              __not_<is_convertible<const _Up &, _Tp>>,
              __not_<__converts_from_optional<_Tp, _Up>>> = false>
explicit constexpr optional(const optional<_Up> &__t) noexcept(
    is_nothrow_constructible_v<_Tp, const _Up &>) {
  if (__t)
    emplace(*__t);
}

// 这个移动构造函数的调用条件，以下皆为true：
// 		1. _Up是optional的；
//		2. _Tp可以用_Up构造；
// 		3. _Tp可以转换为_Up；
//		4. _Tp不可以用optional<_Up>的某一种形式构造。
template <typename _Up,
          _Requires<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, _Up>,
                    is_convertible<_Up, _Tp>,
                    __not_<__converts_from_optional<_Tp, _Up>>> = true>
constexpr optional(optional<_Up> &&__t) noexcept(
    is_nothrow_constructible_v<_Tp, _Up>) {
  if (__t)
    emplace(std::move(*__t));
}

// 这个移动构造函数的调用条件，以下至少有一个为false：
// 		1. _Up是optional的；
//		2. _Tp可以用_Up构造；
// 		3. _Tp不可以转换为_Up；
//		4. _Tp不可以用optional<_Up>的某一种形式构造。
template <typename _Up,
          _Requires<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, _Up>,
                    __not_<is_convertible<_Up, _Tp>>,
                    __not_<__converts_from_optional<_Tp, _Up>>> = false>
explicit constexpr optional(optional<_Up> &&__t) noexcept(
    is_nothrow_constructible_v<_Tp, _Up>) {
  if (__t)
    emplace(std::move(*__t));
}

// 这个移动构造函数的调用条件：_Tp不可以由_Args...构造。
template <typename... _Args,
          _Requires<is_constructible<_Tp, _Args...>> = false>
explicit constexpr optional(in_place_t, _Args &&...__args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
    : _Base(std::in_place, std::forward<_Args>(__args)...) {}

// 这个移动构造函数的调用条件：_Tp不可以由initializer_list<_Up>和_Args...构造。
template <typename _Up, typename... _Args,
          _Requires<is_constructible<_Tp, initializer_list<_Up> &,
                                     _Args...>> = false>
explicit constexpr optional(
    in_place_t, initializer_list<_Up> __il,
    _Args &&...__args) noexcept(is_nothrow_constructible_v<_Tp,
                                                           initializer_list<
                                                               _Up> &,
                                                           _Args...>)
    : _Base(std::in_place, __il, std::forward<_Args>(__args)...) {}
```



### `emplace`函数

``` cpp
template <typename... _Args>
enable_if_t<is_constructible_v<_Tp, _Args...>, _Tp &> emplace(
    _Args &&...__args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>) {
  this->_M_reset();
  this->_M_construct(std::forward<_Args>(__args)...);
  return this->_M_get();
}

template <typename _Up, typename... _Args>
enable_if_t<is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>,
            _Tp &>
emplace(initializer_list<_Up> __il, _Args &&...__args) noexcept(
    is_nothrow_constructible_v<_Tp, initializer_list<_Up> &, _Args...>) {
  this->_M_reset();
  this->_M_construct(__il, std::forward<_Args>(__args)...);
  return this->_M_get();
}
```

这个函数的两种实现方式，主要是输入参数的区别，都是利用输入的参数构造一个对象。



### 拷贝、移动赋值函数

``` cpp
// Assignment operators.
optional &operator=(nullopt_t) noexcept {
  this->_M_reset();
  return *this;
}

// 这个函数的调用条件：
//		1. _Up不是optional的；&&
// 		2. ! (_Tp是标量 && _Tp和decay之后的_Up是一个类型) &&
// 		3. _Tp可以用_Up构造 &&
//		4. 可以用_Up赋值_Tp
template <typename _Up = _Tp>
enable_if_t<
    __and_v<__not_self<_Up>,
            __not_<__and_<is_scalar<_Tp>, is_same<_Tp, decay_t<_Up>>>>,
            is_constructible<_Tp, _Up>, is_assignable<_Tp &, _Up>>,
    optional &>
operator=(_Up &&__u) noexcept(__and_v<is_nothrow_constructible<_Tp, _Up>,
                                      is_nothrow_assignable<_Tp &, _Up>>) {
  if (this->_M_is_engaged())
    this->_M_get() = std::forward<_Up>(__u);
  else
    this->_M_construct(std::forward<_Up>(__u));

  return *this;
}

template <typename _Up>
enable_if_t<
    __and_v<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, const _Up &>,
            is_assignable<_Tp &, _Up>,
            __not_<__converts_from_optional<_Tp, _Up>>,
            __not_<__assigns_from_optional<_Tp, _Up>>>,
    optional &>
operator=(const optional<_Up> &__u) noexcept(
    __and_v<is_nothrow_constructible<_Tp, const _Up &>,
            is_nothrow_assignable<_Tp &, const _Up &>>) {
  if (__u) {
    if (this->_M_is_engaged())
      this->_M_get() = *__u;
    else
      this->_M_construct(*__u);
  } else {
    this->_M_reset();
  }
  return *this;
}

template <typename _Up>
enable_if_t<__and_v<__not_<is_same<_Tp, _Up>>, is_constructible<_Tp, _Up>,
                    is_assignable<_Tp &, _Up>,
                    __not_<__converts_from_optional<_Tp, _Up>>,
                    __not_<__assigns_from_optional<_Tp, _Up>>>,
            optional &>
operator=(optional<_Up> &&__u) noexcept(
    __and_v<is_nothrow_constructible<_Tp, _Up>,
            is_nothrow_assignable<_Tp &, _Up>>) {
  if (__u) {
    if (this->_M_is_engaged())
      this->_M_get() = std::move(*__u);
    else
      this->_M_construct(std::move(*__u));
  } else {
    this->_M_reset();
  }

  return *this;
}
```



### `swap`函数

``` cpp
// Swap.
void swap(optional &__other) noexcept(
    is_nothrow_move_constructible_v<_Tp> &&is_nothrow_swappable_v<_Tp>) {
  using std::swap;

  if (this->_M_is_engaged() && __other._M_is_engaged())
    swap(this->_M_get(), __other._M_get());
  else if (this->_M_is_engaged()) {
    __other._M_construct(std::move(this->_M_get()));
    this->_M_destruct();
  } else if (__other._M_is_engaged()) {
    this->_M_construct(std::move(__other._M_get()));
    __other._M_destruct();
  }
}
```

这个函数的核心点是：当this和other都有值的时候，swap它们具体的值；如果只有一方有值的话，直接利用其来构建另一方，并将其销毁。



### 访问数据成员

``` cpp
// Observers.
constexpr const _Tp *operator->() const noexcept {
	return std::__addressof(this->_M_get());
}

constexpr _Tp *operator->() noexcept {
	return std::__addressof(this->_M_get());
}

constexpr const _Tp &operator*() const &noexcept { return this->_M_get(); }

constexpr _Tp &operator*() &noexcept { return this->_M_get(); }

constexpr _Tp &&operator*() &&noexcept { return std::move(this->_M_get()); }

constexpr const _Tp &&operator*() const &&noexcept {
	return std::move(this->_M_get());
}

constexpr explicit operator bool() const noexcept {
  return this->_M_is_engaged();
}

constexpr bool has_value() const noexcept { return this->_M_is_engaged(); }

constexpr const _Tp &value() const & {
  return this->_M_is_engaged()
             ? this->_M_get()
             : (__throw_bad_optional_access(), this->_M_get());
}

constexpr _Tp &value() & {
  return this->_M_is_engaged()
             ? this->_M_get()
             : (__throw_bad_optional_access(), this->_M_get());
}

constexpr _Tp &&value() && {
  return this->_M_is_engaged()
             ? std::move(this->_M_get())
             : (__throw_bad_optional_access(), std::move(this->_M_get()));
}

constexpr const _Tp &&value() const && {
  return this->_M_is_engaged()
             ? std::move(this->_M_get())
             : (__throw_bad_optional_access(), std::move(this->_M_get()));
}

template <typename _Up> constexpr _Tp value_or(_Up &&__u) const & {
  static_assert(is_copy_constructible_v<_Tp>);
  static_assert(is_convertible_v<_Up &&, _Tp>);

  return this->_M_is_engaged() ? this->_M_get()
                               : static_cast<_Tp>(std::forward<_Up>(__u));
}

template <typename _Up> constexpr _Tp value_or(_Up &&__u) && {
  static_assert(is_move_constructible_v<_Tp>);
  static_assert(is_convertible_v<_Up &&, _Tp>);

  return this->_M_is_engaged() ? std::move(this->_M_get())
                               : static_cast<_Tp>(std::forward<_Up>(__u));
}
```

这两个`operator*`和`operator->`直接访问的是`optional`所承载的实际数据成员。

`value`函数提供了对实际数据成员的访问，但是当它没有承载实际值的时候会抛出异常，这是和上面两个operator的区别。

`value_or`函数的意思：当它承载实际值时，返回这个值；否则，返回输入的参数。



## 6 相关函数

### `optional`类型之间比较

``` cpp
template <typename _Tp, typename _Up>
constexpr auto operator<(const optional<_Tp> &__lhs,
                         const optional<_Up> &__rhs)
    ->__optional_lt_t<_Tp, _Up> {
  return static_cast<bool>(__rhs) && (!__lhs || *__lhs < *__rhs);
}

template <typename _Tp, typename _Up>
constexpr auto operator>(const optional<_Tp> &__lhs,
                         const optional<_Up> &__rhs)
    ->__optional_gt_t<_Tp, _Up> {
  return static_cast<bool>(__lhs) && (!__rhs || *__lhs > *__rhs);
}
```

一般会先判断`optional`对象是否有值，然后判断它们的值的关系。



### `optional`与`nullopt`类型的比较

``` cpp
template <typename _Tp>
constexpr bool operator<(const optional<_Tp> & /* __lhs */,
                         nullopt_t) noexcept {
  return false;
}

template <typename _Tp>
constexpr bool operator<(nullopt_t, const optional<_Tp> &__rhs) noexcept {
  return static_cast<bool>(__rhs);
}

template <typename _Tp>
constexpr bool operator>(const optional<_Tp> &__lhs, nullopt_t) noexcept {
  return static_cast<bool>(__lhs);
}

template <typename _Tp>
constexpr bool operator>(nullopt_t,
                         const optional<_Tp> & /* __rhs */) noexcept {
  return false;
}
```

判断`nullopt`是否大于某个值，一定是false的；判断`nullopt`是否小于某个值，要看这个值是否engaged。



### `optional`与普通值的比较

``` cpp
template <typename _Tp, typename _Up>
constexpr auto operator==(const optional<_Tp> &__lhs, const _Up &__rhs)
    ->__optional_eq_t<_Tp, _Up> {
  return __lhs && *__lhs == __rhs;
}

template <typename _Tp, typename _Up>
constexpr auto operator==(const _Up &__lhs, const optional<_Tp> &__rhs)
    ->__optional_eq_t<_Up, _Tp> {
  return __rhs && __lhs == *__rhs;
}
```

先判断`optional`类型是否engaged，如果没有engaged的话返回false；如果engaged的话，再取出来里面的值和另外一值比较。



### `swap`函数

``` cpp
template <typename _Tp>
inline enable_if_t<is_move_constructible_v<_Tp> && is_swappable_v<_Tp>> swap(
    optional<_Tp> & __lhs,
    optional<_Tp> & __rhs) noexcept(noexcept(__lhs.swap(__rhs))) {
  __lhs.swap(__rhs);
}

template <typename _Tp>
enable_if_t<!(is_move_constructible_v<_Tp> && is_swappable_v<_Tp>)> swap(
    optional<_Tp> &, optional<_Tp> &) = delete;
```

当`optional`的模板参数类型，即实际存储的值的类型，具备两个条件时：可移动构造、可交换的，调用`optional`的类成员函数`swap`。



### `make_optiional`函数

``` cpp
template <typename _Tp>
constexpr enable_if_t<is_constructible_v<decay_t<_Tp>, _Tp>,
                      optional<decay_t<_Tp>>>
make_optional(_Tp && __t) noexcept(
    is_nothrow_constructible_v<optional<decay_t<_Tp>>, _Tp>) {
  return optional<decay_t<_Tp>>{std::forward<_Tp>(__t)};
}

template <typename _Tp, typename... _Args>
constexpr enable_if_t<is_constructible_v<_Tp, _Args...>, optional<_Tp>>
make_optional(_Args &&
              ...__args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>) {
  return optional<_Tp>{in_place, std::forward<_Args>(__args)...};
}

template <typename _Tp, typename _Up, typename... _Args>
constexpr enable_if_t<
    is_constructible_v<_Tp, initializer_list<_Up> &, _Args...>, optional<_Tp>>
make_optional(initializer_list<_Up> __il, _Args && ...__args) noexcept(
    is_nothrow_constructible_v<_Tp, initializer_list<_Up> &, _Args...>) {
  return optional<_Tp>{in_place, __il, std::forward<_Args>(__args)...};
}
```

这个函数的主要逻辑就是调用`optional`的构造函数。



### hash相关函数

``` cpp
template <typename _Tp, typename _Up = remove_const_t<_Tp>,
          bool = __poison_hash<_Up>::__enable_hash_call>
struct __optional_hash_call_base {
  size_t operator()(const optional<_Tp> &__t) const
      noexcept(noexcept(hash<_Up>{}(*__t))) {
    // We pick an arbitrary hash for disengaged optionals which hopefully
    // usual values of _Tp won't typically hash to.
    constexpr size_t __magic_disengaged_hash = static_cast<size_t>(-3333);
    return __t ? hash<_Up>{}(*__t) : __magic_disengaged_hash;
  }
};

template <typename _Tp, typename _Up>
struct __optional_hash_call_base<_Tp, _Up, false> {};

template <typename _Tp>
struct hash<optional<_Tp>> : private __poison_hash<remove_const_t<_Tp>>,
                             public __optional_hash_call_base<_Tp> {
  using result_type [[__deprecated__]] = size_t;
  using argument_type [[__deprecated__]] = optional<_Tp>;
};
```

当`optional`类型是engaged的时候，使用它存储的实际值来进行hash；当engaged为false的时候，返回的值时自定义的一个magic number。



## 7 问题

- 为什么经常遇见同时存在这两种的构造函数呢？即参数为`(Args... args)`和`(initializer_list<U>, Args... args)`。万能引用不是也可以指代初始化列表吗？
