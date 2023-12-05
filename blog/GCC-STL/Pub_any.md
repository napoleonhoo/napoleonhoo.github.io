# any

## 1 主要变量与函数

### 存储

``` cpp
// Holds either pointer to a heap object or the contained object itself.
union _Storage {
  constexpr _Storage() : _M_ptr{nullptr} {}

  // Prevent trivial copies of this type, buffer might hold a non-POD.
  _Storage(const _Storage &) = delete;
  _Storage &operator=(const _Storage &) = delete;

  void *_M_ptr;
  aligned_storage<sizeof(_M_ptr), alignof(void *)>::type _M_buffer;
};
```

这个共用体`_Storage`是any类的存储实体，它可以使用`_M_buffer`作为实际的存储，或者使用`_M_ptr`指向在heap上存储的实际对象。



### 那么如何判定是存储在哪儿的呢？

``` cpp
template <typename _Tp, typename _Safe = is_nothrow_move_constructible<_Tp>,
          bool _Fits = (sizeof(_Tp) <= sizeof(_Storage)) &&
                       (alignof(_Tp) <= alignof(_Storage))>
using _Internal = std::integral_constant<bool, _Safe::value && _Fits>;
```

所以，如果一个类型是不抛异常的可移动构造，且能放入进`_Storage`的，则是internal的。



对于是否internal，对应有不同的manager：

``` cpp
template <typename _Tp>
using _Manager =
    conditional_t<_Internal<_Tp>::value, _Manager_internal<_Tp>,
                  _Manager_external<_Tp>>;
```



### 主要数据成员

``` cpp
private:
  enum _Op {
    _Op_access,
    _Op_get_type_info,
    _Op_clone,
    _Op_destroy,
    _Op_xfer
  };

  union _Arg {
    void *_M_obj;
    const std::type_info *_M_typeinfo;
    any *_M_any;
  };

  void (*_M_manager)(_Op, const any *, _Arg *);
  _Storage _M_storage;
```

`_M_manager`就是负责管理具体数据的函数指针，具体见下。

从具体代码可以看出，共用体`_Arg`的两个主要成员变量：`_M_obj`指向实际的数据位置，如`_M_buffer`或`_M_ptr`；`_M_any`指向一个any对象。

`_Op`、`_Arg`一般都是在`_M_manager`内使用的。



### 内部、外部存储manager

``` cpp
// Manage in-place contained object.
template <typename _Tp> struct _Manager_internal {
  static void _S_manage(_Op __which, const any *__any, _Arg *__arg)  {
    // The contained object is in _M_storage._M_buffer
    auto __ptr = reinterpret_cast<const _Tp *>(&__any->_M_storage._M_buffer);
    switch (__which) {
    case _Op_access:
      __arg->_M_obj = const_cast<_Tp *>(__ptr);
      break;
    case _Op_get_type_info:
#if __cpp_rtti
      __arg->_M_typeinfo = &typeid(_Tp);
#endif
      break;
    case _Op_clone:
      ::new (&__arg->_M_any->_M_storage._M_buffer) _Tp(*__ptr);
      __arg->_M_any->_M_manager = __any->_M_manager;
      break;
    case _Op_destroy:
      __ptr->~_Tp();
      break;
    case _Op_xfer:
      ::new (&__arg->_M_any->_M_storage._M_buffer)
          _Tp(std::move(*const_cast<_Tp *>(__ptr)));
      __ptr->~_Tp();
      __arg->_M_any->_M_manager = __any->_M_manager;
      const_cast<any *>(__any)->_M_manager = nullptr;
      break;
    }
  }

  template <typename _Up>
  static void _S_create(_Storage &__storage, _Up &&__value) {
    void *__addr = &__storage._M_buffer;
    ::new (__addr) _Tp(std::forward<_Up>(__value));
  }

  template <typename... _Args>
  static void _S_create(_Storage &__storage, _Args &&...__args) {
    void *__addr = &__storage._M_buffer;
    ::new (__addr) _Tp(std::forward<_Args>(__args)...);
  }
};

// Manage external contained object.
template <typename _Tp> struct _Manager_external {
  static void _S_manage(_Op __which, const any *__any, _Arg *__arg) ) {
    // The contained object is *_M_storage._M_ptr
    auto __ptr = static_cast<const _Tp *>(__any->_M_storage._M_ptr);
    switch (__which) {
    case _Op_access:
      __arg->_M_obj = const_cast<_Tp *>(__ptr);
      break;
    case _Op_get_type_info:
#if __cpp_rtti
      __arg->_M_typeinfo = &typeid(_Tp);
#endif
      break;
    case _Op_clone:
      __arg->_M_any->_M_storage._M_ptr = new _Tp(*__ptr);
      __arg->_M_any->_M_manager = __any->_M_manager;
      break;
    case _Op_destroy:
      delete __ptr;
      break;
    case _Op_xfer:
      __arg->_M_any->_M_storage._M_ptr = __any->_M_storage._M_ptr;
      __arg->_M_any->_M_manager = __any->_M_manager;
      const_cast<any *>(__any)->_M_manager = nullptr;
      break;
    }
  }

  template <typename _Up>
  static void _S_create(_Storage &__storage, _Up &&__value) {
    __storage._M_ptr = new _Tp(std::forward<_Up>(__value));
  }
  template <typename... _Args>
  static void _S_create(_Storage &__storage, _Args &&...__args) {
    __storage._M_ptr = new _Tp(std::forward<_Args>(__args)...);
  }
};
```

`_S_manage`函数有以下的几种模式：`_Op_access`：访问；`_Op_clone`：用于拷贝；`_Op_destroy`：用于销毁；`_Op_xfer`：用于移动。

`_S_create`函数，都是获取`__storage.M_buffer`，然后在上面创建`_Tp`的对象。

`_Manager_internal`、`_Manager_external`的区别主要是内部manager使用内部buffer创建对象，而外部manager在heap上创建对象。



### `has_value`函数

``` cpp
bool has_value() const noexcept { return _M_manager != nullptr; }
```

返回`_M_manager`是否是`nullptr`。



### 默认、拷贝、移动构造函数

``` cpp
constexpr any() noexcept : _M_manager(nullptr) {}
```

默认构造函数：创建一个空的对象。



``` cpp
any(const any &__other) {
  if (!__other.has_value())
    _M_manager = nullptr;
  else {
    _Arg __arg;
    __arg._M_any = this;
    __other._M_manager(_Op_clone, &__other, &__arg);
  }
}
```

拷贝构造函数：

- 当other是空的话，直接设置`_M_manager`为`nullptr`。
- 否则，调用other的`_M_manager`函数，将other**拷贝**到this，op参数为`_Op_clone`。



``` cpp
any(any &&__other) noexcept {
  if (!__other.has_value())
    _M_manager = nullptr;
  else {
    _Arg __arg;
    __arg._M_any = this;
    __other._M_manager(_Op_xfer, &__other, &__arg);
  }
}
```

移动构造函数：

- 当other是空的话，直接设置`_M_manager`为`nullptr`。
- 否则，调用other的`_M_manager`函数，将other**移动**到this，op参数为`_Op_xfer`。



### 其它构造函数

``` cpp
template <typename _Tp, typename _VTp = _Decay_if_not_any<_Tp>,
					typename _Mgr = _Manager<_VTp>,
					enable_if_t<is_copy_constructible<_VTp>::value &&
  												!__is_in_place_type<_VTp>::value,
											bool> = true>
  any(_Tp &&__value) : _M_manager(&_Mgr::_S_manage) {
  _Mgr::_S_create(_M_storage, std::forward<_Tp>(__value));
}

template <typename _Tp, typename... _Args, typename _VTp = decay_t<_Tp>,
          typename _Mgr = _Manager<_VTp>,
          __any_constructible_t<_VTp, _Args &&...> = false>
  explicit any(in_place_type_t<_Tp>, _Args &&...__args)
  : _M_manager(&_Mgr::_S_manage) {
    _Mgr::_S_create(_M_storage, std::forward<_Args>(__args)...);
}

template <
		typename _Tp, typename _Up, typename... _Args,
    typename _VTp = decay_t<_Tp>, typename _Mgr = _Manager<_VTp>,
    __any_constructible_t<_VTp, initializer_list<_Up>, _Args &&...> = false>
  explicit any(in_place_type_t<_Tp>, initializer_list<_Up> __il,
               _Args &&...__args)
  : _M_manager(&_Mgr::_S_manage) {
    _Mgr::_S_create(_M_storage, __il, std::forward<_Args>(__args)...);
}
```

以上的三个函数都用来从某一个类型中构造出来any对象，它们的区别如下：

- 第一个函数的调用场景为：当decay之后的参数类型`_Tp`是可以拷贝构造的，且不是必须原地构造的。这种情况，直接拷贝一个传入的`__value`来构造一个any对象。
- 第二个函数的调用场景为：当decay之后的参数类型`_Tp`是可以拷贝构造的，且使用`<_Tp, _Args &&...>`是可以被构造的。这种情况，直接传入参数`__args`来构造一个any对象。
- 第三个函数的调用场景基本和第二个相同，区别在于构造的时候对了一个初始化列表来构造any对象。



备注：

- `__is_in_place_type`的作用：表明是需要原地构造的。
- `__any_constructible_t`的定义，即`_Tp`是否可以拷贝构造且利用`_Args...`构造：

``` cpp
template <typename _Res, typename _Tp, typename... _Args>
using __any_constructible =
		enable_if<__and_<is_copy_constructible<_Tp>,
									 	 is_constructible<_Tp, _Args...>>::value,
							_Res>;

template <typename _Tp, typename... _Args>
using __any_constructible_t =
		typename __any_constructible<bool, _Tp, _Args...>::type;
```



### 析构函数

``` cpp
~any() { reset(); }

void reset() noexcept {
  if (has_value()) {
    _M_manager(_Op_destroy, this, nullptr);
    _M_manager = nullptr;
  }
}
```

析构函数实际调用的`reset`函数，当this不为空的时候，调用`_M_manager`，op参数为`_Op_destroy`将对象销毁。



### 拷贝、移动赋值函数

``` cpp
any &operator=(const any &__rhs) {
  *this = any(__rhs);
  return *this;
}
```



``` cpp
any &operator=(any &&__rhs) noexcept {
  if (!__rhs.has_value())
    reset();
  else if (this != &__rhs) {
    reset();
    _Arg __arg;
    __arg._M_any = this;
    __rhs._M_manager(_Op_xfer, &__rhs, &__arg);
  }
  return *this;
}
```

当rhs为空时，需要销毁this。当this和rhs属于不同对象时，销毁this，调用rhs的`_M_manager`将rhs移动到this，op参数为`_Op_xfer`。



### 其他赋值函数

``` cpp
template <typename _Tp>
enable_if_t<is_copy_constructible<_Decay_if_not_any<_Tp>>::value, any &>
operator=(_Tp &&__rhs) {
  *this = any(std::forward<_Tp>(__rhs));
  return *this;
}
```

调用上述其他构造函数中的第一个。



### `emplace`函数

``` cpp
template <typename _Tp, typename... _Args, typename _Mgr = _Manager<_Tp>>
void __do_emplace(_Args &&...__args) {
  reset();
  _Mgr::_S_create(_M_storage, std::forward<_Args>(__args)...);
  _M_manager = &_Mgr::_S_manage;
}

template <typename _Tp, typename _Up, typename... _Args,
typename _Mgr = _Manager<_Tp>>
  void __do_emplace(initializer_list<_Up> __il, _Args &&...__args) {
  reset();
  _Mgr::_S_create(_M_storage, __il, std::forward<_Args>(__args)...);
  _M_manager = &_Mgr::_S_manage;
}

template <typename _VTp, typename... _Args>
using __emplace_t =
    typename __any_constructible<_VTp &, _VTp, _Args...>::type;

template <typename _Tp, typename... _Args>
__emplace_t<decay_t<_Tp>, _Args...> emplace(_Args &&...__args) {
  using _VTp = decay_t<_Tp>;
  __do_emplace<_VTp>(std::forward<_Args>(__args)...);
  any::_Arg __arg;
  this->_M_manager(any::_Op_access, this, &__arg);
  return *static_cast<_VTp *>(__arg._M_obj);
}

template <typename _Tp, typename _Up, typename... _Args>
__emplace_t<decay_t<_Tp>, initializer_list<_Up>, _Args &&...>
emplace(initializer_list<_Up> __il, _Args &&...__args) {
  using _VTp = decay_t<_Tp>;
  __do_emplace<_VTp, _Up>(__il, std::forward<_Args>(__args)...);
  any::_Arg __arg;
  this->_M_manager(any::_Op_access, this, &__arg);
  return *static_cast<_VTp *>(__arg._M_obj);
}
```

`emplace`函数利用传入的参数构造对象，并取代this中的对象。具体的做法：

1. 销毁目前持有的对象；
2. 利用传入的参数构造新的对象；
3. 获取实际的存储指针，并返回。



### `swap`函数

``` cpp
void swap(any &__rhs) noexcept {
  if (!has_value() && !__rhs.has_value())
    return;

  if (has_value() && __rhs.has_value()) {
    if (this == &__rhs)
      return;

    any __tmp;
    _Arg __arg;
    __arg._M_any = &__tmp;
    __rhs._M_manager(_Op_xfer, &__rhs, &__arg);
    __arg._M_any = &__rhs;
    _M_manager(_Op_xfer, this, &__arg);
    __arg._M_any = this;
    __tmp._M_manager(_Op_xfer, &__tmp, &__arg);
  } else {
    any *__empty = !has_value() ? this : &__rhs;
    any *__full = !has_value() ? &__rhs : this;
    _Arg __arg;
    __arg._M_any = __empty;
    __full->_M_manager(_Op_xfer, __full, &__arg);
  }
}
```

- 当this和rhs都为空时，直接返回，不需任何操作。
- 当二者都不为空时，创建一个中间变量，利用它来做swap。
- 当二者有一个为空是，直接将不为空的**移动**到为空的那个即可。



## 2 相关函数

### `swap`函数

``` cpp
inline void swap(any & __x, any & __y) noexcept { __x.swap(__y); }
```

直接调用对象的成员函数。



### `make_any`函数

``` cpp
/// Create an any holding a @c _Tp constructed from @c __args.
template <typename _Tp, typename... _Args> any make_any(_Args && ...__args) {
  return any(in_place_type<_Tp>, std::forward<_Args>(__args)...);
}

/// Create an any holding a @c _Tp constructed from @c __il and @c __args.
template <typename _Tp, typename _Up, typename... _Args>
any make_any(initializer_list<_Up> __il, _Args && ...__args) {
  return any(in_place_type<_Tp>, __il, std::forward<_Args>(__args)...);
}
```

这个函数的主要作用就是用传入的参数调用any的构造函数来构造一个any对象。



### `any_cast`相关函数

``` cpp
template <typename _ValueType> inline _ValueType any_cast(const any &__any) {
  using _Up = __remove_cvref_t<_ValueType>;
  static_assert(
      any::__is_valid_cast<_ValueType>(),
      "Template argument must be a reference or CopyConstructible type");
  static_assert(
      is_constructible_v<_ValueType, const _Up &>,
      "Template argument must be constructible from a const value.");
  auto __p = any_cast<_Up>(&__any);
  if (__p)
    return static_cast<_ValueType>(*__p);
  __throw_bad_any_cast();
}

template <typename _ValueType> inline _ValueType any_cast(any & __any) {
  using _Up = __remove_cvref_t<_ValueType>;
  static_assert(
      any::__is_valid_cast<_ValueType>(),
      "Template argument must be a reference or CopyConstructible type");
  static_assert(is_constructible_v<_ValueType, _Up &>,
                "Template argument must be constructible from an lvalue.");
  auto __p = any_cast<_Up>(&__any);
  if (__p)
    return static_cast<_ValueType>(*__p);
  __throw_bad_any_cast();
}

template <typename _ValueType> inline _ValueType any_cast(any && __any) {
  using _Up = __remove_cvref_t<_ValueType>;
  static_assert(
      any::__is_valid_cast<_ValueType>(),
      "Template argument must be a reference or CopyConstructible type");
  static_assert(is_constructible_v<_ValueType, _Up>,
                "Template argument must be constructible from an rvalue.");
  auto __p = any_cast<_Up>(&__any);
  if (__p)
    return static_cast<_ValueType>(std::move(*__p));
  __throw_bad_any_cast();
}

template <typename _Tp> void *__any_caster(const any *__any) {
  // any_cast<T> returns non-null if __any->type() == typeid(T) and
  // typeid(T) ignores cv-qualifiers so remove them:
  using _Up = remove_cv_t<_Tp>;
  // The contained value has a decayed type, so if decay_t<U> is not U,
  // then it's not possible to have a contained value of type U:
  if constexpr (!is_same_v<decay_t<_Up>, _Up>)
    return nullptr;
  // Only copy constructible types can be used for contained values:
  else if constexpr (!is_copy_constructible_v<_Up>)
    return nullptr;
  // First try comparing function addresses, which works without RTTI
  else if (__any->_M_manager == &any::_Manager<_Up>::_S_manage) {
    any::_Arg __arg;
    __any->_M_manager(any::_Op_access, __any, &__arg);
    return __arg._M_obj;
  }
  return nullptr;
}

template <typename _ValueType>
inline const _ValueType *any_cast(const any *__any) noexcept {
  if constexpr (is_object_v<_ValueType>)
    if (__any)
      return static_cast<_ValueType *>(__any_caster<_ValueType>(__any));
  return nullptr;
}

template <typename _ValueType>
inline _ValueType *any_cast(any * __any) noexcept {
  if constexpr (is_object_v<_ValueType>)
    if (__any)
      return static_cast<_ValueType *>(__any_caster<_ValueType>(__any));
  return nullptr;
}
```

`any_cast`函数主要有两种形式：输入为引用和输入为指针的。它们的具体操作简述如下：

- 输入为引用的`any_cast`函数会取输入参数的地址，然后调用输入为指针的`any_cast`函数。
- 调用`__any_cast`函数，它的作用主要是获取any对象指向具体对象的指针，并返回。
- 将`__any_cast`函数的返回值强制转换为需要的类型并返回。



## 问题

- any的拷贝移动函数调用的不是自己吗？

