# § string

## basic\_string (\_GLIBCXX\_USE\_CXX11\_ABI)

### 简介


### 定义

``` cpp
  template<typename _CharT, typename _Traits, typename _Alloc>
    class basic_string
```

模板参数中`_CharT`是字符的类型，如`char/wchar`等。

### 成员变量

``` cpp
    // Use empty-base optimization: http://www.cantrip.org/emptyopt.html
    struct _Alloc_hider : allocator_type // TODO check __is_final
    {
#if __cplusplus < 201103L
      _Alloc_hider(pointer __dat, const _Alloc &__a = _Alloc())
          : allocator_type(__a), _M_p(__dat) {}
#else
      _Alloc_hider(pointer __dat, const _Alloc &__a)
          : allocator_type(__a), _M_p(__dat) {}

      _Alloc_hider(pointer __dat, _Alloc &&__a = _Alloc())
          : allocator_type(std::move(__a)), _M_p(__dat) {}
#endif

      pointer _M_p; // The actual data.
    };

    _Alloc_hider _M_dataplus;

    union {
      _CharT _M_local_buf[_S_local_capacity + 1];
      size_type _M_allocated_capacity;
    };
```

实际指向数据的指针是：`_M_dataplus._M_p`。当数据量小于`_S_local_capacity`时，使用本地buf，指向`_M_local_buf`。当大于时，这个指向外部的一个buf。

### 构造函数

``` cpp
  // 创建一个空的字符串。
  basic_string()
        _GLIBCXX_NOEXCEPT_IF(is_nothrow_default_constructible<_Alloc>::value)
        : _M_dataplus(_M_local_data()) {
      _M_set_length(0);
    }
    // 和上面的函数的区别：是传入了allocator，并用它来分配空间。
    explicit basic_string(const _Alloc &__a) _GLIBCXX_NOEXCEPT
        : _M_dataplus(_M_local_data(), __a) {
      _M_set_length(0);
    }

    // 拷贝构造函数
    basic_string(const basic_string &__str)
        : _M_dataplus(_M_local_data(), _Alloc_traits::_S_select_on_copy(
                                           __str._M_get_allocator())) {
      _M_construct(__str._M_data(), __str._M_data() + __str.length());
    }
    basic_string(const basic_string &__str, const _Alloc &__a)
        : _M_dataplus(_M_local_data(), __a) {
      _M_construct(__str.begin(), __str.end());
    }

    // 和上面的区别是多一个__pos和allocator。它可以用自己的内存分配器从另一个字符串的子串中构造。
    basic_string(const basic_string &__str, size_type __pos,
                 const _Alloc &__a = _Alloc())
        : _M_dataplus(_M_local_data(), __a) {
      const _CharT *__start =
          __str._M_data() + __str._M_check(__pos, "basic_string::basic_string");
      _M_construct(__start, __start + __str._M_limit(__pos, npos));
    }
    // 和上面的区别是对一个__n参数，它从__str的__pos位置拷贝__n个字符来构造一个字符串。
    basic_string(const basic_string &__str, size_type __pos, size_type __n)
        : _M_dataplus(_M_local_data()) {
      const _CharT *__start =
          __str._M_data() + __str._M_check(__pos, "basic_string::basic_string");
      _M_construct(__start, __start + __str._M_limit(__pos, __n));
    }
    basic_string(const basic_string &__str, size_type __pos, size_type __n,
                 const _Alloc &__a)
        : _M_dataplus(_M_local_data(), __a) {
      const _CharT *__start =
          __str._M_data() + __str._M_check(__pos, "string::string");
      _M_construct(__start, __start + __str._M_limit(__pos, __n));
    }
    basic_string(const _CharT *__s, size_type __n, const _Alloc &__a = _Alloc())
        : _M_dataplus(_M_local_data(), __a) {
      _M_construct(__s, __s + __n);
    }

    // 这个函数用一个C风格的字符串（如：char*）中拷贝并构造一个字符串。
    basic_string(const _CharT *__s, const _Alloc &__a = _Alloc())
        : _M_dataplus(_M_local_data(), __a) {
      _M_construct(__s, __s ? __s + traits_type::length(__s) : __s + npos);
    }

    // 构造一个字符串，它的内容是__n个__c。
    basic_string(size_type __n, _CharT __c, const _Alloc &__a = _Alloc())
        : _M_dataplus(_M_local_data(), __a) {
      _M_construct(__n, __c);
    }

    // 移动构造函数
    // 当__str使用的是本地buf时，使用trais_type::copy函数直接拷贝。
    // 否则，将本string的数据指针指向__str的。
    // 最后，将__str的数据指针指向本地buf，比国内设置为0。
    basic_string(basic_string &&__str) noexcept
        : _M_dataplus(_M_local_data(), std::move(__str._M_get_allocator())) {
      if (__str._M_is_local()) {
        traits_type::copy(_M_local_buf, __str._M_local_buf,
                          _S_local_capacity + 1);
      } else {
        _M_data(__str._M_data());
        _M_capacity(__str._M_allocated_capacity);
      }

      // Must use _M_length() here not _M_set_length() because
      // basic_stringbuf relies on writing into unallocated capacity so
      // we mess up the contents if we put a '\0' in the string.
      _M_length(__str.length());
      __str._M_data(__str._M_local_data());
      __str._M_set_length(0);
    }
    basic_string(basic_string &&__str,
                 const _Alloc &__a) noexcept(_Alloc_traits::_S_always_equal())
        : _M_dataplus(_M_local_data(), __a) {
      if (__str._M_is_local()) {
        traits_type::copy(_M_local_buf, __str._M_local_buf,
                          _S_local_capacity + 1);
        _M_length(__str.length());
        __str._M_set_length(0);
      } else if (_Alloc_traits::_S_always_equal() ||
                 __str.get_allocator() == __a) {
        _M_data(__str._M_data());
        _M_length(__str.length());
        _M_capacity(__str._M_allocated_capacity);
        __str._M_data(__str._M_local_buf);
        __str._M_set_length(0);
      } else
        _M_construct(__str.begin(), __str.end());
    }
    
    // 利用initializer_list的构造函数
    basic_string(initializer_list<_CharT> __l, const _Alloc &__a = _Alloc())
        : _M_dataplus(_M_local_data(), __a) {
      _M_construct(__l.begin(), __l.end());
    }

    // 从string_view构造字符串
    template <typename _Tp, typename = _If_sv<_Tp, void>>
    basic_string(const _Tp &__t, size_type __pos, size_type __n,
                 const _Alloc &__a = _Alloc())
        : basic_string(_S_to_string_view(__t).substr(__pos, __n), __a) {}
    template <typename _Tp, typename = _If_sv<_Tp, void>>
    explicit basic_string(const _Tp &__t, const _Alloc &__a = _Alloc())
        : basic_string(__sv_wrapper(_S_to_string_view(__t)), __a) {}
```

#### \_M\_construct

``` cpp
    // 根据输入参数是否是整型类型，调用不同的函数。
    template <typename _InIterator>
    void _M_construct(_InIterator __beg, _InIterator __end) {
      typedef typename std::__is_integer<_InIterator>::__type _Integral;
      _M_construct_aux(__beg, __end, _Integral());
    }

    // 当输入参数类型不为整型类型时，根据其是否为input iterator有两种不同的处理流程。
    // _M_construct_aux is used to implement the 21.3.1 para 15 which
    // requires special behaviour if _InIterator is an integral type
    template <typename _InIterator>
    void _M_construct_aux(_InIterator __beg, _InIterator __end,
                          std::__false_type) {
      typedef typename iterator_traits<_InIterator>::iterator_category _Tag;
      _M_construct(__beg, __end, _Tag());
    }

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 438. Ambiguity in the "do the right thing" clause
    template <typename _Integer>
    void _M_construct_aux(_Integer __beg, _Integer __end, std::__true_type) {
      _M_construct_aux_2(static_cast<size_type>(__beg), __end);
    }

    void _M_construct_aux_2(size_type __req, _CharT __c) {
      _M_construct(__req, __c);
    }

    // For Input Iterators, used in istreambuf_iterators, etc.
    // NB: This is the special case for Input Iterators, used in
    // istreambuf_iterators, etc.
    // Input Iterators have a cost structure very different from
    // pointers, calling for a different coding style.
    template <typename _InIterator>
    void _M_construct(_InIterator __beg, _InIterator __end,
                      std::input_iterator_tag)  {
      size_type __len = 0;
      size_type __capacity = size_type(_S_local_capacity);

      // 当__len < _S_local_capacity的时候，持续将数据复制进入_M_data()。
      // 此时，_M_data指向的是本地buf。
      while (__beg != __end && __len < __capacity) {
        _M_data()[__len++] = *__beg;
        ++__beg;
      }

      __try {
        // 当未复制完成时，持续将数据复制进入_M_data()。
        while (__beg != __end) {
          // 当_len == _S_local_capacity的时候，创建一个长度为__len+1的内存（__another），
          // 并将数据从_M_data()复制到__another中，并把_M_data()指向__another。
          // 奇怪的现象：每次只分配多一个，不知道是否和注释中的istreambuf_iterator相关。
          if (__len == __capacity) {
            // Allocate more space.
            __capacity = __len + 1;
            pointer __another = _M_create(__capacity, __len);
            this->_S_copy(__another, _M_data(), __len);
            _M_dispose();
            _M_data(__another);
            _M_capacity(__capacity);
          }
          _M_data()[__len++] = *__beg;
          ++__beg;
        }
      }
      __catch(...) {
        _M_dispose();
        __throw_exception_again;
      }

      _M_set_length(__len);
    }

    // 创建一个大小为__end-__beg大小的内存。并调用_S_copy_chars复制数据。
    // For forward_iterators up to random_access_iterators, used for
    // string::iterator, _CharT*, etc.
    template <typename _FwdIterator>
    void _M_construct(_FwdIterator __beg, _FwdIterator __end,
                      std::forward_iterator_tag) {
      // NB: Not required, but considered best practice.
      if (__gnu_cxx::__is_null_pointer(__beg) && __beg != __end)
        std::__throw_logic_error(__N("basic_string::"
                                    "_M_construct null not valid"));

      size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));

      if (__dnew > size_type(_S_local_capacity)) {
        _M_data(_M_create(__dnew, size_type(0)));
        _M_capacity(__dnew);
      }

      // Check for out_of_range and length_error exceptions.
      __try {
        this->_S_copy_chars(_M_data(), __beg, __end);
      }
      __catch(...) {
        _M_dispose();
        __throw_exception_again;
      }

      _M_set_length(__dnew);
    }

    // 根据__n的是否大于_S_local_capacity来决定是否创建一个内存块，并赋值。
    void _M_construct(size_type __n, _CharT __c) {
      if (__n > size_type(_S_local_capacity)) {
        _M_data(_M_create(__n, size_type(0)));
        _M_capacity(__n);
      }

      if (__n)
        this->_S_assign(_M_data(), __n, __c);

      _M_set_length(__n);
    }
```

#### \_M\_create

``` cpp
  // 1. 创建的最大的大小不能超过max_size()
  // 2. 指数倍地增长，即new_capacity=2*old_capacity。但是不能超过max_size()。
  // 为什么是指数倍增长呢？因为这是用“摊还（直译是分期偿还，amortize）”的策略来满足分配必须是线性的要求。
  _M_create(size_type & __capacity, size_type __old_capacity) {
    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 83.  String::npos vs. string::max_size()
    if (__capacity > max_size())
      std::__throw_length_error(__N("basic_string::_M_create"));

    // The below implements an exponential growth policy, necessary to
    // meet amortized linear time requirements of the library: see
    // http://gcc.gnu.org/ml/libstdc++/2001-07/msg00085.html.
    if (__capacity > __old_capacity && __capacity < 2 * __old_capacity) {
      __capacity = 2 * __old_capacity;
      // Never allocate a string bigger than max_size.
      if (__capacity > max_size())
        __capacity = max_size();
    }

    // NB: Need an array of char_type[__capacity], plus a terminating
    // null char_type() element.
    return _Alloc_traits::allocate(_M_get_allocator(), __capacity + 1);
  }
```

### 赋值相关

``` cpp
    // 下述的3个函数，都有一个共同的特点。当__n==1的时候，直接用trais_type::assign。
    // When __n = 1 way faster than the general multichar
    // traits_type::copy/move/assign.
    static void _S_copy(_CharT *__d, const _CharT *__s, size_type __n) {
      if (__n == 1)
        traits_type::assign(*__d, *__s);
      else
        traits_type::copy(__d, __s, __n);
    }

    static void _S_move(_CharT *__d, const _CharT *__s, size_type __n) {
      if (__n == 1)
        traits_type::assign(*__d, *__s);
      else
        traits_type::move(__d, __s, __n);
    }

    static void _S_assign(_CharT *__d, size_type __n, _CharT __c) {
      if (__n == 1)
        traits_type::assign(*__d, __c);
      else
        traits_type::assign(__d, __n, __c);
    }

    // _S_copy_chars和上述3个函数的区别是，上述3个函数有起始位置与长度作为参数，
    // 下面的函数是用的其实位置和终止位置作为参数。
    // _S_copy_chars is a separate template to permit specialization
    // to optimize for the common case of pointers as iterators.
    template <class _Iterator>
    static void _S_copy_chars(_CharT *__p, _Iterator __k1, _Iterator __k2) {
      for (; __k1 != __k2; ++__k1, (void)++__p)
        traits_type::assign(*__p, *__k1); // These types are off.
    }

    static void _S_copy_chars(_CharT *__p, iterator __k1,
                              iterator __k2) _GLIBCXX_NOEXCEPT {
      _S_copy_chars(__p, __k1.base(), __k2.base());
    }

    static void _S_copy_chars(_CharT *__p, const_iterator __k1,
                              const_iterator __k2) _GLIBCXX_NOEXCEPT {
      _S_copy_chars(__p, __k1.base(), __k2.base());
    }

    static void _S_copy_chars(_CharT *__p, _CharT *__k1,
                              _CharT *__k2) _GLIBCXX_NOEXCEPT {
      _S_copy(__p, __k1, __k2 - __k1);
    }

    static void _S_copy_chars(_CharT *__p, const _CharT *__k1,
                              const _CharT *__k2) _GLIBCXX_NOEXCEPT {
      _S_copy(__p, __k1, __k2 - __k1);
    }

    // 首先要分配空间，然后调用_S_copy。
    _M_assign(const basic_string &__str) {
      if (this != &__str) {
        const size_type __rsize = __str.length();
        const size_type __capacity = capacity();

        if (__rsize > __capacity) {
          size_type __new_capacity = __rsize;
          pointer __tmp = _M_create(__new_capacity, __capacity);
          _M_dispose();
          _M_data(__tmp);
          _M_capacity(__new_capacity);
        }

        if (__rsize)
          this->_S_copy(_M_data(), __str._M_data(), __rsize);

        _M_set_length(__rsize);
      }
    }

    // 参数解释：
    //    __pos：目的字符串（即本字符串）的起始位置，__len1：目的字符串的长度。
    //    __s：源字符串起始位置，__len2：源字符串的长度。
    // 作用：将源字符串的内容复制到目的字符串。
    _M_replace(size_type __pos, size_type __len1, const _CharT *__s,
              const size_type __len2) {
      _M_check_length(__len1, __len2, "basic_string::_M_replace");

      const size_type __old_size = this->size();
      // __len2-__len1是新旧字符串的差值。
      const size_type __new_size = __old_size + __len2 - __len1;

      if (__new_size <= this->capacity()) {
        pointer __p = this->_M_data() + __pos;

        // 现在的字符串长度为__len1，完成操作之后空间会变为__len2。
        // __len1这个位置之后的长度就是__how_much。
        const size_type __how_much = __old_size - __pos - __len1;
        if (_M_disjunct(__s)) { // 当__s和当前字符串不重叠的时候。
          if (__how_much && __len1 != __len2)
            // 将__how_much这个点之后的数据向后移，为后续的复制腾出来位置。
            this->_S_move(__p + __len2, __p + __len1, __how_much);
          if (__len2)
            this->_S_copy(__p, __s, __len2);
        } else {
          // Work in-place.
          if (__len2 && __len2 <= __len1)
            this->_S_move(__p, __s, __len2);
          if (__how_much && __len1 != __len2)
            this->_S_move(__p + __len2, __p + __len1, __how_much);
          if (__len2 > __len1) {
            if (__s + __len2 <= __p + __len1)
              this->_S_move(__p, __s, __len2);
            else if (__s >= __p + __len1) {
              // Hint to middle end that __p and __s overlap
              // (PR 98465).
              const size_type __poff = (__s - __p) + (__len2 - __len1);
              this->_S_copy(__p, __p + __poff, __len2);
            } else {
              const size_type __nleft = (__p + __len1) - __s;
              this->_S_move(__p, __s, __nleft);
              this->_S_copy(__p + __nleft, __p + __len2, __len2 - __nleft);
            }
          }
        }
      } else
        // 当new_size大于当前的capacity时，需要增加内存，因此有不同的处理流程。
        this->_M_mutate(__pos, __len1, __s, __len2);

      this->_M_set_length(__new_size);
      return *this;
    }

    // 作用：将__n2个字符__c，复制到本字符串的__pos1位置，本字符串从__pos1开始的长度为__n1。
    // _M_replace_aux是_M_replace的简化版本，因为它的源字符串只是__n2个__c的重复。
    basic_string &_M_replace_aux(size_type __pos1, size_type __n1, size_type __n2, _CharT __c) {
      _M_check_length(__n1, __n2, "basic_string::_M_replace_aux");

      const size_type __old_size = this->size();
      const size_type __new_size = __old_size + __n2 - __n1;

      if (__new_size <= this->capacity()) {
        pointer __p = this->_M_data() + __pos1;

        const size_type __how_much = __old_size - __pos1 - __n1;
        if (__how_much && __n1 != __n2)
          this->_S_move(__p + __n2, __p + __n1, __how_much);
      } else
        this->_M_mutate(__pos1, __n1, 0, __n2);

      if (__n2)
        this->_S_assign(this->_M_data() + __pos1, __n2, __c);

      this->_M_set_length(__new_size);
      return *this;
    }
```

``` cpp
    basic_string &assign(const basic_string &__str) {
#if __cplusplus >= 201103L
      if (_Alloc_traits::_S_propagate_on_copy_assign()) {
        if (!_Alloc_traits::_S_always_equal() && !_M_is_local() &&
            _M_get_allocator() != __str._M_get_allocator()) {
          // Propagating allocator cannot free existing storage so must
          // deallocate it before replacing current allocator.
          if (__str.size() <= _S_local_capacity) {
            _M_destroy(_M_allocated_capacity);
            _M_data(_M_local_data());
            _M_set_length(0);
          } else {
            const auto __len = __str.size();
            auto __alloc = __str._M_get_allocator();
            // If this allocation throws there are no effects:
            auto __ptr = _Alloc_traits::allocate(__alloc, __len + 1);
            _M_destroy(_M_allocated_capacity);
            _M_data(__ptr);
            _M_capacity(__len);
            _M_set_length(__len);
          }
        }
        std::__alloc_on_copy(_M_get_allocator(), __str._M_get_allocator());
      }
#endif
      this->_M_assign(__str);
      return *this;
    }

    // 这是和移动赋值函数不一样的操作内容。
    basic_string &
    assign(basic_string &&__str) noexcept(_Alloc_traits::_S_nothrow_move()) {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2063. Contradictory requirements for string move assignment
      return *this = std::move(__str);
    }
    basic_string &assign(const basic_string &__str, size_type __pos,
                         size_type __n = npos) {
      return _M_replace(size_type(0), this->size(),
                        __str._M_data() +
                            __str._M_check(__pos, "basic_string::assign"),
                        __str._M_limit(__pos, __n));
    }
    basic_string &assign(const _CharT *__s, size_type __n) {
      __glibcxx_requires_string_len(__s, __n);
      return _M_replace(size_type(0), this->size(), __s, __n);
    }
    basic_string &assign(const _CharT *__s) {
      __glibcxx_requires_string(__s);
      return _M_replace(size_type(0), this->size(), __s,
                        traits_type::length(__s));
    }
    basic_string &assign(size_type __n, _CharT __c) {
      return _M_replace_aux(size_type(0), this->size(), __n, __c);
    }
#if __cplusplus >= 201103L
    template <class _InputIterator,
              typename = std::_RequireInputIter<_InputIterator>>
#else
    template <class _InputIterator>
#endif
    basic_string &assign(_InputIterator __first, _InputIterator __last) {
      return this->replace(begin(), end(), __first, __last);
    }
    basic_string &assign(initializer_list<_CharT> __l) {
      return this->assign(__l.begin(), __l.size());
    }

    // string_view相关的assign
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> assign(const _Tp &__svt) {
      __sv_type __sv = __svt;
      return this->assign(__sv.data(), __sv.size());
    }
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> assign(const _Tp &__svt, size_type __pos,
                                       size_type __n = npos) {
      __sv_type __sv = __svt;
      return _M_replace(size_type(0), this->size(),
                        __sv.data() + std::__sv_check(__sv.size(), __pos,
                                                      "basic_string::assign"),
                        std::__sv_limit(__sv.size(), __pos, __n));
    }
```

``` cpp
    basic_string &replace(size_type __pos, size_type __n,
                          const basic_string &__str) {
      return this->replace(__pos, __n, __str._M_data(), __str.size());
    }
    basic_string &replace(size_type __pos1, size_type __n1,
                          const basic_string &__str, size_type __pos2,
                          size_type __n2 = npos) {
      return this->replace(__pos1, __n1,
                           __str._M_data() +
                               __str._M_check(__pos2, "basic_string::replace"),
                           __str._M_limit(__pos2, __n2));
    }
    basic_string &replace(size_type __pos, size_type __n1, const _CharT *__s,
                          size_type __n2) {
      __glibcxx_requires_string_len(__s, __n2);
      return _M_replace(_M_check(__pos, "basic_string::replace"),
                        _M_limit(__pos, __n1), __s, __n2);
    }
    basic_string &replace(size_type __pos, size_type __n1, const _CharT *__s) {
      __glibcxx_requires_string(__s);
      return this->replace(__pos, __n1, __s, traits_type::length(__s));
    }
    basic_string &replace(size_type __pos, size_type __n1, size_type __n2,
                          _CharT __c) {
      return _M_replace_aux(_M_check(__pos, "basic_string::replace"),
                            _M_limit(__pos, __n1), __n2, __c);
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          const basic_string &__str) {
      return this->replace(__i1, __i2, __str._M_data(), __str.size());
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          const _CharT *__s, size_type __n) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      return this->replace(__i1 - begin(), __i2 - __i1, __s, __n);
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          const _CharT *__s) {
      __glibcxx_requires_string(__s);
      return this->replace(__i1, __i2, __s, traits_type::length(__s));
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          size_type __n, _CharT __c) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      return _M_replace_aux(__i1 - begin(), __i2 - __i1, __n, __c);
    }
    template <class _InputIterator,
              typename = std::_RequireInputIter<_InputIterator>>
    basic_string &replace(const_iterator __i1, const_iterator __i2,
                          _InputIterator __k1, _InputIterator __k2) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      __glibcxx_requires_valid_range(__k1, __k2);
      return this->_M_replace_dispatch(__i1, __i2, __k1, __k2,
                                       std::__false_type());
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          _CharT *__k1, _CharT *__k2) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      __glibcxx_requires_valid_range(__k1, __k2);
      return this->replace(__i1 - begin(), __i2 - __i1, __k1, __k2 - __k1);
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          const _CharT *__k1, const _CharT *__k2) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      __glibcxx_requires_valid_range(__k1, __k2);
      return this->replace(__i1 - begin(), __i2 - __i1, __k1, __k2 - __k1);
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          iterator __k1, iterator __k2) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      __glibcxx_requires_valid_range(__k1, __k2);
      return this->replace(__i1 - begin(), __i2 - __i1, __k1.base(),
                           __k2 - __k1);
    }
    basic_string &replace(__const_iterator __i1, __const_iterator __i2,
                          const_iterator __k1, const_iterator __k2) {
      _GLIBCXX_DEBUG_PEDASSERT(begin() <= __i1 && __i1 <= __i2 &&
                               __i2 <= end());
      __glibcxx_requires_valid_range(__k1, __k2);
      return this->replace(__i1 - begin(), __i2 - __i1, __k1.base(),
                           __k2 - __k1);
    }
    basic_string &replace(const_iterator __i1, const_iterator __i2,
                          initializer_list<_CharT> __l) {
      return this->replace(__i1, __i2, __l.begin(), __l.size());
    }
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> replace(size_type __pos, size_type __n,
                                        const _Tp &__svt) {
      __sv_type __sv = __svt;
      return this->replace(__pos, __n, __sv.data(), __sv.size());
    }
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> replace(size_type __pos1, size_type __n1,
                                        const _Tp &__svt, size_type __pos2,
                                        size_type __n2 = npos) {
      __sv_type __sv = __svt;
      return this->replace(
          __pos1, __n1,
          __sv.data() +
              std::__sv_check(__sv.size(), __pos2, "basic_string::replace"),
          std::__sv_limit(__sv.size(), __pos2, __n2));
    }
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> replace(const_iterator __i1,
                                        const_iterator __i2, const _Tp &__svt) {
      __sv_type __sv = __svt;
      return this->replace(__i1 - begin(), __i2 - __i1, __sv);
    }

    template <class _Integer>
    basic_string &_M_replace_dispatch(const_iterator __i1, const_iterator __i2,
                                      _Integer __n, _Integer __val,
                                      __true_type) {
      return _M_replace_aux(__i1 - begin(), __i2 - __i1, __n, __val);
    }

    template <class _InputIterator>
    basic_string &_M_replace_dispatch(const_iterator __i1, const_iterator __i2,
                                      _InputIterator __k1, _InputIterator __k2,
                                      __false_type) {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2788. unintentionally require a default constructible allocator
      const basic_string __s(__k1, __k2, this->get_allocator());
      const size_type __n1 = __i2 - __i1;
      return _M_replace(__i1 - begin(), __n1, __s._M_data(), __s.size());
    }
                              
```

### copy 函数

``` cpp
    size_type copy(_CharT *__s, size_type __n, size_type __pos = 0) const  {
      _M_check(__pos, "basic_string::copy");
      __n = _M_limit(__pos, __n);
      __glibcxx_requires_string_len(__s, __n);
      if (__n)
        _S_copy(__s, _M_data() + __pos, __n);
      // 21.3.5.7 par 3: do not append null.  (good.)
      return __n;
    }
```

### swap 函数


本函数根据字符串是否使用本地buf和是否为空，有不同的操作。主要逻辑见下表：

| this is local? | s is local ? | this length > 0 | s length > 0 | operation |
| - | - | - | - | - |
| Y | Y | Y | Y | 创建中间变量tmp，通过拷贝互相交换值 |
| Y | Y | N | Y | 将s拷贝至this，然后设置s的长度为0 |
| Y | Y | Y | N | 将this拷贝至s，然后设置this的长度0 |
| Y | N | - | - | 将this的本地buf拷贝到s的本地buf，将this的数据指针指向s的，s的数据指针指向本地buf |
| N | Y | - | - | 将s的本地buf拷贝到s的本地buf，将s的数据指针指向this的， this的数据指针指向本地buf。 |
| N | N | - | - | 使用tmp作为中间过度变量，交换this和s数据指针。 |


``` cpp
  void swap(basic_string &__s) _GLIBCXX_NOEXCEPT {
    if (this == &__s)
      return;

    _Alloc_traits::_S_on_swap(_M_get_allocator(), __s._M_get_allocator());

    if (_M_is_local())
      if (__s._M_is_local()) {
        if (length() && __s.length()) {
          _CharT __tmp_data[_S_local_capacity + 1];
          traits_type::copy(__tmp_data, __s._M_local_buf,
                            _S_local_capacity + 1);
          traits_type::copy(__s._M_local_buf, _M_local_buf,
                            _S_local_capacity + 1);
          traits_type::copy(_M_local_buf, __tmp_data, _S_local_capacity + 1);
        } else if (__s.length()) {
          traits_type::copy(_M_local_buf, __s._M_local_buf,
                            _S_local_capacity + 1);
          _M_length(__s.length());
          __s._M_set_length(0);
          return;
        } else if (length()) {
          traits_type::copy(__s._M_local_buf, _M_local_buf,
                            _S_local_capacity + 1);
          __s._M_length(length());
          _M_set_length(0);
          return;
        }
      } else {
        const size_type __tmp_capacity = __s._M_allocated_capacity;
        traits_type::copy(__s._M_local_buf, _M_local_buf,
                          _S_local_capacity + 1);
        _M_data(__s._M_data());
        __s._M_data(__s._M_local_buf);
        _M_capacity(__tmp_capacity);
      }
    else {
      const size_type __tmp_capacity = _M_allocated_capacity;
      if (__s._M_is_local()) {
        traits_type::copy(_M_local_buf, __s._M_local_buf,
                          _S_local_capacity + 1);
        __s._M_data(_M_data());
        _M_data(_M_local_buf);
      } else {
        pointer __tmp_ptr = _M_data();
        _M_data(__s._M_data());
        __s._M_data(__tmp_ptr);
        _M_capacity(__s._M_allocated_capacity);
      }
      __s._M_capacity(__tmp_capacity);
    }

    const size_type __tmp_length = length();
    _M_length(__s.length());
    __s._M_length(__tmp_length);
  }
```

### compare 函数

`compare`函数的重载形式有很多，但区别不大，都是：
1. 比较最小共同长度的子字符串，此时调用`traits_type::compare`，如果不同的话，则返回1或-1.
2. 如果上述比较是相同的话，返回它们长度的差值，此时调用`_S_compare`。

``` cpp
    // 这个函数的作用就是返回__n1-__n2，并返回差值。
    static int _S_compare(size_type __n1, size_type __n2) _GLIBCXX_NOEXCEPT {
      const difference_type __d = difference_type(__n1 - __n2);

      if (__d > __gnu_cxx::__numeric_traits<int>::__max)
        return __gnu_cxx::__numeric_traits<int>::__max;
      else if (__d < __gnu_cxx::__numeric_traits<int>::__min)
        return __gnu_cxx::__numeric_traits<int>::__min;
      else
        return int(__d);
    }

    int compare(const basic_string &__str) const {
      const size_type __size = this->size();
      const size_type __osize = __str.size();
      const size_type __len = std::min(__size, __osize);

      int __r = traits_type::compare(_M_data(), __str.data(), __len);
      if (!__r)
        __r = _S_compare(__size, __osize);
      return __r;
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, int> compare(const _Tp &__svt) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      const size_type __size = this->size();
      const size_type __osize = __sv.size();
      const size_type __len = std::min(__size, __osize);

      int __r = traits_type::compare(_M_data(), __sv.data(), __len);
      if (!__r)
        __r = _S_compare(__size, __osize);
      return __r;
    }
    template <typename _Tp>
    _If_sv<_Tp, int> compare(size_type __pos, size_type __n,
                             const _Tp &__svt) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return __sv_type(*this).substr(__pos, __n).compare(__sv);
    }
    template <typename _Tp>
    _If_sv<_Tp, int> compare(size_type __pos1, size_type __n1, const _Tp &__svt,
                             size_type __pos2, size_type __n2 = npos) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return __sv_type(*this)
          .substr(__pos1, __n1)
          .compare(__sv.substr(__pos2, __n2));
    }
#endif // C++17

    int compare(size_type __pos, size_type __n,
                const basic_string &__str) const  {
      _M_check(__pos, "basic_string::compare");
      __n = _M_limit(__pos, __n);
      const size_type __osize = __str.size();
      const size_type __len = std::min(__n, __osize);
      int __r = traits_type::compare(_M_data() + __pos, __str.data(), __len);
      if (!__r)
        __r = _S_compare(__n, __osize);
      return __r;
    }
    int compare(size_type __pos1, size_type __n1, const basic_string &__str,
                size_type __pos2, size_type __n2 = npos) const  {
      _M_check(__pos1, "basic_string::compare");
      __str._M_check(__pos2, "basic_string::compare");
      __n1 = _M_limit(__pos1, __n1);
      __n2 = __str._M_limit(__pos2, __n2);
      const size_type __len = std::min(__n1, __n2);
      int __r =
          traits_type::compare(_M_data() + __pos1, __str.data() + __pos2, __len);
      if (!__r)
        __r = _S_compare(__n1, __n2);
      return __r;
    }
    int compare(const _CharT *__s) const _GLIBCXX_NOEXCEPT  {
      __glibcxx_requires_string(__s);
      const size_type __size = this->size();
      const size_type __osize = traits_type::length(__s);
      const size_type __len = std::min(__size, __osize);
      int __r = traits_type::compare(_M_data(), __s, __len);
      if (!__r)
        __r = _S_compare(__size, __osize);
      return __r;
    }
    int compare(size_type __pos, size_type __n1, const _CharT *__s) const  {
      __glibcxx_requires_string(__s);
      _M_check(__pos, "basic_string::compare");
      __n1 = _M_limit(__pos, __n1);
      const size_type __osize = traits_type::length(__s);
      const size_type __len = std::min(__n1, __osize);
      int __r = traits_type::compare(_M_data() + __pos, __s, __len);
      if (!__r)
        __r = _S_compare(__n1, __osize);
      return __r;
    }
    int compare(size_type __pos, size_type __n1, const _CharT *__s,
                size_type __n2) const  {
      __glibcxx_requires_string_len(__s, __n2);
      _M_check(__pos, "basic_string::compare");
      __n1 = _M_limit(__pos, __n1);
      const size_type __len = std::min(__n1, __n2);
      int __r = traits_type::compare(_M_data() + __pos, __s, __len);
      if (!__r)
        __r = _S_compare(__n1, __n2);
      return __r;
    }
```

### find 相关函数

``` cpp

    size_type find(const _CharT *__s, size_type __pos,
                   size_type __n) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string_len(__s, __n);
      const size_type __size = this->size();

      if (__n == 0)
        return __pos <= __size ? __pos : npos;
      if (__pos >= __size)
        return npos;

      const _CharT __elem0 = __s[0];
      const _CharT *const __data = data();
      const _CharT *__first = __data + __pos;
      const _CharT *const __last = __data + __size;
      size_type __len = __size - __pos;

      while (__len >= __n) {
        // Find the first occurrence of __elem0:
        // 先查找__s的第一个元素。
        __first = traits_type::find(__first, __len - __n + 1, __elem0);
        if (!__first)
          return npos;
        // Compare the full strings from the first occurrence of __elem0.
        // We already know that __first[0] == __s[0] but compare them again
        // anyway because __s is probably aligned, which helps memcmp.
        // 如果找到的话，以这个点为起点，比较长度为__n的本字符串是否和__s相同。
        if (traits_type::compare(__first, __s, __n) == 0)
          return __first - __data;
        __len = __last - ++__first;
      }
      return npos;
    }


    size_type find(const basic_string &__str,
                   size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      return this->find(__str.data(), __pos, __str.size());
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> find(const _Tp &__svt, size_type __pos = 0) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->find(__sv.data(), __pos, __sv.size());
    }
#endif // C++17
    size_type find(const _CharT *__s,
                   size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string(__s);
      return this->find(__s, __pos, traits_type::length(__s));
    }

    size_type find(_CharT __c, size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      size_type __ret = npos;
      const size_type __size = this->size();
      if (__pos < __size) {
        const _CharT *__data = _M_data();
        const size_type __n = __size - __pos;
        const _CharT *__p = traits_type::find(__data + __pos, __n, __c);
        if (__p)
          __ret = __p - __data;
      }
      return __ret;
    }
```

``` cpp

    size_type rfind(const basic_string &__str,
                    size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      return this->rfind(__str.data(), __pos, __str.size());
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> rfind(const _Tp &__svt, size_type __pos = npos) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->rfind(__sv.data(), __pos, __sv.size());
    }
#endif // C++17

    size_type rfind(const _CharT *__s, size_type __pos,
                    size_type __n) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string_len(__s, __n);
      const size_type __size = this->size();
      if (__n <= __size) {
        __pos = std::min(size_type(__size - __n), __pos);
        const _CharT *__data = _M_data();
        do {
          if (traits_type::compare(__data + __pos, __s, __n) == 0)
            return __pos;
        } while (__pos-- > 0);
      }
      return npos;
    }

    size_type rfind(const _CharT *__s, size_type __pos = npos) const {
      __glibcxx_requires_string(__s);
      return this->rfind(__s, __pos, traits_type::length(__s));
    }

    size_type rfind(_CharT __c, size_type __pos = npos) const _GLIBCXX_NOEXCEPT  {
      size_type __size = this->size();
      if (__size) {
        if (--__size > __pos)
          __size = __pos;
        for (++__size; __size-- > 0;)
          if (traits_type::eq(_M_data()[__size], __c))
            return __size;
      }
      return npos;
    }

```

``` cpp

    size_type find_first_of(const basic_string &__str,
                            size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      return this->find_first_of(__str.data(), __pos, __str.size());
    }
#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> find_first_of(const _Tp &__svt,
                                         size_type __pos = 0) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->find_first_of(__sv.data(), __pos, __sv.size());
    }
#endif // C++17
    size_type find_first_of(const _CharT *__s, size_type __pos,
                            size_type __n) const _GLIBCXX_NOEXCEPT  {
      __glibcxx_requires_string_len(__s, __n);
      for (; __n && __pos < this->size(); ++__pos) {
        const _CharT *__p = traits_type::find(__s, __n, _M_data()[__pos]);
        if (__p)
          return __pos;
      }
      return npos;
    }
    size_type find_first_of(const _CharT *__s,
                            size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string(__s);
      return this->find_first_of(__s, __pos, traits_type::length(__s));
    }
    size_type find_first_of(_CharT __c,
                            size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      return this->find(__c, __pos);
    }

    size_type find_last_of(const basic_string &__str,
                           size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      return this->find_last_of(__str.data(), __pos, __str.size());
    }
#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> find_last_of(const _Tp &__svt,
                                        size_type __pos = npos) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->find_last_of(__sv.data(), __pos, __sv.size());
    }
#endif // C++17
    size_type find_last_of(const _CharT *__s, size_type __pos,
                           size_type __n) const _GLIBCXX_NOEXCEPT  {
    __glibcxx_requires_string_len(__s, __n);
      size_type __size = this->size();
      if (__size && __n) {
        if (--__size > __pos)
          __size = __pos;
        do {
          if (traits_type::find(__s, __n, _M_data()[__size]))
            return __size;
        } while (__size-- != 0);
      }
      return npos;
    }
    size_type find_last_of(const _CharT *__s,
                           size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string(__s);
      return this->find_last_of(__s, __pos, traits_type::length(__s));
    }
    size_type find_last_of(_CharT __c,
                           size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      return this->rfind(__c, __pos);
    }
    size_type find_first_not_of(const basic_string &__str,
                                size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      return this->find_first_not_of(__str.data(), __pos, __str.size());
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> find_first_not_of(const _Tp &__svt,
                                             size_type __pos = 0) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->find_first_not_of(__sv.data(), __pos, __sv.size());
    }
#endif // C++17
    size_type find_first_not_of(const _CharT *__s, size_type __pos,
                                size_type __n) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string_len(__s, __n);
      for (; __pos < this->size(); ++__pos)
        if (!traits_type::find(__s, __n, _M_data()[__pos]))
          return __pos;
      return npos;
    }
    size_type find_first_not_of(const _CharT *__s,
                                size_type __pos = 0) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string(__s);
      return this->find_first_not_of(__s, __pos, traits_type::length(__s));
    }
    size_type find_first_not_of(_CharT __c,
                                size_type __pos = 0) const _GLIBCXX_NOEXCEPT  {
      for (; __pos < this->size(); ++__pos)
        if (!traits_type::eq(_M_data()[__pos], __c))
          return __pos;
      return npos;
    }

    size_type find_last_not_of(const basic_string &__str,
                               size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      return this->find_last_not_of(__str.data(), __pos, __str.size());
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, size_type> find_last_not_of(const _Tp &__svt,
                                            size_type __pos = npos) const
        noexcept(is_same<_Tp, __sv_type>::value) {
      __sv_type __sv = __svt;
      return this->find_last_not_of(__sv.data(), __pos, __sv.size());
    }
#endif // C++17
    size_type find_last_not_of(const _CharT *__s, size_type __pos,
                               size_type __n) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string_len(__s, __n);
      size_type __size = this->size();
      if (__size) {
        if (--__size > __pos)
          __size = __pos;
        do {
          if (!traits_type::find(__s, __n, _M_data()[__size]))
            return __size;
        } while (__size--);
      }
      return npos;
    }
    size_type find_last_not_of(const _CharT *__s,
                               size_type __pos = npos) const _GLIBCXX_NOEXCEPT {
      __glibcxx_requires_string(__s);
      return this->find_last_not_of(__s, __pos, traits_type::length(__s));
    }
    size_type find_last_not_of(_CharT __c,
                               size_type __pos = npos) const _GLIBCXX_NOEXCEPT T {
      size_type __size = this->size();
      if (__size) {
        if (--__size > __pos)
          __size = __pos;
        do {
          if (!traits_type::eq(_M_data()[__size], __c))
            return __size;
        } while (__size--);
      }
      return npos;
    }
```

### starts\_with & ends\_with 函数

``` cpp
    bool starts_with(basic_string_view<_CharT, _Traits> __x) const noexcept {
      return __sv_type(this->data(), this->size()).starts_with(__x);
    }

    bool starts_with(_CharT __x) const noexcept {
      return __sv_type(this->data(), this->size()).starts_with(__x);
    }

    bool starts_with(const _CharT *__x) const noexcept {
      return __sv_type(this->data(), this->size()).starts_with(__x);
    }

    bool ends_with(basic_string_view<_CharT, _Traits> __x) const noexcept {
      return __sv_type(this->data(), this->size()).ends_with(__x);
    }

    bool ends_with(_CharT __x) const noexcept {
      return __sv_type(this->data(), this->size()).ends_with(__x);
    }

    bool ends_with(const _CharT *__x) const noexcept {
      return __sv_type(this->data(), this->size()).ends_with(__x);
    }
```

### substr函数

这个函数调用了`basic_string`的构造函数。

``` cpp
    basic_string substr(size_type __pos = 0, size_type __n = npos) const {
      return basic_string(*this, _M_check(__pos, "basic_string::substr"), __n);
    }
```

### operator=

``` cpp
    basic_string &operator=(const basic_string &__str) {
      return this->assign(__str);
    }
    basic_string &operator=(const _CharT *__s) { return this->assign(__s); }
    basic_string &operator=(_CharT __c) {
      this->assign(1, __c);
      return *this;
    }
    basic_string &
    operator=(basic_string &&__str) noexcept(_Alloc_traits::_S_nothrow_move()) {
      if (!_M_is_local() && _Alloc_traits::_S_propagate_on_move_assign() &&
          !_Alloc_traits::_S_always_equal() &&
          _M_get_allocator() != __str._M_get_allocator()) {
        // Destroy existing storage before replacing allocator.
        _M_destroy(_M_allocated_capacity);
        _M_data(_M_local_data());
        _M_set_length(0);
      }
      // Replace allocator if POCMA is true.
      std::__alloc_on_move(_M_get_allocator(), __str._M_get_allocator());

      // 当__str使用的本地buffer时，直接拷贝。
      if (__str._M_is_local()) {
        // We've always got room for a short string, just copy it
        // (unless this is a self-move, because that would violate the
        // char_traits::copy precondition that the ranges don't overlap).
        if (__builtin_expect(std::__addressof(__str) != this, true)) {
          if (__str.size())
            this->_S_copy(_M_data(), __str._M_data(), __str.size());
          _M_set_length(__str.size());
        }
      } else if (_Alloc_traits::_S_propagate_on_move_assign() ||
                 _Alloc_traits::_S_always_equal() ||
                 _M_get_allocator() == __str._M_get_allocator()) {
        // TODO: 看下allocator traits相关内容
        // Just move the allocated pointer, our allocator can free it.
        pointer __data = nullptr;
        size_type __capacity;
        if (!_M_is_local()) {
          if (_Alloc_traits::_S_always_equal()) {
            // __str can reuse our existing storage.
            __data = _M_data();
            __capacity = _M_allocated_capacity;
          } else // __str can't use it, so free it.
            _M_destroy(_M_allocated_capacity);
        }

        _M_data(__str._M_data());
        _M_length(__str.length());
        _M_capacity(__str._M_allocated_capacity);
        if (__data) {
          __str._M_data(__data);
          __str._M_capacity(__capacity);
        } else
          __str._M_data(__str._M_local_buf);
      } else // Need to do a deep copy
        assign(__str);
      __str.clear();
      return *this;
    }

    basic_string &operator=(initializer_list<_CharT> __l) {
      this->assign(__l.begin(), __l.size());
      return *this;
    }
#endif // C++11
    // string_view相关
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> operator=(const _Tp &__svt) {
      return this->assign(__svt);
    }
```

### iterator举例

可以看到string的iterator是从指针中构造的。

``` cpp
    iterator begin() _GLIBCXX_NOEXCEPT { return iterator(_M_data()); }
```

### operator\[\] & at 函数

``` cpp
    const_reference operator[](size_type __pos) const _GLIBCXX_NOEXCEPT {
      __glibcxx_assert(__pos <= size());
      return _M_data()[__pos];
    }
    reference operator[](size_type __pos) {
      // Allow pos == size() both in C++98 mode, as v3 extension,
      // and in C++11 mode.
      __glibcxx_assert(__pos <= size());
      // In pedantic mode be strict in C++98 mode.
      _GLIBCXX_DEBUG_PEDASSERT(__cplusplus >= 201103L || __pos < size());
      return _M_data()[__pos];
    }
    const_reference at(size_type __n) const {
      if (__n >= this->size())
        __throw_out_of_range_fmt(__N("basic_string::at: __n "
                                     "(which is %zu) >= this->size() "
                                     "(which is %zu)"),
                                 __n, this->size());
      return _M_data()[__n];
    }
    reference at(size_type __n) {
      if (__n >= size())
        __throw_out_of_range_fmt(__N("basic_string::at: __n "
                                     "(which is %zu) >= this->size() "
                                     "(which is %zu)"),
                                 __n, this->size());
      return _M_data()[__n];
    }
```

### front & back 函数

``` cpp
    reference front() noexcept {
      __glibcxx_assert(!empty());
      return operator[](0);
    }

    const_reference front() const noexcept {
      __glibcxx_assert(!empty());
      return operator[](0);
    }

    reference back() noexcept {
      __glibcxx_assert(!empty());
      return operator[](this->size() - 1);
    }

    const_reference back() const noexcept {
      __glibcxx_assert(!empty());
      return operator[](this->size() - 1);
    }
```

### operator+=

``` cpp
    // TODO: push_back 和 append 的区别。
    basic_string &operator+=(const basic_string &__str) {
      return this->append(__str);
    }

    basic_string &operator+=(const _CharT *__s) { return this->append(__s); }

    basic_string &operator+=(_CharT __c) {
      this->push_back(__c);
      return *this;
    }

#if __cplusplus >= 201103L
    basic_string &operator+=(initializer_list<_CharT> __l) {
      return this->append(__l.begin(), __l.size());
    }
#endif // C++11

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> operator+=(const _Tp &__svt) {
      return this->append(__svt);
    }
#endif
```

### append & push_back 函数

``` cpp
    // 以下四个append函数的重载形式，都是调用的_M_append来实现的。
    basic_string &append(const basic_string &__str) {
      return _M_append(__str._M_data(), __str.size());
    }
    basic_string &append(const basic_string &__str, size_type __pos,
                         size_type __n = npos) {
      return _M_append(__str._M_data() +
                           __str._M_check(__pos, "basic_string::append"),
                       __str._M_limit(__pos, __n));
    }
    basic_string &append(const _CharT *__s, size_type __n) {
      __glibcxx_requires_string_len(__s, __n);
      _M_check_length(size_type(0), __n, "basic_string::append");
      return _M_append(__s, __n);
    }
    basic_string &append(const _CharT *__s) {
      __glibcxx_requires_string(__s);
      const size_type __n = traits_type::length(__s);
      _M_check_length(size_type(0), __n, "basic_string::append");
      return _M_append(__s, __n);
    }

    // 这个append函数的重载形式，将__n个字符__c append到string的最后。
    // 调用了_M_replace_aux函数实现。
    basic_string &append(size_type __n, _CharT __c) {
      return _M_replace_aux(this->size(), size_type(0), __n, __c);
    }

#if __cplusplus >= 201103L
    basic_string &append(initializer_list<_CharT> __l) {
      return this->append(__l.begin(), __l.size());
    }
#endif // C++11

#if __cplusplus >= 201103L
    template <class _InputIterator,
              typename = std::_RequireInputIter<_InputIterator>>
#else
    template <class _InputIterator>
#endif
    basic_string &append(_InputIterator __first, _InputIterator __last) {
      return this->replace(end(), end(), __first, __last);
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> append(const _Tp &__svt) {
      __sv_type __sv = __svt;
      return this->append(__sv.data(), __sv.size());
    }

    template <typename _Tp>
    _If_sv<_Tp, basic_string &> append(const _Tp &__svt, size_type __pos,
                                       size_type __n = npos) {
      __sv_type __sv = __svt;
      return _M_append(__sv.data() + std::__sv_check(__sv.size(), __pos,
                                                     "basic_string::append"),
                       std::__sv_limit(__sv.size(), __pos, __n));
    }
#endif // C++17

    // push_back函数将单个字符__c放到字符串的最后。
    // 逻辑比较清晰：__size + 1 > capacity的时候，需要进行内存的扩充。
    // 最后，将字符__c赋值给这个字符串的倒数第二位。（如前所述，倒数第一
    // 位是\0）。
    void push_back(_CharT __c) {
      const size_type __size = this->size();
      if (__size + 1 > this->capacity())
        this->_M_mutate(__size, size_type(0), 0, size_type(1));
      traits_type::assign(this->_M_data()[__size], __c);
      this->_M_set_length(__size + 1);
    }

    // 逻辑比较清晰了，当要append的长度小于capacity的时候，直接赋值过去。
    // 否则，需要重新创建内存，再拷贝。
    basic_string &_M_append(const _CharT *__s, size_type __n) {
      const size_type __len = __n + this->size();

      if (__len <= this->capacity()) {
        if (__n)
          this->_S_copy(this->_M_data() + this->size(), __s, __n);
      } else
        this->_M_mutate(this->size(), size_type(0), __s, __n);

      this->_M_set_length(__len);
      return *this;
    }
```

### insert函数

``` cpp
    // 大部分insert函数的实现，都是对replace函数的调用。
    // 但在insert单个字符和多个重复字符的时候，直接调用了_M_replace_aux。
#if __cplusplus >= 201103L
    iterator insert(const_iterator __p, size_type __n, _CharT __c) {
      _GLIBCXX_DEBUG_PEDASSERT(__p >= begin() && __p <= end());
      const size_type __pos = __p - begin();
      this->replace(__p, __p, __n, __c);
      return iterator(this->_M_data() + __pos);
    }
#else
    void insert(iterator __p, size_type __n, _CharT __c) {
      this->replace(__p, __p, __n, __c);
    }
#endif

#if __cplusplus >= 201103L
    template <class _InputIterator,
              typename = std::_RequireInputIter<_InputIterator>>
    iterator insert(const_iterator __p, _InputIterator __beg,
                    _InputIterator __end) {
      _GLIBCXX_DEBUG_PEDASSERT(__p >= begin() && __p <= end());
      const size_type __pos = __p - begin();
      this->replace(__p, __p, __beg, __end);
      return iterator(this->_M_data() + __pos);
    }
#else
    template <class _InputIterator>
    void insert(iterator __p, _InputIterator __beg, _InputIterator __end) {
      this->replace(__p, __p, __beg, __end);
    }
#endif

#if __cplusplus >= 201103L
    iterator insert(const_iterator __p, initializer_list<_CharT> __l) {
      return this->insert(__p, __l.begin(), __l.end());
    }

#ifdef _GLIBCXX_DEFINING_STRING_INSTANTIATIONS
    // See PR libstdc++/83328
    void insert(iterator __p, initializer_list<_CharT> __l) {
      _GLIBCXX_DEBUG_PEDASSERT(__p >= begin() && __p <= end());
      this->insert(__p - begin(), __l.begin(), __l.size());
    }
#endif
#endif // C++11

    basic_string &insert(size_type __pos1, const basic_string &__str) {
      return this->replace(__pos1, size_type(0), __str._M_data(), __str.size());
    }
    basic_string &insert(size_type __pos1, const basic_string &__str,
                         size_type __pos2, size_type __n = npos) {
      return this->replace(__pos1, size_type(0),
                           __str._M_data() +
                               __str._M_check(__pos2, "basic_string::insert"),
                           __str._M_limit(__pos2, __n));
    }
    basic_string &insert(size_type __pos, const _CharT *__s, size_type __n) {
      return this->replace(__pos, size_type(0), __s, __n);
    }
    basic_string &insert(size_type __pos, const _CharT *__s) {
      __glibcxx_requires_string(__s);
      return this->replace(__pos, size_type(0), __s, traits_type::length(__s));
    }


    basic_string &insert(size_type __pos, size_type __n, _CharT __c) {
      return _M_replace_aux(_M_check(__pos, "basic_string::insert"),
                            size_type(0), __n, __c);
    }
    iterator insert(__const_iterator __p, _CharT __c) {
      _GLIBCXX_DEBUG_PEDASSERT(__p >= begin() && __p <= end());
      const size_type __pos = __p - begin();
      _M_replace_aux(__pos, size_type(0), size_type(1), __c);
      return iterator(_M_data() + __pos);
    }

#if __cplusplus >= 201703L
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> insert(size_type __pos, const _Tp &__svt) {
      __sv_type __sv = __svt;
      return this->insert(__pos, __sv.data(), __sv.size());
    }
    template <typename _Tp>
    _If_sv<_Tp, basic_string &> insert(size_type __pos1, const _Tp &__svt,
                                       size_type __pos2, size_type __n = npos) {
      __sv_type __sv = __svt;
      return this->replace(
          __pos1, size_type(0),
          __sv.data() +
              std::__sv_check(__sv.size(), __pos2, "basic_string::insert"),
          std::__sv_limit(__sv.size(), __pos2, __n));
    }
#endif // C++17

```


### erase函数

所有的`erase`函数都调用了`_M_erase`来完成主要工作。

``` cpp
    basic_string &erase(size_type __pos = 0, size_type __n = npos) {
      _M_check(__pos, "basic_string::erase");
      if (__n == npos)
        this->_M_set_length(__pos);
      else if (__n != 0)
        this->_M_erase(__pos, _M_limit(__pos, __n));
      return *this;
    }
    iterator erase(__const_iterator __position) {
      _GLIBCXX_DEBUG_PEDASSERT(__position >= begin() && __position < end());
      const size_type __pos = __position - begin();
      this->_M_erase(__pos, size_type(1));
      return iterator(_M_data() + __pos);
    }

    iterator erase(__const_iterator __first, __const_iterator __last) {
      _GLIBCXX_DEBUG_PEDASSERT(__first >= begin() && __first <= __last &&
                               __last <= end());
      const size_type __pos = __first - begin();
      if (__last == end())
        this->_M_set_length(__pos);
      else
        this->_M_erase(__pos, __last - __first);
      return iterator(this->_M_data() + __pos);
    }
    
    // 逻辑比较简单：
    // 1. 计算出来字符串中__pos+__n之后还有多少字符（即__how_much）。
    // 2. 将__how_much移动到__pos之后。
    // 3. 设置length。
    void _M_erase(size_type __pos, size_type __n) {
      const size_type __how_much = length() - __pos - __n;

      if (__how_much && __n)
        this->_S_move(_M_data() + __pos, _M_data() + __pos + __n, __how_much);

      _M_set_length(length() - __n);
    }

```


### 辅助类成员函数

``` cpp
    // 检查传入的__pos是否是大于此字符串的大小。
    size_type _M_check(size_type __pos, const char *__s) const {
      if (__pos > this->size())
        __throw_out_of_range_fmt(__N("%s: __pos (which is %zu) > "
                                     "this->size() (which is %zu)"),
                                 __s, __pos, this->size());
      return __pos;
    }

    // TODO：
    void _M_check_length(size_type __n1, size_type __n2,
                         const char *__s) const {
      if (this->max_size() - (this->size() - __n1) < __n2)
        __throw_length_error(__N(__s));
    }

    // 
    // NB: _M_limit doesn't check for a bad __pos value.
    size_type _M_limit(size_type __pos,
                       size_type __off) const _GLIBCXX_NOEXCEPT {
      const bool __testoff = __off < this->size() - __pos;
      return __testoff ? __off : this->size() - __pos;
    }

    // 判断传入__s参数是否和此字符串是否在内存上重合。
    // True if _Rep and source do not overlap.
    bool _M_disjunct(const _CharT *__s) const _GLIBCXX_NOEXCEPT {
      return (less<const _CharT *>()(__s, _M_data()) ||
              less<const _CharT *>()(_M_data() + this->size(), __s));
    }

    // 参数说明：
    //    __pos：目的字符串（本字符串）的起始位置，__len1：目的字符串的长度。
    //    __s：源字符串的起始位置，__len2：源字符串的长度。
    // 作用：将源字符串__s开始，长度为__len2的子串，复制到目的字符串从__pos开始的位置。
    _M_mutate(size_type __pos, size_type __len1, const _CharT *__s, size_type __len2) {
      // __how_much为本字符串__len1后面的长度（尾巴）。
      const size_type __how_much = length() - __pos - __len1;

      size_type __new_capacity = length() + __len2 - __len1;
      // 创建一个__new_capacity的新字符串。
      pointer __r = _M_create(__new_capacity, capacity());

      if (__pos)
        // 将本字符串__pos之前的位置复制到新字符串。
        this->_S_copy(__r, _M_data(), __pos);
      if (__s && __len2)
        // 赋值__s开始长度为__len2的子串复制到新字符串。
        this->_S_copy(__r + __pos, __s, __len2);
      if (__how_much)
        // 复制尾巴部分到新字符串。
        this->_S_copy(__r + __pos + __len2, _M_data() + __pos + __len1,
                      __how_much);

      _M_dispose();
      _M_data(__r);
      _M_capacity(__new_capacity);
    }
```

### pop_back函数

`pop_back`函数也调用了`_M_erase`函数，相当于erase掉最后一个字符。

``` cpp
    void pop_back() noexcept {
      __glibcxx_assert(!empty());
      _M_erase(size() - 1, 1);
    }
```

### 部分成员函数

``` cpp
    ///  Returns the size() of the largest possible %string.
    size_type max_size() const _GLIBCXX_NOEXCEPT {
      return (_Alloc_traits::max_size(_M_get_allocator()) - 1) / 2;
    }

    // 由此可以看出，std::string最后的一个字符是'\0'。
    void _M_set_length(size_type __n) {
      _M_length(__n);
      traits_type::assign(_M_data()[__n], _CharT());
    }

    void clear() _GLIBCXX_NOEXCEPT { _M_set_length(0); }

    // capacity函数区分是否使用了本地buf。
    size_type capacity() const _GLIBCXX_NOEXCEPT {
      return _M_is_local() ? size_type(_S_local_capacity)
                           : _M_allocated_capacity;
    }
```

### resize、reserve、shrink_to_fit

``` cpp
    // 当__size<__n时，即通过resize来扩容时，将新扩容的内存的数据设为__c。
    // 否则，直接设置string的长度为__n。
    void resize(size_type __n, _CharT __c)  {
      const size_type __size = this->size();
      if (__size < __n)
        this->append(__n - __size, __c);
      else if (__n < __size)
        this->_M_set_length(__n);
    }

    // 将默认的填充值是“\0”。
    void resize(size_type __n) { this->resize(__n, _CharT()); }

    // 当需要reserve的size小于等于当前的capacity时，不做任何处理。（这和
    // vector/unordered_map/unordered_set的reserve函数是不同的。）
    // 否则的话，创建一个新的内存，并将当前的数据拷贝到新的内存，
    // 并将数据指针指向新的内存。
    // 另外，还有一点注意的是，这里为什么不会判断是否是使用本地buf来决定是否分配新的空间呢？
    // 原因是：capacity最小的值就是本地buf的最大容量了。任何大于capacity的size都不会使用的是
    // 本地buf。
    void reserve(size_type __res) {
      const size_type __capacity = capacity();
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2968. Inconsistencies between basic_string reserve and
      // vector/unordered_map/unordered_set reserve functions
      // P0966 reserve should not shrink
      if (__res <= __capacity)
        return;

      pointer __tmp = _M_create(__res, __capacity);
      this->_S_copy(__tmp, _M_data(), length() + 1);
      _M_dispose();
      _M_data(__tmp);
      _M_capacity(__res);
    }
```

### 析构函数

``` cpp
    ~basic_string() {
      _M_dispose();
    }

    // 这个函数的内容就是当不使用local buffer时，调用`_Alloc_traits::deallocate()`函数。
    // 使用本地buffer的字符串则不需要这些。
    void _M_dispose() {
      if (!_M_is_local())
        _M_destroy(_M_allocated_capacity);
    }
    void _M_destroy(size_type __size) throw() {
      _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1);
    }
```

## basic\_string (!\_GLIBCXX\_USE\_CXX11\_ABI)

### 声明式

``` cpp
  template <typename _CharT, typename _Traits, typename _Alloc>
  class basic_string;
```

### \_Rep

`_Rep_base`结构体说明：
1. 字符串的实际上`_M_length + 1`的长度，因为后面必须有一个“\0”。
2. `_M_capacity >= _M_length`，分配内存的时候，总是分配`(_M_capacity + 1) * sizeof(_CharT)`的长度。
3. `_M_refcount`有三种状态：
   1. -1：泄露了。
   2. 0：一个引用。
   3. `n > 0`：意味着`n + 1`个引用。
4. 所有的值都为0的时候，意味着是一个空的字符串。

``` cpp
    struct _Rep_base {
      size_type _M_length;
      size_type _M_capacity;
      _Atomic_word _M_refcount;
    };
```
``` cpp
    struct _Rep : _Rep_base {
      // Types:
      typedef typename __gnu_cxx::__alloc_traits<_Alloc>::template rebind<
          char>::other _Raw_bytes_alloc;

      // (Public) Data members:

      // The maximum number of individual char_type elements of an
      // individual string is determined by _S_max_size. This is the
      // value that will be returned by max_size().  (Whereas npos
      // is the maximum number of bytes the allocator can allocate.)
      // If one was to divvy up the theoretical largest size string,
      // with a terminating character and m _CharT elements, it'd
      // look like this:
      // npos = sizeof(_Rep) + (m * sizeof(_CharT)) + sizeof(_CharT)
      // Solving for m:
      // m = ((npos - sizeof(_Rep))/sizeof(CharT)) - 1
      // In addition, this implementation quarters this amount.
      static const size_type _S_max_size;
      static const _CharT _S_terminal;

      // The following storage is init'd to 0 by the linker, resulting
      // (carefully) in an empty string with one reference.
      static size_type _S_empty_rep_storage[];

      static _Rep &_S_empty_rep() _GLIBCXX_NOEXCEPT {
        // NB: Mild hack to avoid strict-aliasing warnings.  Note that
        // _S_empty_rep_storage is never modified and the punning should
        // be reasonably safe in this case.
        void *__p = reinterpret_cast<void *>(&_S_empty_rep_storage);
        return *reinterpret_cast<_Rep *>(__p);
      }

      bool _M_is_leaked() const _GLIBCXX_NOEXCEPT {
#if defined(__GTHREADS)
        // _M_refcount is mutated concurrently by _M_refcopy/_M_dispose,
        // so we need to use an atomic load. However, _M_is_leaked
        // predicate does not change concurrently (i.e. the string is either
        // leaked or not), so a relaxed load is enough.
        return __atomic_load_n(&this->_M_refcount, __ATOMIC_RELAXED) < 0;
#else
        return this->_M_refcount < 0;
#endif
      }

      bool _M_is_shared() const _GLIBCXX_NOEXCEPT {
#if defined(__GTHREADS)
        // _M_refcount is mutated concurrently by _M_refcopy/_M_dispose,
        // so we need to use an atomic load. Another thread can drop last
        // but one reference concurrently with this check, so we need this
        // load to be acquire to synchronize with release fetch_and_add in
        // _M_dispose.
        return __atomic_load_n(&this->_M_refcount, __ATOMIC_ACQUIRE) > 0;
#else
        return this->_M_refcount > 0;
#endif
      }

      void _M_set_leaked() _GLIBCXX_NOEXCEPT { this->_M_refcount = -1; }

      void _M_set_sharable() _GLIBCXX_NOEXCEPT { this->_M_refcount = 0; }

      void _M_set_length_and_sharable(size_type __n) _GLIBCXX_NOEXCEPT {
#if _GLIBCXX_FULLY_DYNAMIC_STRING == 0
        if (__builtin_expect(this != &_S_empty_rep(), false))
#endif
        {
          this->_M_set_sharable(); // One reference.
          this->_M_length = __n;
          traits_type::assign(this->_M_refdata()[__n], _S_terminal);
          // grrr. (per 21.3.4)
          // You cannot leave those LWG people alone for a second.
        }
      }

      _CharT *_M_refdata() throw() {
        return reinterpret_cast<_CharT *>(this + 1);
      }

      _CharT *_M_grab(const _Alloc &__alloc1, const _Alloc &__alloc2) {
        return (!_M_is_leaked() && __alloc1 == __alloc2) ? _M_refcopy()
                                                         : _M_clone(__alloc1);
      }

      // Create & Destroy
      static _Rep *_S_create(size_type, size_type, const _Alloc &);

      void _M_dispose(const _Alloc &__a) _GLIBCXX_NOEXCEPT {
#if _GLIBCXX_FULLY_DYNAMIC_STRING == 0
        if (__builtin_expect(this != &_S_empty_rep(), false))
#endif
        {
          // Be race-detector-friendly.  For more info see bits/c++config.
          _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&this->_M_refcount);
          // Decrement of _M_refcount is acq_rel, because:
          // - all but last decrements need to release to synchronize with
          //   the last decrement that will delete the object.
          // - the last decrement needs to acquire to synchronize with
          //   all the previous decrements.
          // - last but one decrement needs to release to synchronize with
          //   the acquire load in _M_is_shared that will conclude that
          //   the object is not shared anymore.
          if (__gnu_cxx::__exchange_and_add_dispatch(&this->_M_refcount, -1) <=
              0) {
            _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&this->_M_refcount);
            _M_destroy(__a);
          }
        }
      } // XXX MT

      void _M_destroy(const _Alloc &) throw();

      _CharT *_M_refcopy() throw() {
#if _GLIBCXX_FULLY_DYNAMIC_STRING == 0
        if (__builtin_expect(this != &_S_empty_rep(), false))
#endif
          __gnu_cxx::__atomic_add_dispatch(&this->_M_refcount, 1);
        return _M_refdata();
      } // XXX MT

      _CharT *_M_clone(const _Alloc &, size_type __res = 0);
    };
```

### 核心数据

`_M_dataplus._M_p`指向实际的数据指针。

``` cpp
    struct _Alloc_hider : _Alloc {
      _Alloc_hider(_CharT *__dat, const _Alloc &__a) _GLIBCXX_NOEXCEPT
          : _Alloc(__a),
            _M_p(__dat) {}

      _CharT *_M_p; // The actual data.
    };
    mutable _Alloc_hider _M_dataplus;
```


## basic\_string\_view

### 定义

``` cpp
  template<typename _CharT, typename _Traits = std::char_traits<_CharT>>
    class basic_string_view
```

### 成员变量

``` cpp
      size_t	    _M_len;
      const _CharT* _M_str;
```

## char\_traits

### \_Char\_types

``` cpp
  /**
   *  @brief  Mapping from character type to associated types.
   *
   *  @note This is an implementation class for the generic version
   *  of char_traits.  It defines int_type, off_type, pos_type, and
   *  state_type.  By default these are unsigned long, streamoff,
   *  streampos, and mbstate_t.  Users who need a different set of
   *  types, but who don't need to change the definitions of any function
   *  defined in char_traits, can specialize __gnu_cxx::_Char_types
   *  while leaving __gnu_cxx::char_traits alone. */
  template<typename _CharT>
    struct _Char_types
    {
      typedef unsigned long   int_type;
      typedef std::streampos  pos_type;
      typedef std::streamoff  off_type;
      typedef std::mbstate_t  state_type;
    };
```

这是char\_traits的普通实现版本，它typedef了上面这几种type。



### char\_traits定义

``` cpp
  template<typename _CharT>
    struct char_traits
```

### 主要函数

- assign（单字符，赋值）、eq（单字符是否相等）、lt（单字符小于）
- compare：字符串比较，有固定的比较长度n；length：字符串长度；find：字符串中找字符
- copy：字符串复制，调用`std::copy`；assign：字符串赋值，调用`std::fill_n`
- to\_char\_type、to\_int\_type：将输入的int\_type、char\_type转换为相应的char\_type、int\_type。
- eq\_int\_type：将char\_type转换为int\_type，并比较是否相等。
- eof：将`_GLIBCXX_STDIO_EOF`转换为int\_type；not\_eof比较传入的int\_type是否不是eof。

### char\_traits对char的特化版本

- 这里的char\_type是char、int\_type是int，其他和基础版本一样。
- assign、eq和基础版本一样。
- lt（字符小于）：会将char类型强制转换为unsigned char再进行比较。
- compare：字符串比较，会判断输入参数是否是const的，如果是的话，会逐个比较；否则调用`__builtin_memcmp`。
- length：字符串长度，会判断输入参数是否是const的，如果是的话，调用`__gnu_cxx::char_traits<char_type>::length`；否则调用`__builtin_strlen`。
- find：会判断输入参数是否是const的，如果是的话，调用`__gnu_cxx::char_traits<char_type>::find`；否则调用`__builtin_memchr`。
- move：会判断是否出现在常量求值的场合，如果是的话，调用`__gnu_cxx::char_traits<char_type>::move`；否则调用`__builtin_memmove`。
- copy：会判断输是否出现在常量求值的场合，如果是的话，调用`__gnu_cxx::char_traits<char_type>::copy`；否则调用`__builtin_memcpy`。
- assign：会判断是否出现在常量求值的场合，如果是的话，调用`__gnu_cxx::char_traits<char_type>::assign`；否则调用`__builtin_memset`。
