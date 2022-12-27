# § iterator

## 1 iterator

``` cpp
  /**
   *  @brief  Common %iterator class.
   *
   *  This class does nothing but define nested typedefs.  %Iterator classes
   *  can inherit from this class to save some work.  The typedefs are then
   *  used in specializations and overloading.
   *
   *  In particular, there are no default implementations of requirements
   *  such as @c operator++ and the like.  (How could there be?)
  */
  template<typename _Category, typename _Tp, typename _Distance = ptrdiff_t,
           typename _Pointer = _Tp*, typename _Reference = _Tp&>
    struct iterator
    {
      /// One of the @link iterator_tags tag types@endlink.
      typedef _Category  iterator_category;
      /// The type "pointed to" by the iterator.
      typedef _Tp        value_type;
      /// Distance between iterators is represented as this type.
      typedef _Distance  difference_type;
      /// This type represents a pointer-to-value_type.
      typedef _Pointer   pointer;
      /// This type represents a reference-to-value_type.
      typedef _Reference reference;
    };
```

这个类利用传入的模板参数定义了一些typedef，并没有默认的一些实现。



## 2 iterator\_traits

``` cpp
  /**
   *  @brief  Traits class for iterators.
   *
   *  This class does nothing but define nested typedefs.  The general
   *  version simply @a forwards the nested typedefs from the Iterator
   *  argument.  Specialized versions for pointers and pointers-to-const
   *  provide tighter, more correct semantics.
  */
  template<typename _Iterator>
    struct iterator_traits;

  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // 2408. SFINAE-friendly common_type/iterator_traits is missing in C++14
  template<typename _Iterator, typename = __void_t<>>
    struct __iterator_traits { };

#if ! __cpp_lib_concepts

  template<typename _Iterator>
    struct __iterator_traits<_Iterator,
			     __void_t<typename _Iterator::iterator_category,
				      typename _Iterator::value_type,
				      typename _Iterator::difference_type,
				      typename _Iterator::pointer,
				      typename _Iterator::reference>>
    {
      typedef typename _Iterator::iterator_category iterator_category;
      typedef typename _Iterator::value_type        value_type;
      typedef typename _Iterator::difference_type   difference_type;
      typedef typename _Iterator::pointer           pointer;
      typedef typename _Iterator::reference         reference;
    };
#endif // ! concepts

  template<typename _Iterator>
    struct iterator_traits
    : public __iterator_traits<_Iterator> { };
```

iterator\_traits也是“转发（forward）”了模板参数的\_Iterator的typedef。



## 3 reverse\_iterator

**双向**和**随机**访问迭代器有相应的reverse迭代器，它用来**反向**地遍历数据结构。反向迭代器和相应的普通迭代器的关系是：`&*(reverse_iterator(i)) == &*(i - 1)`。这个设计是基于这样一个事实：一定有一个在数组之后的有效指针，而不一定有一个在数组之前的有效的指针。



#### 定义

``` cpp
template<typename _Iterator>
    class reverse_iterator
    : public iterator<typename iterator_traits<_Iterator>::iterator_category,
		      typename iterator_traits<_Iterator>::value_type,
		      typename iterator_traits<_Iterator>::difference_type,
		      typename iterator_traits<_Iterator>::pointer,
                      typename iterator_traits<_Iterator>::reference>
     template<typename _Iter>
	friend class reverse_iterator;
```

它的模板参数是一种迭代器，继承自iterator，并且和其他的迭代器作为模板参数的反向迭代器是friend的。



#### 唯一protected成员变量

``` cpp
      _Iterator current;
```



#### 构造、拷贝构造、赋值构造函数

``` cpp
      /**
       *  The default constructor value-initializes member @p current.
       *  If it is a pointer, that means it is zero-initialized.
      */
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 235 No specification of default ctor for reverse_iterator
      // 1012. reverse_iterator default ctor should value initialize
      _GLIBCXX17_CONSTEXPR
      reverse_iterator() : current() { }

      /**
       *  This %iterator will move in the opposite direction that @p x does.
      */
      explicit _GLIBCXX17_CONSTEXPR
      reverse_iterator(iterator_type __x) : current(__x) { }

      /**
       *  The copy constructor is normal.
      */
      _GLIBCXX17_CONSTEXPR
      reverse_iterator(const reverse_iterator& __x)
      : current(__x.current) { }

#if __cplusplus >= 201103L
      reverse_iterator& operator=(const reverse_iterator&) = default;
#endif

      /**
       *  A %reverse_iterator across other types can be copied if the
       *  underlying %iterator can be converted to the type of @c current.
      */
      template<typename _Iter>
	_GLIBCXX17_CONSTEXPR
        reverse_iterator(const reverse_iterator<_Iter>& __x)
	: current(__x.current) { }

#if __cplusplus >= 201103L
      template<typename _Iter>
	_GLIBCXX17_CONSTEXPR
	reverse_iterator&
	operator=(const reverse_iterator<_Iter>& __x)
	{
	  current = __x.current;
	  return *this;
	}
#endif
```

默认构造函数的内容就是进行默认初始化protected变量`_Iterator current`。拷贝、赋值构造函数的主要内容也是拷贝、赋值这个变量。



#### base 函数

``` cpp
      /**
       *  @return  @c current, the %iterator used for underlying work.
      */
      _GLIBCXX17_CONSTEXPR iterator_type
      base() const
      { return current; }
```

返回当前迭代器变量。



#### \_S\_to\_pointer 函数

``` cpp
      template<typename _Tp>
	static _GLIBCXX17_CONSTEXPR _Tp*
	_S_to_pointer(_Tp* __p)
        { return __p; }

      template<typename _Tp>
	static _GLIBCXX17_CONSTEXPR pointer
	_S_to_pointer(_Tp __t)
        { return __t.operator->(); }
    };
```

对于原生指针，直接返回；对于其他类型的指针，调用`operator->`。



#### operator*、->函数

``` cpp
      /**
       *  @return  A reference to the value at @c --current
       *
       *  This requires that @c --current is dereferenceable.
       *
       *  @warning This implementation requires that for an iterator of the
       *           underlying iterator type, @c x, a reference obtained by
       *           @c *x remains valid after @c x has been modified or
       *           destroyed. This is a bug: http://gcc.gnu.org/PR51823
      */
      _GLIBCXX17_CONSTEXPR reference
      operator*() const
      {
	_Iterator __tmp = current;
	return *--__tmp;
      }

      /**
       *  @return  A pointer to the value at @c --current
       *
       *  This requires that @c --current is dereferenceable.
      */
      _GLIBCXX17_CONSTEXPR pointer
      operator->() const
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// 1052. operator-> should also support smart pointers
	_Iterator __tmp = current;
	--__tmp;
	return _S_to_pointer(__tmp);
      }
```

`operator*`的目的就是将当前位置减一，然后返回。`operator->`的目的就是将当前位置减一。`current`这个值实际上不等于`operator*`取出来的值，它取出来的值实际上比`current`永远小1，即在前面一位。



#### operator ++、--、+、+=、-、-=、\[\]函数

``` cpp
      /**
       *  @return  @c *this
       *
       *  Decrements the underlying iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator&
      operator++()
      {
	--current;
	return *this;
      }

      /**
       *  @return  The original value of @c *this
       *
       *  Decrements the underlying iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator
      operator++(int)
      {
	reverse_iterator __tmp = *this;
	--current;
	return __tmp;
      }

      /**
       *  @return  @c *this
       *
       *  Increments the underlying iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator&
      operator--()
      {
	++current;
	return *this;
      }

      /**
       *  @return  A reverse_iterator with the previous value of @c *this
       *
       *  Increments the underlying iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator
      operator--(int)
      {
	reverse_iterator __tmp = *this;
	++current;
	return __tmp;
      }

      /**
       *  @return  A reverse_iterator that refers to @c current - @a __n
       *
       *  The underlying iterator must be a Random Access Iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator
      operator+(difference_type __n) const
      { return reverse_iterator(current - __n); }

      /**
       *  @return  *this
       *
       *  Moves the underlying iterator backwards @a __n steps.
       *  The underlying iterator must be a Random Access Iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator&
      operator+=(difference_type __n)
      {
	current -= __n;
	return *this;
      }

      /**
       *  @return  A reverse_iterator that refers to @c current - @a __n
       *
       *  The underlying iterator must be a Random Access Iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator
      operator-(difference_type __n) const
      { return reverse_iterator(current + __n); }

      /**
       *  @return  *this
       *
       *  Moves the underlying iterator forwards @a __n steps.
       *  The underlying iterator must be a Random Access Iterator.
      */
      _GLIBCXX17_CONSTEXPR reverse_iterator&
      operator-=(difference_type __n)
      {
	current += __n;
	return *this;
      }

      _GLIBCXX17_CONSTEXPR reference
      operator[](difference_type __n) const
      { return *(*this + __n); }

```



#### operator ==、!=、<、>、<=、>= 函数

``` cpp
  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator==(const reverse_iterator<_Iterator>& __x,
	       const reverse_iterator<_Iterator>& __y)
    { return __x.base() == __y.base(); }

  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator<(const reverse_iterator<_Iterator>& __x,
	      const reverse_iterator<_Iterator>& __y)
    { return __y.base() < __x.base(); }

  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator!=(const reverse_iterator<_Iterator>& __x,
	       const reverse_iterator<_Iterator>& __y)
    { return !(__x == __y); }

  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator>(const reverse_iterator<_Iterator>& __x,
	      const reverse_iterator<_Iterator>& __y)
    { return __y < __x; }

  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator<=(const reverse_iterator<_Iterator>& __x,
	       const reverse_iterator<_Iterator>& __y)
    { return !(__y < __x); }

  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR bool
    operator>=(const reverse_iterator<_Iterator>& __x,
	       const reverse_iterator<_Iterator>& __y)
    { return !(__x < __y); }

  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 280. Comparison of reverse_iterator to const reverse_iterator.

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator==(const reverse_iterator<_IteratorL>& __x,
	       const reverse_iterator<_IteratorR>& __y)
    { return __x.base() == __y.base(); }

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator<(const reverse_iterator<_IteratorL>& __x,
	      const reverse_iterator<_IteratorR>& __y)
    { return __x.base() > __y.base(); }

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator!=(const reverse_iterator<_IteratorL>& __x,
	       const reverse_iterator<_IteratorR>& __y)
    { return __x.base() != __y.base(); }

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator>(const reverse_iterator<_IteratorL>& __x,
	      const reverse_iterator<_IteratorR>& __y)
    { return __x.base() < __y.base(); }

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator<=(const reverse_iterator<_IteratorL>& __x,
	       const reverse_iterator<_IteratorR>& __y)
    { return __x.base() >= __y.base(); }

  template<typename _IteratorL, typename _IteratorR>
    inline _GLIBCXX17_CONSTEXPR bool
    operator>=(const reverse_iterator<_IteratorL>& __x,
	       const reverse_iterator<_IteratorR>& __y)
    { return __x.base() <= __y.base(); }
```

两个迭代器的比较，就是base函数的比较，或者说是protected变量`current`的比较。



#### make\_reverse\_iterator

``` cpp
  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 2285. make_reverse_iterator
  /// Generator function for reverse_iterator.
  template<typename _Iterator>
    inline _GLIBCXX17_CONSTEXPR reverse_iterator<_Iterator>
    make_reverse_iterator(_Iterator __i)
    { return reverse_iterator<_Iterator>(__i); }
```

生成一个reverse_iterator，调用其构造函数。



## 4 back\_insert\_iterator

#### 定义

``` cpp
  template<typename _Container>
    class back_insert_iterator
    : public iterator<output_iterator_tag, void, void, void, void>
```



#### 构造函数

``` cpp
      /// The only way to create this %iterator is with a container.
      explicit _GLIBCXX20_CONSTEXPR
      back_insert_iterator(_Container& __x)
      : container(std::__addressof(__x)) { }
```

构造函数只能通过一个`_Container`参数构造。



#### operator= 函数

``` cpp
      _GLIBCXX20_CONSTEXPR
      back_insert_iterator&
      operator=(const typename _Container::value_type& __value)
      {
	container->push_back(__value);
	return *this;
      }

      _GLIBCXX20_CONSTEXPR
      back_insert_iterator&
      operator=(typename _Container::value_type&& __value)
      {
	container->push_back(std::move(__value));
	return *this;
      }
```

这种迭代器的`operator=`就使用`push_back`将值放入容器的最后一位。



#### operator*、++ 函数

``` cpp
      /// Simply returns *this.
      _GLIBCXX20_CONSTEXPR
      back_insert_iterator&
      operator*()
      { return *this; }

      /// Simply returns *this.  (This %iterator does not @a move.)
      _GLIBCXX20_CONSTEXPR
      back_insert_iterator&
      operator++()
      { return *this; }

      /// Simply returns *this.  (This %iterator does not @a move.)
      _GLIBCXX20_CONSTEXPR
      back_insert_iterator
      operator++(int)
      { return *this; }
```

`operator*`函数、`operator++`函数只是返回`*this`。



#### back\_inserter

``` cpp
  /**
   *  @param  __x  A container of arbitrary type.
   *  @return  An instance of back_insert_iterator working on @p __x.
   *
   *  This wrapper function helps in creating back_insert_iterator instances.
   *  Typing the name of the %iterator requires knowing the precise full
   *  type of the container, which can be tedious and impedes generic
   *  programming.  Using this function lets you take advantage of automatic
   *  template parameter deduction, making the compiler match the correct
   *  types for you.
  */
  template<typename _Container>
    _GLIBCXX20_CONSTEXPR
    inline back_insert_iterator<_Container>
    back_inserter(_Container& __x)
    { return back_insert_iterator<_Container>(__x); }
```

利用传入的容器类型返回一个back\_insert\_iterator。



## 5 front\_insert\_iterator

基本设计和back\_insert\_iterator一致，区别是`operator=`调用容器的`push_front`。当然，还有一个`front_inserter`函数，利用传入的容器参数，返回一个front\_insert\_iterator。



## 6 insert\_iterator

#### 定义

``` cpp
template<typename _Container>
    class insert_iterator
    : public iterator<output_iterator_tag, void, void, void, void>
```

insert\_iterator和back\_insert\_iterator、front\_insert\_iterator类似，模板参数都是一个容器类型。



#### protected变量

``` cpp
      _Container* container;
      _Iter iter;
```

只有两个，一个是容器**指针**，一个是它的迭代器。



#### 构造函数

``` cpp
      /**
       *  The only way to create this %iterator is with a container and an
       *  initial position (a normal %iterator into the container).
      */
      _GLIBCXX20_CONSTEXPR
      insert_iterator(_Container& __x, _Iter __i)
      : container(std::__addressof(__x)), iter(__i) {}
```

唯一的构造函数，参数是一个容器类型，和它的迭代器。



#### operator=函数

``` cpp
      _GLIBCXX20_CONSTEXPR
      insert_iterator&
      operator=(const typename _Container::value_type& __value)
      {
	iter = container->insert(iter, __value);
	++iter;
	return *this;
      }

      _GLIBCXX20_CONSTEXPR
      insert_iterator&
      operator=(typename _Container::value_type&& __value)
      {
	iter = container->insert(iter, std::move(__value));
	++iter;
	return *this;
      }
```

`operator=`函数的作用是调用容器的`insert`函数，将传入的value插入到容器，并返回插入位置的下一个位置。



其他的函数和back\_insert\_iterator、front\_insert\_iterator类似，另外，也有一个`inserter`函数，接受一个容器和它的迭代器，并返回一个相应的insert\_iterator。



## 7  move\_iterator

#### 定义

``` cpp
template<typename _Iterator>
    class move_iterator;
```

\_move\_iterator和之前的迭代器区别不大，唯一的不同是解引用操作将iterator的代表的值转为右值引用。另外，也存在着一个名为`make_move_iterator`。

#### 相关代码

``` cpp
      using __traits_type = iterator_traits<_Iterator>;
      using __base_ref = typename __traits_type::reference;
      typedef typename conditional<is_reference<__base_ref>::value,
			 typename remove_reference<__base_ref>::type&&,
			 __base_ref>::type		reference;

      explicit _GLIBCXX17_CONSTEXPR
      move_iterator(iterator_type __i)
      : _M_current(std::move(__i)) { }

      _GLIBCXX17_CONSTEXPR reference
      operator*() const
      { return static_cast<reference>(*_M_current); }

      _GLIBCXX17_CONSTEXPR reference
      operator[](difference_type __n) const
      { return std::move(_M_current[__n]); }

```

## 8 \_\_is\_move\_iterator

``` cpp
  //
  // Move iterator type
  //
  template<typename _Tp>
    struct __is_move_iterator
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Iterator>
    struct __is_move_iterator<reverse_iterator<_Iterator> >
      : __is_move_iterator<_Iterator>
    { };

  template<typename _Iterator>
    struct __is_move_iterator<move_iterator<_Iterator> >
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };
```

## 9 \_\_miter\_base

``` cpp
  // Fallback implementation of the function in bits/stl_iterator.h used to
  // remove the move_iterator wrapper.
  template<typename _Iterator>
    _GLIBCXX20_CONSTEXPR
    inline _Iterator
    __miter_base(_Iterator __it)
    { return __it; }

  template<typename _Iterator>
    _GLIBCXX20_CONSTEXPR
    auto
    __miter_base(reverse_iterator<_Iterator> __it)
    -> decltype(__make_reverse_iterator(__miter_base(__it.base())))
    { return __make_reverse_iterator(__miter_base(__it.base())); }

  template<typename _Iterator>
    auto
    __miter_base(move_iterator<_Iterator> __it)
    -> decltype(__miter_base(__it.base()))
    { return __miter_base(__it.base()); }
```

目的是去掉move\_iterator的wrapper。\_\_make\_reverse\_iterator和make\_reverse\_iterator功能相同，设计是为了给C++11也能使用这个功能。

## 10 \_\_niter\_base

``` cpp
  // Fallback implementation of the function in bits/stl_iterator.h used to
  // remove the __normal_iterator wrapper. See copy, fill, ...
  template<typename _Iterator>
    _GLIBCXX20_CONSTEXPR
    inline _Iterator
    __niter_base(_Iterator __it)
    _GLIBCXX_NOEXCEPT_IF(std::is_nothrow_copy_constructible<_Iterator>::value)
    { return __it; }

  template<typename _Iterator>
    _GLIBCXX20_CONSTEXPR
    auto
    __niter_base(reverse_iterator<_Iterator> __it)
    -> decltype(__make_reverse_iterator(__niter_base(__it.base())))
    { return __make_reverse_iterator(__niter_base(__it.base())); }
  
  template<typename _Iterator, typename _Container>
    _GLIBCXX20_CONSTEXPR
    _Iterator
    __niter_base(__gnu_cxx::__normal_iterator<_Iterator, _Container> __it)
    _GLIBCXX_NOEXCEPT_IF(std::is_nothrow_copy_constructible<_Iterator>::value)
    { return __it.base(); }
```