# align

## aligned_storage

定义(\<type\_traits\>)：

``` cpp
  template <std::size_t _Len, std::size_t _Align = __alignof__(
                                  typename __aligned_storage_msa<_Len>::__type)>
  struct aligned_storage;
```

> 提供嵌套类型 *`type`* ，其为平凡的标准布局类型，适于作为任何大小至多为 `Len` 且对齐要求为 `Align` 的因数的对象的未初始化存储。
>
> 参见[网址](https://www.noerror.net/zh/cpp/types/aligned_storage.html)。

使用方法举例（见`any::_Storage`）：

``` cpp
aligned_storage<sizeof(_M_ptr), alignof(void *)>::type _M_buffer;
```

则，对于`_M_buffer`来说，



## alignas

语法：

``` cpp
alignas( expression )
alignas( type-id )
alignas( pack... )
```

声明一个类型或对象的对齐要求。

举例1：

``` cpp
struct alignas(float) test_struct;
```

每个`test_struct`的对象都会对齐到`alignof(float)`个字节。

举例2：

``` cpp
alignas(64) char cacheline[64];
```

数组`cacheline`的每个元素会对齐到64个字节。



## alignof

语法：

``` cpp
alignof(type-id)
```

如：

``` cpp
alignof(int) // = 4
```

返回某一种类型的对齐方式（以字节为单位）。
