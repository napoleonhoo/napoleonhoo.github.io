# § static data member initialization

参考链接：[cppreference-static](https://en.cppreference.com/w/cpp/language/static)



静态成员变量一般需要在类内声明，类外仍然需要定义、初始化一遍。

当C++17以来，静态成员变量可以被声明为inline的，可以在类内定义，并且不需要类外定义。

``` cpp
struct X {
	  inline static int n = 1;
};
```



#### const static member

如果静态成员变量为整型或枚举类型，声明为const但不是volatile的，可以使用initializer初始化。

``` cpp
struct X {
  const static int n = 1;
  const static int m{2}; // since C++11
  const static int k;
};
const int X::k = 3;
```



自C++11依赖，如果LiteralType（字面值类型）被声明为constexpr，必须要使用initializer来初始化。

``` cpp
struct X {
  constexpr static int arr[] = {1, 2, 3};
  constexpr static std::complex<double> n = {1, 2};
  constexpr static int k; // Error: 必须要有一个initializer。
```



