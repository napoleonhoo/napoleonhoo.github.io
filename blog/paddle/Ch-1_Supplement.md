# Supplement



## boost

### boost::variant

参考链接：https://www.boost.org/doc/libs/1_61_0/doc/html/variant.html

Multi-type single-value，类似non-POD type的union。

``` cpp
boost::variant<int, std::string> u("hello world");
```



### boost::static_visitor

