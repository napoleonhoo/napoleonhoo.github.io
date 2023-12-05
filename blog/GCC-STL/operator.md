# operator

## 1 `sizeof...` operator

这个操作符返回的是形参包（parameter pack）中参数的个数。举例如下：

``` cpp
#include <iostream>

template <typename... _Types>
class TestClass {
public:
    TestClass() { std::cout << sizeof...(_Types) << std::endl; }
};

int main(int argc, char** argv) {
    TestClass<int, int, float> tc;
    return 0;
}
```

上述代码输出“3”，代表TestClass实例化时是用了3个模板参数。