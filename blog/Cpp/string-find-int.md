
# 使用`int`作为参数调用`string`的`find`函数的一个问题

这是一个在实际情况中发现的问题，这里根据此问题摘要写了一下代码如下（当然实际的问题要比这隐藏得多）。相信大家已经猜到了这段代码的本意，即在一串用逗号数字中找到某个特定的数字，但是由于失误，`find`的输入参数是一个`int`，而没有将其先转为字符串。

有问题代码：

``` cpp
  uint64_t u = 40462377004;
  std::string s = "43712811694,893213124";
  bool found = (s.find(u) == std::string::npos);
  std::cout << found << std::endl;
```

大家猜测这里`found`的值是多少呢？

这里的答案是`true`。

为什么呢？大家可以看到，这里`find`函数的声明是什么样的呢？

``` cpp
  size_type find( CharT ch, size_type pos = 0 ) const;
```

注意到，这里的输入参数应该是`char`类型，而我们输入的参数是`uint64_t`类型的，所以，在调用过程中，就涉及到从`uint64_t`到`char`的向下转换（downcast）。大家都知道，`char`是一个字节，对于`uint64_t`就是最低一个字节，八进制是2c，十进制就是44，根据ASCII表，字符就是‘,’（英文中的逗号）。所以，上面的代码相当于是执行了下面这句话，因此`found`显然是`true`。

``` cpp
  bool found = s.find(',') == std::string::npos;
```

