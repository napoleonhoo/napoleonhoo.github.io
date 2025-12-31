---
layout: default
---

# §Ch3 Layers

## Layer
继承自Component。  
以`*Layer`结尾的类，继承自Layer类。一般会实现Layer中的这三个接口。
```cpp
class Layer : public Component {
public:
    virtual void initialize(DnnInstance *instance, abacus::Config conf) {
    }
    virtual void finalize() {
    }
    virtual void feed_forward() {
    }
    virtual void back_propagate() {
    }
};
```
除此之外，根据每个层的用途不同，也会实现其他的函数。大多数会使用一个Function类作为一个私有成员变量，函数`feed_forward() back_propagete()`就会调用Function的`compute()`函数。

## Component
Component实现了最基本的功能，专为工厂类设计。提供了`set_name() get_name() shared_from_this()`这三个主要函数。

## *Function
继承自Component。  
大多数Function提供了对`initialize() compute()`这两个函数的实现。
