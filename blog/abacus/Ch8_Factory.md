---
layout: default
---

# §Ch8 Factory

先附一下源代码：
```cpp
template<class T>
class Factory : public VirtualObject {
public:
	typedef std::function<std::shared_ptr<T>()> producer_t;

	template<class TT>
	void add(const std::string& name) {
		add(name, []()->std::shared_ptr<TT> {
			return std::make_shared_ptr<TT>();
		});
	}
	void add(const std::string& name, producer_t producer) {
		CHECK(_items.insert({name, producer_t}).second) << "Factory item[" << name << "] already exits";
	}
	template<class TT = T>
	std::shared_ptr<TT> produce(const std::string& name) {
		auto it = _items.find(name);
		CHECK(it != _items.end()) << "Factory item[" << name << "] not found";
		std::shared_ptr<T> obj = it->second();
		CHECK(obj) << "Factor item is empty: " << name;
		std::shared_ptr<TT> x = std::dynamic_pointer_cast<TT>(obj);
		CHECK(x) << "Factory item[" << name << "] can not cast from " << typeid(
                 *obj).name() << " to " << typeid(TT).name();
        return x;
	}
private:
	std::map<std::string, producer_t> _items;
};
```
解释一下这个类：
- 这个类的主要数据结构是一个map，它的value是一个函数对象。
- 只有一个参数的add函数，调用有两个参数的add函数，第二个参数是一个函数对象，它的返回值是一个模板参数类型的共享指针，函数体中返回的是一个对模板参数调用`make_shared_ptr`。由此可见，`_items`的第二个参数，即`producer_t`的类型，是一个返回值为`shared_ptr<T>`的共享指针构造函数构造函数。
- get函数中，在找到相应的名称后，有`std::shared_ptr<T> obj = it->second();`，即调用此构造函数，并返回相应类型的共享指针。

这样设计的好处有：
- 提供了一个参数的add函数，简化了接口的使用。
- 在使用具体的对象指针时才进行构造，一方面不占用过多的内存（因为在abacus中，这个是用来存储各个层的对象的，一个模型一般不会用到所有的层。）；另一方面如果构造函数比较复杂的时候，可以减轻初始化时的负担。