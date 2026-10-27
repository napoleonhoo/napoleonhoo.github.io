# C++ 易错点：几个空指针的 case

> C++ 的空指针，是造成线上问题，如 coredump 等，最大的诱因之一。加上空指针保护，应该作为程序员的“习惯动作”。下面展示了一些空指针相关的 case，当你看下去的时候，会发现人们在出错这个方面，真的有无穷无尽的方法！有些逻辑很明显，有些则更隐蔽一些。真实环境永远会更复杂！

## Case 1 无效的判空逻辑

``` cpp
void function(int* ptr) {
    if (ptr == nullptr) {
	    VLOG(10) << "ptr is null";
    }
}
```

这里的错误就很明显了，这段代码的来源是一位新人开发者的手误，然而却最终造成了线上问题。

## Case 2 bool 表达式中的空指针

``` cpp
void function(std::unordered_map<int, int>* ptr) {
    if (ptr != nullptr || ptr->find(10) != ptr->end()) {
	    return;
    }
}
```

这里的错误涉及一个 bool 表达式，当 `ptr` 为空时，正常应该直接返回。而如果像上面这样写呢，就还会执行后面的 `find` 函数，此时就会 coredump 了。修改方案也很简单，`||` 改成 `&&` 就可以了。

## Case 3 map 的 value 是个空指针

``` cpp
class C {
 public:
	void func() {}
};

void function(std::unordered_map<int, C*>& m) {
    m[1]->func();
}
```

这里有一个小的知识点，如果前面的代码，给这个 map 放入了 key 为 1，且 value 为一个非空的 `C*` 的话，那就没有什么问题。但是如果 map 中没有 key 为 1 的元素，那么 map 的 `operator[]` 会先插入一个 key 为 1，value 为默认构造（此处即为空指针）的元素。这样，上面的代码就会造成 coredump 了。

那么如何修改呢？最简单的方式，当然就是加上空指针。不过在这种业务场景下，既然造成了这个 case，说明我们在预期 map 中有一个 key 为 1 的元素，而现在却没有。最好的方式是先去查找 map 中有没有 key 为 1 的元素，同样，空指针判断也要保留。可以如下修改：


``` cpp
void function(std::unordered_map<int, C*>& m) {
	auto iter = m.find(1);
	if (iter == m.end()) {
		return;
	}
	if (iter->second == nullptr) {
		return;
	}
	// ...
}
```

## Case 4 强制类型转换误用引发的空指针

``` cpp
class A {
 public:
	virtual void func() = 0;
};

class C : public A {
 public:
	void func() {}
};

class D : public A {
 public:
	void func() {}
};

void function(A* ptr) {
	D* d = dynamic_cast<D*>(ptr);
	d->func();
}

int main() {
	A* ptr = new C();
	function(ptr);
	delete ptr;
	return 0;
}
```

`C` 和 `D` 虽然都是 `A` 的子类型，但当使用 `dynamic_cast` 将 `C` 的指针强制转换为 `D` 的指针时，就会得到空指针的结果。

这里发生 case 的实际情况很复杂，对 `function` 的调用离它的定义十万八千里，同时 `ptr` 的初始化又是工厂函数根据配置来生成的。当我将这个问题整理出来，这里的问题其实就显而易见了。但是在生产环境中，问题往往不会如此明显。

## 结论

真正的线上环境是很复杂的，特别是涉及到具体业务的时候。但是，无论业务如何复杂，从技术的角度上来讲，加一层空指针判断的保险，永远是最好的选择！