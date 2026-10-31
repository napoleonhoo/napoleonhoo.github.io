# C++ coredump：几个空指针的 case

> 空指针解引用是 C++ 程序 coredump 的常见原因之一。加在指针使用前进行有效的空指针检查，应当成为开发者的“习惯动作”。本文整理了几个与空指针相关的典型案例：有些错误一眼就能发现，有些则隐藏在表达式、容器接口或类型转换中。真实环境永远会更复杂！

## Case 1: 无效的判空逻辑

``` cpp
void function(int* ptr) {
    if (ptr == nullptr) {
	    VLOG(10) << "ptr is null";
    }
}
```

这段代码虽然判断了 `ptr` 是否为空，却只打印了一条日志，并没有终止函数的执行。如果后续代码继续解引用 `ptr`，仍然会触发崩溃。这是一个“马虎大意”的手误，却最终演变为线上问题。

## Case 2: bool 表达式中的空指针

``` cpp
void function(std::unordered_map<int, int>* ptr) {
    if (ptr != nullptr || ptr->find(10) != ptr->end()) {
	    return;
    }
}
```

问题出在 `||` 的短路求值逻辑上：

- 当 `ptr != nullptr` 为 `true` 时，右侧表达式不会执行，函数会直接返回；
- 当 `ptr == nullptr` 时，左侧表达式为 `false`，程序反而会继续计算右侧表达式，从而解引用空指针并导致崩溃。

如果这里的意图是“当 `ptr` 为空，或者容器中不存在 key 为 `10` 的元素时直接返回”，可以改成：

```cpp
void function(std::unordered_map<int, int>* ptr) {
    if (ptr == nullptr || ptr->find(10) == ptr->end()) {
        return;
    }
    
    // ...
}
```
## Case 3: map 的 value 是个空指针

``` cpp
class C {
 public:
	void func() {}
};

void function(std::unordered_map<int, C*>& m) {
    m[1]->func();
}
```

这里涉及 `std::unordered_map::operator[]` 的行为：

- 如果 key `1` 已经存在，`m[1]` 会返回对应的 value；
- 如果 key `1` 不存在，`operator[]` 会先插入一个新元素，并对 value 进行默认初始化。

由于 value 的类型是 `C*`，默认初始化后得到的是空指针。因此，当 key `1` 不存在时，`m[1]->func()` 实际上是在解引用空指针，可能导致程序崩溃。

在这个场景中，仅仅补充空指针判断还不够。既然业务预期 map 中存在 key `1`，就应当同时检查元素是否存在以及对应的指针是否有效：

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

使用 `find` 还有一个额外好处：查询失败时不会像 `operator[]` 那样修改容器。

## Case 4: 强制类型转换误用引发的空指针

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

`C` 和 `D` 虽然都继承自 `A`，但二者之间不存在继承关系。当 `ptr` 实际指向一个 `C` 对象时，使用 `dynamic_cast<D*>` 进行向下转换会失败，并返回 `nullptr`。随后调用 `d->func()`，就会触发空指针解引用。

这个问题在简化后的示例中非常明显，但在真实生产环境里，`function` 的调用位置可能离定义很远，而 `ptr` 指向的实际对象又可能由工厂函数根据配置动态创建。经过较长的调用链后，错误的运行时类型往往不容易被直接发现。

## 总结

真实的线上环境通常包含复杂的业务逻辑、调用链和对象生命周期，空指针问题也因此可能以多种形式出现，特别是涉及到具体业务的时候。但是，无论业务如何复杂，从技术的角度上来讲，加一层空指针判断的保险，永远是最好的选择！