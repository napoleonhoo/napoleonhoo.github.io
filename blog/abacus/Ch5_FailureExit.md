---
layout: default
---

# §Ch5 FailureExit
代码：
```cpp
struct FailureExit;

inline std::vector<FailureExit*>& local_failure_exits() {
	thread_local std::vector<FailureExit*> x;
	return x;
}

struct FailureExit : public boost::noncopyable {
	std::function<void()> func;

	explicit FailureExit(std::function<void()> func_) : func(std::move(func_)) {
		auto& items = local_failure_exits();
		items.push_back(this);
	}

	~FailureExit() {
		auto& items = local_failure_exits();
		CHECK(!items.empty() && items.back() == this);
		items.pop_back();

		if (std::uncaught_exception()) {
			func();
		}
	}
};
```
其中：
- `std::uncaught_exception()`的作用是：检测当前thread中是否有未被捕获的execption。参见：[cppreference](https://en.cppreference.com/w/cpp/error/uncaught_exception)
- 如果只从vector的back来看，就像一个stack，先进后出。

## 一点想法
如何实现类似于golang中的defer呢？可以参考这个类。
- constructor是一个函数对象，destructor中调用这个函数。
- 在代码的一个local namespace中，进入时，定义一个这个类的变量（栈上的）。当执行退出这个local namespace时，这个变量destrcut，自动调用这个destructor。
- 完成了defer的功能。
