---
layout: default
---

# §Ch6 ignore_signal_call

```cpp
// Call func(args...). If interrupted by signal, recall the function.
template<class FUNC, class... ARGS>
auto ignore_signal_call(FUNC&& func, ARGS&&... args) -> typename std::result_of<FUNC(ARGS...)>::type {
	for (;;) {
		auto err = func(args...);

		if (err < 0 && errno == EINTR) {
			LOG(INFO) << "Signal is caught. Ignored.";
			continue;
		}

		return err;
	}
}
```
如果函数执行因为信号中断而返回错误的话，重试；否则返回。
