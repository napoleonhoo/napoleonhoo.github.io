---
layout: default
---

# Others

## CHECK
Code:
```cpp
#define CHECK(condition)  \
      LOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition))) \
             << "Check failed: " #condition " "

#define PCHECK(condition) \
	PLOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NO_TAKEN(!(condition))) \
	        << "Check failed: " #condition " "

#define PLOG_IF(severity, condition) \
	static_cast<void>(0), \
	!(condition) ? (void) 0 : @ac_google_namespace@::LogMessageVoidfy() & PLOG(serverity)

#define LOG_IF(severity, condition) \
  static_cast<void>(0),             \
  !(condition) ? (void) 0 : @ac_google_namespace@::LogMessageVoidify() & LOG(severity)

#ifndef GOOGLE_PREDICT_BRANCH_NOT_TAKEN
#if @ac_cv_have___builtin_expect@
#define GOOGLE_PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))
#else
#define GOOGLE_PREDICT_BRANCH_NOT_TAKEN(x) x
#endif
#endif
```
有点儿意思：
- 一方面CHECK返回值，一方面打印日志。
- 编译期优化都有，`__builtin_expect(x, 0))`。
