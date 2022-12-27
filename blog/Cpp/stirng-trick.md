

# string编程技巧

先看一段示例代码：

```cpp
static string emit_key;
emit_key.clear();
emit_key.resize(32 + key.size());
emit_key.resize(sprintf(&emit_key[0], "%s\t1\t%" PRId64, key.c_str(), tm));
```

