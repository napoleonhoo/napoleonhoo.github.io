---
layout: default
---

# §Ch7 mpi相关

## mpi_check_consistency
```cpp
template <class T>
void mpi_check_consistency(const T* p, int count);
```
检查数据内容是否在所有mpi节点上都一样，通常用于参数一致性检测。参考：http://wiki.baidu.com/pages/viewpage.action?pageId=90772102
