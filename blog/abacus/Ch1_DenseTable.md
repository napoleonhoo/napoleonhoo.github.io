---
layout: default
---

# §Ch1 DenseTable

## DenseTableShard
代码的主要内容很简单，实际存储数据的是一个`std::vector<VALUE>`。整个结构体向64位对齐。

## DenseTable
主要用来存储的数据结构是：
```cpp
typedef DenseTableShard<VALUE> shard_type;
std::unique_ptr<shard_type[]> _local_shards;
```

## DenseSGDValue
其中主要有以下几个数据：
```cpp
template<class T>
struct DenseSGDValue {
	T w;
	T avg_w;
	T ada_d2sum;
	T ada_g2suml
	T mom_velocity;
};

```
代码中，DenseTable模板参数的内容即为DenseSGDValue。
```cpp
// dnn_plugin.h
std::map<int, std::unique_ptr<abacus_table::DenseTable<ParamValue>>> _dense_tables;

// dnn_instance.h
typedef abacus_learner::DenseSGDValue<float> ParamValue;
```
