---
layout: default
title: RocksDB
---

# §Ch0 Introduction

RocksDB是一个高性能嵌入式KV数据库，fork自Google的LevelDB，专为多核CPU而优化，并且为了IO密集型的工作，充分利用了快速存储（如SSD）优化。基于LSM tree数据结构，用C++写成，并提供了多种官方语言绑定。

## 1 特性

RocksDB提供了LevelDB所有的特性，另外还有：

- 事务
- 备份和快照（snapshot）
- Column families
- Bloom filters
- Time to live(TTL) support
- Universal compaction
- Merge operators
- Statistics collection
- Geospatial indexing

## 附录

### 附录1 参考

