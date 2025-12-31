# 常见问题

## 1 多个worker push dense、sparse的互斥？

1. 多个push sparse请求之间串行，shard间并行。
   1. 请求之间使用`ConcurrentExecutionQueue`。（这是一个MPSC队列）
   2. shard间数据不冲突，并行无所谓。
2. 多个push dense请求之间串行，shard间并行。
   1. 请求之间使用`ThreadPool`。
   2. 将每个server上的dense table分位10个shard。（只是这么分，没有实际意义。）

## 2 多个worker的pull、push dense、sparse的互斥？

1. sparse的pull、push之间串行。
   1. 其使用的是同一个`ConcurrentExecutionQueue`。
2. dense的pull不加锁。<font color=red>why?</font>

