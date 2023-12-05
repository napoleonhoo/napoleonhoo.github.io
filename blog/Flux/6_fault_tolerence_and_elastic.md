# 6 容错&弹性伸缩

研究fail的原因，找到容错的必要性。

## 1 容错

### 1.1 coordinator

fail的几率很小。一般是内存不足，单纯容错解决不了问题。

#### 假设必须要的话

没太大必要吧？

### 1.2 worker

fail的几率相对较大，一般是样本问题，有容错的**必要性**。

因为worker是无状态的（stateless）的。甚至对于整个训练任务来说，丢失几个文件来说，也是可以接受的。有容错的**可能性**。

失败的两种情况：超时、其他。

#### 超时

1. 一直检查心跳是否超时，若心跳超时，删除worker并通知elastic迁移worker节点容器。新worker启动后会向coordinator发送regist请求。
2. train：先判断丢失文件数（该worker预取了文件，但未训练）是否达到阈值（例如<1%），未超过直接跳过，超过阈值则将文件重新分配到其他worker。 

#### 其他失败

1. 一般请求失败后，若心跳还未超时, 拉黑5分钟(可配)，之后的prefetch和train请求都不包含这个worker。
2. worker在文件fail达到一定比例的时候，一般是20%，任务直接fail。
3. train：读文件失败直接跳过。另，参考2.

#### 所有失败的共同操作

1. prefetch：prefetch请求失败后直接忽略，错误处理延迟到train时，会发现文件未预取。
2. load/save：它是向随机worker发送的，请求失败后会重新选择worker重试。

#### 为什么要分这两种，即超时和其他？

因为超时可能是机器fail了，或进程fail了（读样本core、OOM等）。而其他的操作一般是读样本失败（不是core了）。

#### 为什么说worker是stateless的。

Worker就是一个训练服务器，类似于TrainningServer，类似于ParameterServer。

从C/S的模式来讲：
- 对于训练来说，C是client，W是server。
- 对于参数来说，push、pull的时候，W是client，S是server。对于load、save来说，C是client，S是server。

比起PServer来说，Worker*更*没有状态，因为PServer需要rank_id信息。

作为整个训练任务来说，是有状态的，因为需要checkpoint等。

### 1.3 server

fail几率很小，一般是内存问题，容错解决不了问题。

也没有必要，因为server是有状态的，容错就需要保存snapshot等，对性能（或资源）影响较大。另外，有patch model机制来做保证。再者，patch model本身就是一种容错机制。

超时直接fail。

#### 假设必须要的话

一般模式：coordinator心跳超时，发起迁移。时间慢，但资源少。

server：时时dump snapshot即可。

coordinator：心跳超时，记住rank_id。

多副本，主从模式：时间快，但可能需要double的server资源。（无意义）

核心问题是样本、后反馈链路较慢（设计端、网络、用户习惯等复杂的事情），训练不是瓶颈，一般不需要花费这么多的资源，防止可能性很小的事情，挽回一个checkpoint间隔左右的时间。

## 2 弹性伸缩

### 2.1 coordinator

不需要伸缩，这个角色只需要一个节点。

### 2.2 worker

只有worker有弹性伸缩的必要性与可能性。

- CPU、MEM利用率不足的节点，进行quota缩减。
- 过于不足的，缩节点。

反之：
- 对CPU、MEM利用率过高的节点，进行quota增加。
- 过于高的，增加节点。

### 2.3 server

不需要伸缩，也不能伸缩。过程操作过于复杂，需要改变哈希、重新加载等。
