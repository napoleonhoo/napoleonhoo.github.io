# 第四章 训练

本章主要是讲述从worker视角看到的训练。

## 1 pull dense

### 1.1 worker发送请求

利用dnn plugin获得所有参数的Tensor，并放入`std::vector<Region>`里面。直接向所有的server发送请求。

### 1.2 server处理

- 调用table的`pull_dense()`函数。将dense数据填入一个array。

- 发送回应将array填入brpc的`cntl->response_attachment`中。

### 1.3 worker收到回应

将`response_attachment`将数据填到`std::vector<Region>`里面。

## 2 fill key：提出key以供pull sparse使用

## 3 pull sparse

### 3.1 worker发送请求

- 将`sign_map`中的数据放入到`SparsePullTaskData`的数据结构中。逻辑就是先根据key和shard的相关关系，计算出要请求的`server_id`，将`sign_map`的主要数据填充到`shard_data`对应`server_id`的`key_value_list`里面。它的主要相关代码如下：

``` cpp
struct ShardSparsePullData {
    struct KeyValue {
        uint64_t key;
        float* value;
        uint64_t uid;
        uint32_t version;
    };
    std::vector<KeyValue> key_value_list;
};

struct SparsePullTaskData {
    std::vector<ShardSparsePullData> shard_data;
};

size_t get_common_server_idx(uint64_t sign) {
    return (sign % _common_total_shard_num) % _server_num;
}
void make_pull_shard_data(CompactSignMap& sign_map,
                          std::shared_ptr<SparsePullTaskData>& shard_sorted_kv_list) {
    size_t server_idx = 0;
    for (auto [key, value, slot_id, uid] : sign_map) {
        server_idx = get_common_server_idx(key);
        shard_sorted_kv_list->shard_data[server_idx].key_value_list.emplace_back(key, value, 0, 0);
    }
 }
```

- 对每个server下面的数据做以下同样的处理：对`key_value_list`中的key进行排序。遍历每一个元素，将key填入到`request_attachment`里面，并去重key。即每个key只发送一遍。

*备注：感觉这个地方可以优化。make pull shard data函数中sign map可以直接memcpy到sparse pull task data。另外，感觉没必要复制一边再去重，是不是可以直接用map这种数据结构。另外，排序是为了去重吗？*

### 3.2 server处理

- 利用key确定global shard id和local shard id，并在相应的shard中寻找这个key。
- 先使用cache，如果能在cache中找到这个key的话，则返回相应的参数。如果没找到，则直接在shard中找，如果能找到的话，则返回相应的参数；否则，创建这个key对应的数据。

### 3.3 worker收到回应

- 将收到的应答复制到相应的`key_value_list`的`value`字段中，这也是在`sign_map`里面。

## 4 input data：将sparse数据回填

- 按`begin_idx`到`end_idx`将sign从`slot_signs`中取出。
- 从`table_sign_map`中找到相应的sign，取出weight，并放入一个vector中。
- 将weight复制到相应的slot的tensor。

## 5 train

调用了abacus、paddle的接口，来获得相应的dense的tensor和grad tensor。

## 6 fill gradient：填上sparse的gradient

- 按`begin_idx`到`end_idx`将sign从`slot_signs`中取出。
- 从`table_sign_map`中找到相应的sign，取出grad，并放入一个vector中。
- 将weight复制到相应的slot的grad tensor。

*备注：grad啥时候去的table sign map。*

## 7 push sparse

### 7.1 worker发送请求

``` cpp
struct ShardSparsePushData {
    ShardSparsePushData() {}
    ~ShardSparsePushData() noexcept {}
    struct KeyValueSlotUid {
        uint64_t key;
        float* value;
        int slot;
        uint64_t uid;
        uint32_t version;
    };
    std::vector<KeyValueSlotUid> key_value_list;
};
struct SparsePushTaskData {
    std::vector<ShardSparsePushData> shard_data;   //sparse数据按key hash分片
    std::vector<std::shared_ptr<SignMapPool::PooledObject>> sign_maps;
};

void make_push_shard_data(size_t table_id,
        CompactSignMap& sign_map,
        std::vector<ShardSparsePushData>& shard_data,
        hash_map<uint64_t, uint32_t>& version_map) {
    size_t server_idx = 0;
    for (auto [key, value, slot_id, uid] : sign_map) {
        server_idx = get_common_server_idx(key);
        shard_data[server_idx].key_value_list.emplace_back(key, value, slot_id, tmp_uid, version_map[key]);
    }
}
```

- 将`table_sign_map`数据根据key，选择相应的server id，将数据放入`ShardSparsePushData`中。生成一个异步push的任务`SparseAsyncTask`的对象，并放入异步队列`_push_sparse_task_queue_map`。
- 在另一个队列消费者函数`push_sparse_task_consume()`，遍历异步队列，将每一个复制到新的`SparseAsyncTask`对象中，并将其push到`task_list`中，计算当前的`merge_count`。
- 当前的`merge_count`未到达`merge_size`时，继续merge。对相同的key去重，相同的key的数值相加，并merge到`task_list`的第0个元素。
- 当前的`merge_count`到达`merge_size`时，向server发送请求。请求前再merge一遍。

### 7.2 server处理

- 分为两种形式，batch更新和单个更新。
- 在batch更新且带有cache的情况下，会先在cache中查找，当cache中没有的时候，会在shard中查找，如果找不到，会初始化一个值；当在shard中找到或者在cache中找到的情况下，会进行更新。更新规则：show、click与push的show、click**相加**，score更新有一个固定的公式，lr、mf等会根据accessor的sgd rule的公式进行更新。更新完之后，会判断是否需要创建mf向量。
- batch更新和分别更新的区别在于何时更新。

### 7.3 worker收到回应

释放相应内存，不需额外数据上的操作。

## 8 push dense

### 8.1 worker发送请求

- worker push的数据来源于grad tensor。
- 将grad数据放入到`DenseAsyncTask`中，并放入`_push_dense_task_queue_map`。和push sparse时类似，仍然是异步化的push。
- push之前先merge。merge的规则就是简单的相加。
- push时分为数据是否压缩。

### 8.2 server处理

根据使用的sgd算法进行更新。

### 8.3 worker收到回应

没有太多处理。
