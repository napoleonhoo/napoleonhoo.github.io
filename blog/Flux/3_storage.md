# 第三章 存储

## 1. server中的存储

server作为模型训练框架中读取所有模型参数，特别是大模型，总量可能达到TB级别。不仅如此，还包括了push到server上的梯度参数。

### 1.1 allocator的设计

主要代码列在下面：

``` cpp
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t arena_size = (8 << 20), size_t max_bytes = 1024) {
        _arena_size = arena_size; // 默认8MB，最大可分配的是1024B。
        _counter = 0;
        _cur_arena = nullptr;
        _cur_arena_index = 0;
        _free_list.resize(max_bytes, nullptr);
    }

    ArenaAllocator(const ArenaAllocator&) = delete;
    ~ArenaAllocator() {
        for (auto ptr : _arena_list) {
            free(ptr);
        }
    }

    void* alloc(size_t bytes) {
        CHECK(bytes < _free_list.size());

        if (_free_list[bytes] != nullptr) {
           	// 当_free_list[bytes]中存在着可用的空间时，取出头节点。
            void* ptr = (void*)_free_list[bytes];
            _free_list[bytes] = _free_list[bytes]->next;
            ++_counter;
            return ptr;
        }
		
        // 当前无arena（初始化的时候），或者当前arena空闲不足够了
        if (_cur_arena == nullptr || _cur_arena_index + bytes > _arena_size) {
            create_new_arena();
        }
        
        // 分配了新的arena，或者当前arena有足够的空间的时候。
        void* ptr = (void*)(_cur_arena + _cur_arena_index);
        _cur_arena_index += bytes;
        ++_counter;
        return ptr;
    }
	
    // release的时候，直接将ptr放入_free_list[bytes]的头。
    // 注：_free_list[bytes]可能是一个链表。
    void release(void* ptr, size_t bytes) {
        Node* node = (Node*)ptr;
        node->next = _free_list[bytes];
        _free_list[bytes] = node;
        --_counter;
    }

    size_t size() const {
        return _counter;
    }

private:
    struct Node {
        Node* next;
    };

    size_t _arena_size;              // arena分配的大小，默认是8<<20，即8MB。
    size_t _counter;                 // 目前在用的arena的个数。 
    std::vector<char*>  _arena_list; // 所有的arena列表。
    char*  _cur_arena;               // 指向现在的arena的指针。
    size_t _cur_arena_index;         // 现在arena的index。
    std::vector<Node*> _free_list;   // 空闲的arena列表。

    void create_new_arena() {
        char* arena = nullptr;
        // 位于<stdlib.h>
        // 原型：int posix_memalign(void **memptr, size_t alignment, size_t size);
        // 分配size bytes的空间，并将地址放入*memptr。这个地址一定是alignment的倍数。
        // alignment的要求必须是2的次方且是sizeof(void*)的倍数。
        // *memptr可以用free来释放。
        posix_memalign((void**)&arena, sizeof(void*), _arena_size);
        // 将其放入_arena_list，可以以后遍历来free。
        _arena_list.push_back(arena);
        // 重置_cur_arena指向刚刚分配的arena，_cur_arena_index设为0.
        _cur_arena = arena;
        _cur_arena_index = 0;
    }
};
```

主要成员变量的作用将上述注释。

主要逻辑之一：alloc

- 当`_free_list`中对应要分配的bytes有空闲的arena时，直接返回。
- 当没有任何的arena（初始化的时候），或者当前arena没有足够空间的时候，创建一个新的arena。
- 当分配好了一个新的arena，或者当前的arena有足够的空间的时候，返回`_cur_arena+_cur_arena_index`指向的地方。

主要逻辑之二：release

- 根据其大小直接放入`_free_list`中。

注意`_free_list`是个vector，它共有`max_bytes`个元素，其中每个元素的类型是Node指针，使用Node的结构，组成了一个链表。

这里的设计，关注到前后文注意到：

- 这并不是通用的设计。对于一般性的程序内存分配，这个设计并非是很好的设计。但是，这个allocator只是给sparse  shard用的。从目的上来看，这是一个绝佳的设计。目的是快速，而非升内存。
- 需要一定的编程能力来使用。对这里分配的内存的使用，并非显式地调用构造函数，而是通过强制类型转换来直接使用这块内存。这就要求使用这个allocator的开发者有着比较清晰的头脑，认真仔细。这里的使用方法确实不同寻常，但确实可行的，可以参看下面的测试代码。

``` cpp
#include <cstdlib>
#include <cstdint>

#include <iostream>

struct Node {
    int32_t a;
};

struct Object {
    int64_t a;
    int64_t b;
    void set_b(int64_t new_b) { b = new_b; }
};

void* alloc(size_t size = sizeof(Object)) {
    void* ptr = malloc(size);
    return ptr;
};

int main(int argc, char** argv) {
    void* ptr = alloc();
    Node* node = (Node*)ptr;
    node->a = 10;
    std::cout << node->a << std::endl;  // 10

    Object* object = (Object*)ptr;
    object->a = 100;
    object->b = 1000;
    std::cout << object->a << std::endl;  // 100
    std::cout << object->b << std::endl;  // 1000
    object->set_b(1);
    std::cout << object->a << std::endl;  // 100
    std::cout << object->b << std::endl;  // 1
}
```

### 1.2 sparse shard

它是一个模板，有两个参数：`KEY`和`VALUE`。从后面sparse table的使用上，可以知道`KEY=uint64_t`和`VALUE=DownpourFixedFeatureValue`。

简单解释一下`DownpourFixedFeatureValue`，它的定义如下：

``` cpp
class DownpourFixedFeatureValue {
public:
    DownpourFixedFeatureValue() = default;
    ~DownpourFixedFeatureValue() = default;
    float* data() {
        return _data;
    }
    size_t size() {
        return static_cast<size_t>(_size);
    }
    void set_size(size_t size) {
        _size = static_cast<uint16_t>(size);
    }
    size_t capacity() {
        return static_cast<size_t>(_capacity);
    }
    void set_capacity(size_t capacity) {
        _capacity = static_cast<uint16_t>(capacity);
    }
private:
    uint16_t _capacity;
    uint16_t _size;
    float _data[0];
};
```

它其实就是包装了一个“flaot”数组，也可以称为是float指针，指向一个“float”数组。（这里的float加引号，是因为看起来它是float的，但实际上不准确，后文详细解释。）即它的private成员`_data`。

主要的数据存储结构：

``` cpp
map_type _buckets[DOWNPOUR_SPARSE_SHARD_BUCKET_NUM];
```

`_buckets`是一个数组，总共有`DOWNPOUR_SPARSE_SHARD_BUCKET_NUM`个元素，每个元素都是`map_type`。

其中，`DOWNPOUR_SPARSE_SHARD_BUCKET_NUM`为：

``` cpp
static const int DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS = 6;
static const size_t DOWNPOUR_SPARSE_SHARD_BUCKET_NUM = (size_t)1 << DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS;
```

其值为64。

其中，`map_type`为：

``` cpp
typedef typename mct::closed_hash_map<KEY, mct::Pointer, std::hash<KEY>> map_type;
```

我们以`insert`函数为例，解释sparse shard的设计。

``` cpp
VALUE& insert(const KEY& key, size_t value_size) {
    size_t hash = _hasher(key);
    size_t bucket = compute_bucket(hash);
    auto res = _buckets[bucket].insert_with_hash( {key, nullptr}, hash);
    if (res.second || res.first->second == nullptr) {
        size_t alloc_size = sizeof(VALUE) + sizeof(float) * value_size;
        res.first->second = _alloc.alloc(alloc_size);
        ((VALUE*)((void*)(res.first->second)))->set_capacity(alloc_size);
    } else {
        VALUE* o_ptr = (VALUE*)((void*)(res.first->second));
        if (o_ptr->size() < value_size) {
            size_t alloc_size = sizeof(VALUE) + sizeof(float) * value_size;
            res.first->second = _alloc.alloc(alloc_size);
            VALUE* n_ptr = (VALUE*)((void*)(res.first->second));
            n_ptr->set_capacity(alloc_size);
            memcpy(n_ptr->data(), o_ptr->data(), sizeof(float) * o_ptr->size());
            _alloc.release(o_ptr, o_ptr->capacity());
        }
    }
    VALUE* o_ptr = (VALUE*)((void*)(res.first->second));
    o_ptr->set_size(value_size);
    return *o_ptr;
}
```

每回插入到shard的时候，都会有一个key和value size作为入参。步骤如下：

1. 计算key的hash值。
2. 计算shard内的bucket id，即`_buckets`数组的索引。
3. 将key插入到相应的bucket，value暂时指定为nullptr。
4. 当insert成功，或者返回的iterator为空时，通过allocator分配内存，并将其赋值给value。
5. 否则的话，判断value的size是否小于value size，如果是的话，需要增加些内存。
6. 返回指向value的指针。

注意点：

1. 这里面用到的allocator就是上面讲到的allocator；
2. sparse shard里面还有cache的设计，也是map类型，但它不是数组。对于enable memory cache的训练任务，在第一次load模型和保存batch model的时候，会将show值按从大到小排序，取固定的k个，这个值大约是所有key的数量乘以cache rate。

### 1.3 sparse table

主要存储结构：

``` cpp
std::unique_ptr<shard_type[]> _local_shards;
```

其中，`shard_type`就是上面讲到的shard类型，简单来讲，`_local_shards`就是shard的数组。它的初始化如下：

``` cpp
_local_shards.reset(new shard_type[_real_local_shard_num]);
```

所以，这个数组的长度是`_real_local_shard_num`，它的值基本上等于shard num / server num，即每个server上load的sparse的shard数量。

### 1.4 dense table

dense参数的存储结构就没那么复杂了，这里简单贴一下存储结构的代码：

``` cpp
Eigen::MatrixXf _data;
```

也就是一个MatrixXf类型。

### 1.5 load dense

由于是多个server都要加载dense table，所以每个server就需要决定加载table的哪个part。看下面的代码

``` cpp
size_t dim_num_per_file = _value_accessor->fea_dim() / file_list.size() + 1;
size_t dim_num_per_shard = _value_accessor->fea_dim() / _shard_num + 1;
size_t start_dim_idx = dim_num_per_shard * _shard_idx; 
size_t start_file_idx = start_dim_idx / dim_num_per_file;
size_t end_file_idx = (dim_num_per_shard * (_shard_idx + 1)) / dim_num_per_file;
end_file_idx = end_file_idx < file_list.size() ? end_file_idx : file_list.size() - 1;

size_t dim_col = _value_accessor->size() / sizeof(float);
```

举个例子：

``` cpp
// if
_value_accessor->fea_dim() = 465052;
file_list.size() = 5;
shard_num = 4;
_shard_idx = 2;
_value_accessor->size() = 5 * sizeof(float); // 一般的dense table（非summary table）都是这个大小
// then
dim_num_per_file = 93011; // 93010.4 + 1
dim_num_per_shard = 116264;
start_dim_idx = 232528;
start_file_idx = 2; // 2.5
end_file_idx = 3; // 3.7
dim_col = 5;
```

上面展示的dense table是5个文件，server的数量是4个。简单来讲，就是**把dense table的所有数据平均分配到每一个server上**。

load时的主要代码，逻辑就比较清晰了。

``` cpp
_data.resize(dim_num_per_shard, dim_col);
for (int i = start_file_idx; i < end_file_idx + 1; ++i) {
    channel_config.path = file_list[i];
    err_no = 0;
    auto read_channel = _afs_client->open_r(channel_config, &err_no);
    size_t file_start_idx = start_dim_idx - i * dim_num_per_file;
    for (size_t file_dim_idx = 0; file_dim_idx < dim_num_per_file; ++file_dim_idx) {
        if (read_channel->read_line(line_data) != 0) {
            break;
        }
        if (dim_idx < dim_num_per_shard && file_dim_idx >= file_start_idx) {
            _value_accessor->parse_from_string(line_data, dim_data_buffer);
            for (size_t col_idx = 0; col_idx < dim_col; ++col_idx) {
                _data(dim_idx, col_idx) = dim_data_buffer[col_idx];
            }
            ++dim_idx;
        }
    }
    read_channel->close();
    start_dim_idx += dim_num_per_file - file_start_idx;
}
```

`_data`是一个`dim_num_per_shard * dim_col`维的矩阵。将读入的数据直接放入`_data`。

### 1.6 load sparse

sparse table也需要每个server加载一部分，确定每个server加载哪个部分的逻辑见下：

``` cpp
static size_t sparse_local_shard_num(uint32_t shard_num, uint32_t server_num, uint32_t rank_id) {
    size_t split_per_server = shard_num / server_num;
    size_t increment = rank_id < (shard_num % server_num)? 1: 0;  
    return split_per_server + increment;
}
_real_local_shard_num = SparseTable::sparse_local_shard_num(_sparse_table_shard_num, _shard_num, _shard_idx);

_start_idx = _shard_idx;
_end_idx = start_idx + _real_local_shard_num * _shard_num;
```

举个例子：

``` cpp
// if
_sparse_table_shard_num = 1950;
_shard_num = 15;
shard_idx = 7;
// then
_real_local_shard_num = 130;
_start_idx = 7;
_idx = 7, 22, 37, 52, 67, 82, 97, 112 ... ;
```

加载代码主要如下：

``` cpp
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < _real_local_shard_num; ++i) {
    FsChannelConfig channel_config = get_converter_config(load_param);
    channel_config.path = file_list[start_idx + i * _shard_num];
   
    auto read_channel = _afs_client->open_r(channel_config, &err_no, ModelType::BATCH_SPARSE);
    char *end = NULL;
    auto& shard = _local_shards[i];
        
    //opt seq batch reader
    if (read_channel->is_seq_reader()) {
        uint64_t key = 0;
        std::string line_value;
        while (read_channel->read_line(key, line_value) == 0 && line_value.size() > 0) {
            if (FLAGS_pslib_open_strict_check) {
                if (!check_key(key, i, channel_config)) {
                    continue;
                }
            }
            if (upgrade_func.status == UpgradeStatus::UPGRADE) [[unlikely]] {
                upgrade_func.upgrade_for_binary(&line_value, line_value.length());
            }
            auto& value = shard.insert(key, line_value.size() / sizeof(float));
            memcpy(value.data(), line_value.data(), line_value.size());
        }
    } else {
        std::string line_data, new_line;
        while (read_channel->read_line(line_data) == 0 && line_data.size() > 1) {
            uint64_t key = std::strtoul(line_data.data(), &end, 10);
            auto& value = shard.insert(key, feature_value_size);
            int parse_size = 0;
            if (upgrade_func.status == UpgradeStatus::UPGRADE) [[unlikely]] {
                upgrade_func.upgrade_for_text(++end, &new_line);
                parse_size = _value_accessor->parse_from_string(new_line, value.data());
            } else {
                parse_size = _value_accessor->parse_from_string(++end, value.data());
            }
            shard.insert(key, parse_size);
        }
    }
    read_channel->close();
}
```

读出的数据，直接insert到sparse shard里面。

## 2 afs上的存储

### 2.1 batch model

#### 2.1.1 sparse

名称对比：

| 大类 | 小类    | abacus名字 | paddle名字 |
| ---- | ------- | ---------- | ---------- |
| lr   | embed   | lr_w       | embed_w    |
|      |         | lr_g       | embed_g    |
| mf   | embdedx | mf_w       | embedx_w   |
|      |         | mf_g       | embedx_g   |

名称与维度：

| 名称 | lr_w | lr_g | mf_w | mf_g |
| ---- | ---- | ---- | ---- | ---- |
| 维度 | 1    | 1    | 8    | 1    |

continous_value_model 取出来的embedding：

|             | show | click | lr_w  | mf_w  |
| ----------- | ---- | ----- | ----- | ----- |
| use_cvm     | √    | √     | 1 dim | 8 dim |
| not use_cvm | x    | x     | 1 dim | 8 dim |

paddle的存储格式，一行是这样的一条，`mf_w`维度可能是0，没有固定的顺序一说：

| feasign | uid  | unseen_days | delta_score | show | click | lr_w | lr_g | slot | mf_g | mf_w |
| ------- | ---- | ----------- | ----------- | ---- | ----- | ---- | ---- | ---- | ---- | ---- |
| 1       | 1    | 1           | 1           | 1    | 1     | 1    | 1    | 1    | 1    | 0、8 |

#### 2.1.2 dense

维度与名称，存储格式，一行是这样的一条，按dense参数的顺序排列下来：

| w    | avg_w | ada_d2sum | ada_g2sum | mom_velocity |
| ---- | ----- | --------- | --------- | ------------ |
| 1    | 1     | 1         | 1         | 1            |

以上是单个文件内的格式。从整体来看，每个server存储自己的那一份数据。

### 2.2 xbox

#### 2.2.1 feature

server将存储在自己身上的数据遍历取出，并写入afs，格式是sequence file。在写入之前，会使用`CommonXboxConverter`来将数据压缩。

#### 2.2.2 feature cache

server将存储在自己身上的cache数据遍历取出，并发送给coordinator，格式是pb的sequence file。在发送之前，会使用`PbXboxConverter`将其压缩。

#### 2.2.3 dnn plugin

这个是由worker保存的，调用了abacus、paddle的相应接口。结构如下：

- uint32 version
- lod information
- *tensor*
  - uint32 version
  - *tensor description*
    - int32 size
    - protobuf message
  - tensor data

## 3 `DownpourCtrFeatureValue`的设计

### 3.1 它是干啥用的？

方便读取value**数组**。注意到这里的数组，通过这么一组函数来使得对（没有明确定义的）数组变得像对很多个private成员变量的使用。

### 3.2 定义

``` cpp
struct DownpourCtrFeatureValue {
    /*
    float unseen_days;
    float delta_score;
    double show;
    double click;
    float embed_w;
    float embed_g2sum;
    float slot;
    uint64_t uid;
    float embedx_g2sum;
    std::vector<float> embedx_w; 
    */

    static int dim(int embedx_dim) {
        return 12 + embedx_dim;
    }
    static int dim_size(size_t dim, int embedx_dim) {
        return sizeof(float);
    }
    static int size(int embedx_dim) {
        return dim(embedx_dim) * sizeof(float);
    }
    static int unseen_days_index() {
        return 0;
    }
    static int delta_score_index() {
        return 1;
    }
    static int show_index() {
        return 2;
    } 
    static int click_index() {
        return 4;
    }
    static int embed_w_index() {
        return 6;
    }
    static int embed_g2sum_index() {
        return 7;
    }
    static int slot_index() {
        return 8;
    }
    static int uid_index() {
        return 9;
    }
    static int embedx_g2sum_index() {
        return 11;
    }
    static int embedx_w_index() {
        return 12;
    }
    static float& unseen_days(float* val) {
        return val[DownpourCtrFeatureValue::unseen_days_index()];
    }
    static float& delta_score(float* val) {
        return val[DownpourCtrFeatureValue::delta_score_index()];
    }
    static double& show(float* val) {
        return ((double*)(val + show_index()))[0];
    }
    static double& click(float* val) {
        return ((double*)(val + click_index()))[0];
    }
    static float& slot(float* val) {
        return val[DownpourCtrFeatureValue::slot_index()];
    }
    static float& embed_w(float* val) {
        return val[DownpourCtrFeatureValue::embed_w_index()];
    }
    static float& embed_g2sum(float* val) {
        return val[DownpourCtrFeatureValue::embed_g2sum_index()];
    }
    static float& embedx_g2sum(float* val) {
        return val[DownpourCtrFeatureValue::embedx_g2sum_index()];
    }
    static float* embedx_w(float* val) {
        return (val + DownpourCtrFeatureValue::embedx_w_index());
    }
};
```

### 3.3 看一下它的使用

以load sparse model解析文件内容时为例：

``` cpp
auto& value = shard.insert(key, feature_value_size);

parse_size = _value_accessor->parse_from_string(++end, value.data());

int DownpourCtrAccessor::parse_from_string(const std::string& str, float* value) {
    value[DownpourCtrFeatureValue::slot_index()] = -1;
    *(double*)(value + DownpourCtrFeatureValue::show_index()) = (double)data_buff_ptr[2];
}

std::string DownpourCtrAccessor::parse_to_string(const float* v, int param_size) {
    os << ((uint64_t*)(v + DownpourCtrFeatureValue::uid_index()))[0] << " ";   
       << (float)((double*)(v + DownpourCtrFeatureValue::show_index()))[0] << " ";
}
```

其中，`value`即前文所述的`DownpourFixedFeatureValue`，`end`指向字符串开头的指针。可以看到这个accessor的使用，利用index对value进行读取操作。

### 3.4 其他

注意到还有其他的value设计：`DownpourCtrPushValue`和`DownpourCtrPullValue`。用来访问push、pull的value，基本设计方式和此类似。

`DownpourCtrAccessor`还有其他的设计，提供了对sparse参数的维度、更新、save、load的操作。专门针对的就是value数组。

