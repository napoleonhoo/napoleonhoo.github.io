# §3.9 Table

## 1 路径

* include /leveldb/table.h
* table/table.cc

## 2 功能

* table是不可变、持久性的。
* table可以被多线程不需要外部同步地安全访问。

## 3 Table::Rep

```cpp
struct Table::Rep {
  ~Rep() {
    delete filter;
    delete[] filter_data;
    delete index_block;
  }

  Options options;
  Status status;
  RandomAccessFile* file;
  uint64_t cache_id;
  FilterBlockReader* filter;
  const char* filter_data;

  BlockHandle metaindex_handle;  // Handle to metaindex_block: saved from footer
  Block* index_block;
};
```

## 4 Table

### 4.1 构造函数等

*备注：构造函数是private的。*

```cpp
explicit Table(Rep* rep) : rep_(rep) {}

Table::~Table() { delete rep_; }
Table(const Table&) = delete;
Table& operator=(const Table&) = delete;
```

### 4.2 Open、ReadFilter、ReadMeta

#### 4.2.1 ReadBlock

相关函数（table/format.h table/format.h）：

ReadBlock的功能是根据block的结构（参见[Ch3.7 TableBuilder](./Ch3.7_TableBuilder.html)中的RawBlock），解码出block的data内容，并赋值给result（类型BlockContents的data成员）。

```cpp
// Read the block identified by "handle" from "file".  On failure
// return non-OK.  On success fill *result and return OK.
Status ReadBlock(RandomAccessFile* file, const ReadOptions& options,
                 const BlockHandle& handle, BlockContents* result);

Status ReadBlock(RandomAccessFile* file, const ReadOptions& options,
                 const BlockHandle& handle, BlockContents* result) {
  result->data = Slice();
  result->cachable = false;
  result->heap_allocated = false;

  // Read the block contents as well as the type/crc footer.
  // See table_builder.cc for the code that built this structure.
  size_t n = static_cast<size_t>(handle.size());
  char* buf = new char[n + kBlockTrailerSize];
  Slice contents;
  Status s = file->Read(handle.offset(), n + kBlockTrailerSize, &contents, buf);
  if (!s.ok()) {
    delete[] buf;
    return s;
  }
  if (contents.size() != n + kBlockTrailerSize) {
    delete[] buf;
    return Status::Corruption("truncated block read");
  }

  // Check the crc of the type and the block contents
  const char* data = contents.data();  // Pointer to where Read put the data
  if (options.verify_checksums) {
    const uint32_t crc = crc32c::Unmask(DecodeFixed32(data + n + 1));
    const uint32_t actual = crc32c::Value(data, n + 1);
    if (actual != crc) {
      delete[] buf;
      s = Status::Corruption("block checksum mismatch");
      return s;
    }
  }

  switch (data[n]) {
    case kNoCompression:
      if (data != buf) {
        // File implementation gave us pointer to some other data.
        // Use it directly under the assumption that it will be live
        // while the file is open.
        delete[] buf;
        result->data = Slice(data, n);
        result->heap_allocated = false;
        result->cachable = false;  // Do not double-cache
      } else {
        result->data = Slice(buf, n);
        result->heap_allocated = true;
        result->cachable = true;
      }

      // Ok
      break;
    case kSnappyCompression: {
      size_t ulength = 0;
      if (!port::Snappy_GetUncompressedLength(data, n, &ulength)) {
        delete[] buf;
        return Status::Corruption("corrupted compressed block contents");
      }
      char* ubuf = new char[ulength];
      if (!port::Snappy_Uncompress(data, n, ubuf)) {
        delete[] buf;
        delete[] ubuf;
        return Status::Corruption("corrupted compressed block contents");
      }
      delete[] buf;
      result->data = Slice(ubuf, ulength);
      result->heap_allocated = true;
      result->cachable = true;
      break;
    }
    default:
      delete[] buf;
      return Status::Corruption("bad block type");
  }

  return Status::OK();
}
```

#### 4.2.2 ReadFilter

ReadFilter主要流程如下：

* 是要根据输入的filter\_handle\_value调用DecodeFrom得到filter\_handle。
* 调用ReadBlock，得到filterBlock中的内容（block）。
* 如果block是分配在堆上的，将block中的data数据赋给`req_->filter_data`。
* 构造一个FilterBlockReader对象赋给`req_->filter`。

```cpp
void Table::ReadFilter(const Slice& filter_handle_value) {
  Slice v = filter_handle_value;
  BlockHandle filter_handle;
  if (!filter_handle.DecodeFrom(&v).ok()) {
    return;
  }

  // We might want to unify with ReadBlock() if we start
  // requiring checksum verification in Table::Open.
  ReadOptions opt;
  if (rep_->options.paranoid_checks) {
    opt.verify_checksums = true;
  }
  BlockContents block;
  if (!ReadBlock(rep_->file, opt, filter_handle, &block).ok()) {
    return;
  }
  if (block.heap_allocated) {
    rep_->filter_data = block.data.data();  // Will need to delete later
  }
  rep_->filter = new FilterBlockReader(rep_->options.filter_policy, block.data);
}
```

#### 4.2.3 ReadMeta

ReadMeta的主要流程如下：

* 根据输入的footer中含有的metaIndexHandle的内容，读出metaIndexHandle的内容（contents）。
* 根据metaIndexBlock的key的固定内容（“filter.“+filterPolicy内容），找到相应的value。
* 找到的value即为filterBlock的Handle编码之后的内容，然后利用其调用ReadFilter。

```cpp
void Table::ReadMeta(const Footer& footer) {
  if (rep_->options.filter_policy == nullptr) {
    return;  // Do not need any metadata
  }

  // TODO(sanjay): Skip this if footer.metaindex_handle() size indicates
  // it is an empty block.
  ReadOptions opt;
  if (rep_->options.paranoid_checks) {
    opt.verify_checksums = true;
  }
  BlockContents contents;
  if (!ReadBlock(rep_->file, opt, footer.metaindex_handle(), &contents).ok()) {
    // Do not propagate errors since meta info is not needed for operation
    return;
  }
  Block* meta = new Block(contents);

  Iterator* iter = meta->NewIterator(BytewiseComparator());
  std::string key = "filter.";
  key.append(rep_->options.filter_policy->Name());
  iter->Seek(key);
  if (iter->Valid() && iter->key() == Slice(key)) {
    ReadFilter(iter->value());
  }
  delete iter;
  delete meta;
}
```

#### 4.2.4 Open

* 读出文件file中的footer内容，并解码内容到footer\_input。
* 根据footer中的indexHandle，读出indexBlock的内容（index\_block\_contents），并根据此，构造出indexBlock（Block对象）。
* 赋值req\_中的metaindex\_index、index\_block。
* 调用ReadMeta。

*备注：*

* *从输入的file中读出table。*
* *table是输出参数。*

```cpp
// Attempt to open the table that is stored in bytes [0..file_size)
// of "file", and read the metadata entries necessary to allow
// retrieving data from the table.
//
// If successful, returns ok and sets "*table" to the newly opened
// table.  The client should delete "*table" when no longer needed.
// If there was an error while initializing the table, sets "*table"
// to nullptr and returns a non-ok status.  Does not take ownership of
// "*source", but the client must ensure that "source" remains live
// for the duration of the returned table's lifetime.
//
// *file must remain live while this Table is in use.
static Status Open(const Options& options, RandomAccessFile* file,
                   uint64_t file_size, Table** table);

Status Table::Open(const Options& options, RandomAccessFile* file,
                   uint64_t size, Table** table) {
  *table = nullptr;
  if (size < Footer::kEncodedLength) {
    return Status::Corruption("file is too short to be an sstable");
  }

  char footer_space[Footer::kEncodedLength];
  Slice footer_input;
  Status s = file->Read(size - Footer::kEncodedLength, Footer::kEncodedLength,
                        &footer_input, footer_space);
  if (!s.ok()) return s;

  Footer footer;
  s = footer.DecodeFrom(&footer_input);
  if (!s.ok()) return s;

  // Read the index block
  BlockContents index_block_contents;
  ReadOptions opt;
  if (options.paranoid_checks) {
    opt.verify_checksums = true;
  }
  s = ReadBlock(file, opt, footer.index_handle(), &index_block_contents);

  if (s.ok()) {
    // We've successfully read the footer and the index block: we're
    // ready to serve requests.
    Block* index_block = new Block(index_block_contents);
    Rep* rep = new Table::Rep;
    rep->options = options;
    rep->file = file;
    rep->metaindex_handle = footer.metaindex_handle();
    rep->index_block = index_block;
    rep->cache_id = (options.block_cache ? options.block_cache->NewId() : 0);
    rep->filter_data = nullptr;
    rep->filter = nullptr;
    *table = new Table(rep);
    (*table)->ReadMeta(footer);
  }

  return s;
}
```

### 4.3 NewIterator

返回一个TwoLevelIterator。

```cpp
Iterator* Table::NewIterator(const ReadOptions& options) const {
  return NewTwoLevelIterator(
      rep_->index_block->NewIterator(rep_->options.comparator),
      &Table::BlockReader, const_cast<Table*>(this), options);
}
```

### 4.4 BlockReader

BlockReader将indexIteratorValue（blockHandle）转变为对相应的区域的iterator。主要流程如下：

* 将输入的arg转换为Table\*类型。
* 将出传入的index\_value解析到handle（BlockHandle对象）。
* 构造在block\_cache中要查找的key。
* 当存在cache时，用cacheID（8byte）+indexHandle的offset（8Byte）组成key，在cache里面寻找。
  * 找得到时，对应的value就是赋值给block（类型Block\*）。
  * 找不到时，直接从文件中读取block到contents（BlockContents对象），并用其初始化block，最后将block放入cache中，并注册deleter为DeleteCachedBlock。
* 当不存在cache时，直接从文件中读取block到contents（BlockContents对象），并用其初始化block。
* 获取block的iterator。
* 当cacheHandle存在时，iter调用RegisterCleanup，函数为DeleteBlock。
* 当cacheHandle不存在时，iter调用RegisterCleanup，函数为ReleaseBlock。

```cpp
static Iterator* BlockReader(void*, const ReadOptions&, const Slice&);

// Convert an index iterator value (i.e., an encoded BlockHandle)
// into an iterator over the contents of the corresponding block.
Iterator* Table::BlockReader(void* arg, const ReadOptions& options,
                             const Slice& index_value) {
  Table* table = reinterpret_cast<Table*>(arg);
  Cache* block_cache = table->rep_->options.block_cache;
  Block* block = nullptr;
  Cache::Handle* cache_handle = nullptr;

  BlockHandle handle;
  Slice input = index_value;
  Status s = handle.DecodeFrom(&input);
  // We intentionally allow extra stuff in index_value so that we
  // can add more features in the future.

  if (s.ok()) {
    BlockContents contents;
    if (block_cache != nullptr) {
      char cache_key_buffer[16];
      EncodeFixed64(cache_key_buffer, table->rep_->cache_id);
      EncodeFixed64(cache_key_buffer + 8, handle.offset());
      Slice key(cache_key_buffer, sizeof(cache_key_buffer));
      cache_handle = block_cache->Lookup(key);
      if (cache_handle != nullptr) {
        block = reinterpret_cast<Block*>(block_cache->Value(cache_handle));
      } else {
        s = ReadBlock(table->rep_->file, options, handle, &contents);
        if (s.ok()) {
          block = new Block(contents);
          if (contents.cachable && options.fill_cache) {
            cache_handle = block_cache->Insert(key, block, block->size(),
                                               &DeleteCachedBlock);
          }
        }
      }
    } else {
      s = ReadBlock(table->rep_->file, options, handle, &contents);
      if (s.ok()) {
        block = new Block(contents);
      }
    }
  }

  Iterator* iter;
  if (block != nullptr) {
    iter = block->NewIterator(table->rep_->options.comparator);
    if (cache_handle == nullptr) {
      iter->RegisterCleanup(&DeleteBlock, block, nullptr);
    } else {
      iter->RegisterCleanup(&ReleaseBlock, block_cache, cache_handle);
    }
  } else {
    iter = NewErrorIterator(s);
  }
  return iter;
}

static void DeleteBlock(void* arg, void* ignored) {
  delete reinterpret_cast<Block*>(arg);
}

static void DeleteCachedBlock(const Slice& key, void* value) {
  Block* block = reinterpret_cast<Block*>(value);
  delete block;
}

static void ReleaseBlock(void* arg, void* h) {
  Cache* cache = reinterpret_cast<Cache*>(arg);
  Cache::Handle* handle = reinterpret_cast<Cache::Handle*>(h);
  cache->Release(handle);
}
```

### 4.3 InternalGet

InternalGet要先找到一个key，先利用indexBlock，再利用filter，如果找得到就对这个key和value执行函数`handle_result`。

```cpp
// Calls (*handle_result)(arg, ...) with the entry found after a call
// to Seek(key).  May not make such a call if filter policy says
// that key is not present.
Status InternalGet(const ReadOptions&, const Slice& key, void* arg,
                   void (*handle_result)(void* arg, const Slice& k,
                                         const Slice& v));

Status Table::InternalGet(const ReadOptions& options, const Slice& k, void* arg,
                          void (*handle_result)(void*, const Slice&,
                                                const Slice&)) {
  Status s;
  Iterator* iiter = rep_->index_block->NewIterator(rep_->options.comparator);
  iiter->Seek(k);
  if (iiter->Valid()) {
    Slice handle_value = iiter->value();
    FilterBlockReader* filter = rep_->filter;
    BlockHandle handle;
    if (filter != nullptr && handle.DecodeFrom(&handle_value).ok() &&
        !filter->KeyMayMatch(handle.offset(), k)) {
      // Not found
    } else {
      Iterator* block_iter = BlockReader(this, options, iiter->value());
      block_iter->Seek(k);
      if (block_iter->Valid()) {
        (*handle_result)(arg, block_iter->key(), block_iter->value());
      }
      s = block_iter->status();
      delete block_iter;
    }
  }
  if (s.ok()) {
    s = iiter->status();
  }
  delete iiter;
  return s;
}
```

