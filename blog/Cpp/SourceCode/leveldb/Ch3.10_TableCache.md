# §Ch3.10 TableCache

## 1 路径

* db/table_cache.h
* db/table_cache.cc

## 2 功能

## 3 成员变量

```cpp
Env* const env_;
const std::string dbname_;
const Options& options_;
Cache* cache_
```

## 4 成员函数

### 4.1 构造、析构函数

```cpp
TableCache::TableCache(const std::string& dbname, const Options& options,
                       int entries)
    : env_(options.env),
      dbname_(dbname),
      options_(options),
      cache_(NewLRUCache(entries)) {}

TableCache::~TableCache() { delete cache_; }
```

### 4.2 FindTable

FindTable的主要流程如下：

* 用输入的file\_number组成key，在cache\_中寻找是否存在，并赋给handle。
* 如果handle为nullptr，用dbname\_、file\_number调用TableFileName组成fname，并调用env\_的NewRandomAccessFile打开。
* 调用Table::Open打开，赋值变量table。
* 赋值tf（TableAndFile类型声明见下）及file、table变量，并将其插入cache\_，且声明deleter为DeleteEntry（定义见下，非成员函数）。

```cpp
struct TableAndFile {
  RandomAccessFile* file;
  Table* table;
};

static void DeleteEntry(const Slice& key, void* value) {
  TableAndFile* tf = reinterpret_cast<TableAndFile*>(value);
  delete tf->table;
  delete tf->file;
  delete tf;
}
```

FindTable定义及声明如下：

```cpp
Status FindTable(uint64_t file_number, uint64_t file_size, Cache::Handle**);

Status TableCache::FindTable(uint64_t file_number, uint64_t file_size,
                             Cache::Handle** handle) {
  Status s;
  char buf[sizeof(file_number)];
  EncodeFixed64(buf, file_number);
  Slice key(buf, sizeof(buf));
  *handle = cache_->Lookup(key);
  if (*handle == nullptr) {
    std::string fname = TableFileName(dbname_, file_number);
    RandomAccessFile* file = nullptr;
    Table* table = nullptr;
    s = env_->NewRandomAccessFile(fname, &file);
    if (!s.ok()) {
      std::string old_fname = SSTTableFileName(dbname_, file_number);
      if (env_->NewRandomAccessFile(old_fname, &file).ok()) {
        s = Status::OK();
      }
    }
    if (s.ok()) {
      s = Table::Open(options_, file, file_size, &table);
    }

    if (!s.ok()) {
      assert(table == nullptr);
      delete file;
      // We do not cache error results so that if the error is transient,
      // or somebody repairs the file, we recover automatically.
    } else {
      TableAndFile* tf = new TableAndFile;
      tf->file = file;
      tf->table = table;
      *handle = cache_->Insert(key, tf, 1, &DeleteEntry);
    }
  }
  return s;
}
```

### 4.2 NewIterator

NewIterator主要流程如下：

- 先调用FindTable，来找到file\_number对应的文件，并赋值handle。
- 利用handle，从cache\_里找到相应的table，建立一个table的iterator，并调用RegisterCleanup，CleanupFunction为UnrefEntry（定义如下，非成员函数），将tableptr指向table。

```cpp
static void UnrefEntry(void* arg1, void* arg2) {
  Cache* cache = reinterpret_cast<Cache*>(arg1);
  Cache::Handle* h = reinterpret_cast<Cache::Handle*>(arg2);
  cache->Release(h);
}
```

NewIterator的定义和声明如下：

```cpp
// Return an iterator for the specified file number (the corresponding
// file length must be exactly "file_size" bytes).  If "tableptr" is
// non-null, also sets "*tableptr" to point to the Table object
// underlying the returned iterator, or to nullptr if no Table object
// underlies the returned iterator.  The returned "*tableptr" object is owned
// by the cache and should not be deleted, and is valid for as long as the
// returned iterator is live.
Iterator* NewIterator(const ReadOptions& options, uint64_t file_number,
                      uint64_t file_size, Table** tableptr = nullptr);

Iterator* TableCache::NewIterator(const ReadOptions& options,
                                  uint64_t file_number, uint64_t file_size,
                                  Table** tableptr) {
  if (tableptr != nullptr) {
    *tableptr = nullptr;
  }

  Cache::Handle* handle = nullptr;
  Status s = FindTable(file_number, file_size, &handle);
  if (!s.ok()) {
    return NewErrorIterator(s);
  }

  Table* table = reinterpret_cast<TableAndFile*>(cache_->Value(handle))->table;
  Iterator* result = table->NewIterator(options);
  result->RegisterCleanup(&UnrefEntry, cache_, handle);
  if (tableptr != nullptr) {
    *tableptr = table;
  }
  return result;
}
```

### 4.3 Get

Get的主要流程如下：

- 利用传入的file\_number调用FindTable，并赋值handle。
- 从cache\_中找到相应的table。
- 调用table的InternalGet。
- cache\_调用Release之前的handle。

```cpp
// If a seek to internal key "k" in specified file finds an entry,
// call (*handle_result)(arg, found_key, found_value).
Status Get(const ReadOptions& options, uint64_t file_number,
           uint64_t file_size, const Slice& k, void* arg,
           void (*handle_result)(void*, const Slice&, const Slice&));

Status TableCache::Get(const ReadOptions& options, uint64_t file_number,
                       uint64_t file_size, const Slice& k, void* arg,
                       void (*handle_result)(void*, const Slice&,
                                             const Slice&)) {
  Cache::Handle* handle = nullptr;
  Status s = FindTable(file_number, file_size, &handle);
  if (s.ok()) {
    Table* t = reinterpret_cast<TableAndFile*>(cache_->Value(handle))->table;
    s = t->InternalGet(options, k, arg, handle_result);
    cache_->Release(handle);
  }
  return s;
}
```

### 4.4 Evict

Evict将file\_number代表的文件从cache\_中Erase。

```cpp
// Evict any entry for the specified file number
void Evict(uint64_t file_number);

void TableCache::Evict(uint64_t file_number) {
  char buf[sizeof(file_number)];
  EncodeFixed64(buf, file_number);
  cache_->Erase(Slice(buf, sizeof(buf)));
}
```

