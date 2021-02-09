# §2.14 Others

## 1 Random

* 路径：util/random.h
* 功能：LevelDB自己实现的伪随机数生成器。

## 2 Hash

* 路径：util/hash.h util/hash.cc
* 功能：LevelDB自己实现的类似Murmur Hash的功能。

## 3 CRC32

* 路径：util/crc32c.h util/crc32c.cc
* 功能：进行CRC32的循环冗余校验。

## 4 MutexLock

* 路径：util/mutexlock.h
* 功能：在构造时对mutex进行lock，在析构时对mutex进行unlock。

## 5 NoDestructor

* 路径：util/no_destructor.h
* 功能：根据注释，这是一个对永远不会被析构的类的封装，这一般用在函数级别的静态变量。

* 主要代码如下：

```cpp
template <typename InstanceType>
class NoDestructor {
 public:
  template <typename... ConstructorArgTypes>
  explicit NoDestructor(ConstructorArgTypes&&... constructor_args) {
    static_assert(sizeof(instance_storage_) >= sizeof(InstanceType),
                  "instance_storage_ is not large enough to hold the instance");
    static_assert(
        alignof(decltype(instance_storage_)) >= alignof(InstanceType),
        "instance_storage_ does not meet the instance's alignment requirement");
    new (&instance_storage_)
        InstanceType(std::forward<ConstructorArgTypes>(constructor_args)...);
  }
  ~NoDestructor() = default;
  NoDestructor(const NoDestructor&) = delete;
  NoDestructor& operator=(const NoDestructor&) = delete;
  InstanceType* get() {
    return reinterpret_cast<InstanceType*>(&instance_storage_);
  }
 private:
  typename std::aligned_storage<sizeof(InstanceType),
                                alignof(InstanceType)>::type instance_storage_;
};
```

## 6 Filename

* 路径：db/filename.h db/filename.cc

* 功能：组成和解析leveldb所需要的各种文件名。

  * LogFileName: dbname/[0-9]+\\.log
  * TableFileName: dbname/[0-9]+\\.ldb
  * SSTTableFileName: dbname/[0-9]+\\.sst
  * DescriptorFileName: dbname/MANIFEST-[0-9]+
  * CurrentFileName: dbname/CURRENT
  * LockFileName:  dbname/LOCK
  * TempFileName:
  * InfoLogFileName: dbname/LOG
  * OldInfoLogFileName: dbname/LOG.old
  * TempFileName: dbname/[0-9]+\\.sst

  备注：上述涉及到数字的部分要符合`uint64_t`的要求。