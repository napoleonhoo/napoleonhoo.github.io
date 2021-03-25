# §2.7 EnvPosix

## 1 路径

* util/env_posix.cc

## 2 功能

* Env在POSIX上的实现

## 3 类

### 3.1 Limiter

#### 3.1.1 功能

Limiter是为了限制资源的使用，避免资源被耗尽。目前是为了限制只读文件和mmap文件的使用，所以我们不会用尽文件描述符、虚拟内存或因为特别大的数据库而遇到内核性能问题。

#### 3.1.2 主要成员变量

* `std::atomic<int> acuqires_allowed_；`可用的资源的总数，由于其不属于任何其它的类，所以可以使用`std::memory_order_relaxed`安全地访问。

#### 3.1.3 主要成员函数

* 构造函数：传入一个最大的资源总量来初始化`acuqires_allowed_`。
* 不允许复制构造和赋值。
* `bool Acuqire()`。获取一个资源，实际上是可用资源总数减一。当无可用资源时，返回`false`；否则返回`true`。
* `void Release()`。释放一个资源，实际上是可用资源总数加一。

### 3.2 PosixSequentialFile

#### 3.2.1 功能

对Env中SequentialFile在POSIX上的实现，并且声明了final关键字，即不能被继承。

#### 3.2.2 主要成员变量

* `const int fd_;`文件描述符。
* `const std::string filename_;`文件名

#### 3.2.3 主要成员函数

* `Status Read(size_t n, Slice* result, char* scratch) override;`从文件中试图读出`n`个字节，放到`scratch`中，`result`指向了`scratch`。主要使用到的库函数：`read`。
* `Status Skip(uint64_t n) override;`跳过文件`n`个字节。主要使用的库函数：`lseek`。

### 3.3 PosixRandomAccessFile

#### 3.3.1 功能

对Env中RandomAccessFile在POSIX上的实现，并且声明了final关键字，即不能被继承。

#### 3.3.2 主要成员变量

* `const bool has_permanent_fd_;`是否拥有这个fd，当没有拥有时，在每次Read时，需要先打开再关闭。
* `const int fd_;`文件描述符，当`has_permanent_fd_`为`false`时，其为-1。
* `Limiter* const fd_limiter_;`
* `const std::string filename_;`

#### 3.3.3 主要成员函数

* 构造函数`PosixRandomAccessFile(std::string filename, int fd, Limiter* fd_limiter);`：用传入的`fd_limiter`赋值给`fd_limiter_`，进行`Acquire()`操作，当成功时，`has_permanent_fd_`为`true`，这时传入的`fd`被赋值给`fd_`。
* 析构函数：当`has_permanent_fd_`为`true`时，需要用`fd_limiter_`进行`Release()`。
* Read：当不拥有这个fd时，需要每次读取时都利用*文件名*打开这个文件；拥有这个fd时，则直接利用fd读取文件。。这里调用了库函数`pread`。

### 3.4 PosixMmapReadableFile

#### 3.4.1 功能

继承自RandomAccessFile，并且声明了final关键字，即不能被继承。

#### 3.4.2 主要成员变量

* `char* const mmap_base_;`指向通过`mmap`获得的内存地址。
* `const size_t length_;`
* `Limiter* const mmap_limiter_;`
* `const std::string filename_;`

#### 3.4.3 主要成员函数

* 构造函数：赋值相关成员变量。
* 析构函数：调用`munmap`，并且使用`mmap_limiter_`调用`Release()`。
* Read：直接使用传入的`offset`参数和`mmap_base_`直接读取响应地址的数据。

### 3.5 PosixWritableFile

#### 3.5.1 功能

继承自WritableFile，并且声明了final关键字，即不能被继承。

#### 3.5.2 主要成员变量

* `char buf_[kWritableFileBufferSize];`缓存数据，`buf_[0, pos_ - 1]`的数据是准备写入`fd_`中的数据。
* `size_t pos_;`
* `int fd_;`
* `const bool is_manifest_;`文件是否是MANIFEST文件。
* `const std::string filename_;`文件名。
* `const std::string dirname_;`文件夹名。

#### 3.5.3 相关变量

* `constexpr const size_t kWritableFileBufferSize = 65536;`

#### 3.5.4 主要成员函数

* `private: static Slice Basename(const std::string& filename);`以最后一个“/”作为分割，得到文件名。如果找不到“/”，则返回`filename`作为Basename。
* `private: static bool IsMnifest(const std::string& filename);`对传入的`filename`取`Basename`，如果是以“MANIFEST”开头，则返回true。
* `private: static std::string Dirname(const std::string& filename;`以最后一个“/”作为分割，得到文件夹名。如果找不到“/”，则返回“.“。
* 构造函数：传入`std::string filename`和`int fd`，赋值给`filename_`、`fd_`，调用`IsManifest`赋值`is_manifest_`，调用`Dirname`赋值`dirname_`，赋值`pos_`为0。
* 析构函数：调用`Close();`。
* `Status WriteUnbuffered(const char* data, size_t size);`当`size`大于0是，循环调用库函数：`write`，`data`写入`fd_`中。
* `Status FlushBuffer();`利用`buf_`，`pos_`调用`WriteUnbffered()`，`pos_`归0，即清空`buf_`。
* `Staus Append(const Slice& data) override;`
  * 将`data_`中的数据使用`std::memcpy`放入`buf_`中，如果`buf_`放得下，直接返回成功。否则调用`FlushBuffer()`，写一次。
  * 如果剩下的数据小于`kWritableFileBufferSize`，则将其通过`std::memcpy`写入`buf_`，并返回。否则，即`buf_`里装不下的时候，直接调用`WriteUnbuffered`，即直接写入`fd_`。
* `Status Close() override;`调用`FlushBuffer`，即将现在`buf_`中的数据写一次，并调用库函数`::close()`关闭`fd_`。
* `Status Flush() override;`调用`FlushBuffer();`
* `static Status SyncFd(int fd, const std::string& fd_path);`对输入参数`fd`调用库函数`::sync(fd)`。
* `Status SyncDirIfManifest();`只针对MANIFEST文件进行sync，首先对`dirname_`调用`::open`，flag为`O_RDONLY | kOpenBaseFlags`，其中`kOpenBaseFlags`为`O_CLOEXEC`。然后调用`SyncFd()`。
* `Status Sync() override;`
  * 先调用`SyncDirIfManifest()`，如果失败，则返回。
  * 调用`FlushBuffer()`，如果失败，则返回。
  * 返回`SyncFd(fd_, filename_);`。

### 3.6 PosixFileLock

#### 3.6.1功能

实现了FileLock。

#### 3.6.2 主要成员变量

* `const int fd_;`
* `const std::string filename_;`

#### 3.6.3 主要成员函数

* 构造函数：赋值成员变量。
* `int fd() const;`返回`fd_`。
* `const std::string& filename() const;`返回`filename_`。

### 3.7 PosixLockTable

#### 3.7.1 功能

* 记录被`PosixEnv::LockFile`所lock的文件
* 不使用`fcnl(F_SETLK)`是因为它并不能提供同一进程内多用户的保护。
* 线程安全。

#### 3.7.2 主要成员变量

* `port::Mutex mu_;`
* `std::set<std::string> locked_files_ GUARDED_BY(mu_);`

#### 3.7.3 主要成员函数

* `bool Insert(const std::string& fname) LOCKS_EXCLUDED(mu_);`
* `void Remove(const std::string& fname) LOCKS_EXCLUDED(mu_);`

#### 3.7.4 备注

在clang编译器下：

* 宏`GUARDED_BY(mu_)`展开至`__attribute__((guareded_by(mu_)))`。
* 宏`LOCKS_EXCLUDED(mu_)`展开至`__attribute((locks_excluded(mu_)))`。

### 3.8 PosixEnv

#### 3.8.1 功能

Env在POSIX上的实现。

#### 3.8.2 主要成员变量

* `port::Mutex background_work_mutex_;`
* `port::CondVar background_work_cv_ GUARDED_BY(background_work_mutex_);`
* `bool started_background_thread_ GUARDED_BY(background_work_mutex_);`
* `std::queue<BackgroundWorkItem> background_work_queue_ GUARDED_BY(background_work_mutex_);`
* `PosixLockTable locks_;`
* `Limiter mmap_limiter_;`
* `Limiter fd_limiter_;`

#### 3.8.3 主要成员函数

* 构造函数：使用`background_work_mutex_`初始化`background_work_cv_`，`started_background_thread_`赋值为false，调用`MaxMmaps()`初始化`mmap_limiter_`，调用`MaxOpenFiles()`初始化`fd_limiter_`。
* 析构函数：被析构是错误的，将错误信息写入`stderr`，并中止进程。调用了函数`std::fwrite`、`std::abort`。

* `NewSequentialFile`返回一个新的`PosixSequentialFile`对象。
* `NewRandomAccessFile`，当`mmap_limiter_`达到上限时，返回一个`PosixRandomAcceFile`对象；否则，调用`mmap`，并返回一个`PosixMmapReadableFile`对象。备注：通过调用库函数`stat`返回了文件的大小。
* `NewWritableFile`返回一个`PosixWritableFile`对象。
* `NewAppendableFile`返回一个`PosixWritableFile`对象。
* `FileExists`调用库函数`access`返回一个文件是否存在。
* `GetChildren`调用库函数`oepndir`打开一个文件夹，然后调用库函数`readdir`读取文件夹，调用库函数`closedir`关闭文件夹。返回读取到的文件夹的内容。
* `RemoveFile`调用库函数`unlink`。
* `CreateDir`调用库函数`mkdir`。
* `RemoveDir`调用库函数`rmdir`。
* `GetFileSize`调用库函数`stat`。
* `RenameFile`调用函数`std::rename`。
* `LockFile`调用库函数`open`将文件打开，把`filename`记录在 `FileLock`里面，使用`LockOrUnlock`将文件锁定，返回`PosixFileLock`对象。
* `UnlockFile`调用`LockOrUnlock`对文件进行解锁，将其从`FileLock`中移除。
* `NewLogger`调用C库函数`open`打开 一个文件，并调用C库函数`fdopen`将fd和数据流关联起来，返回一个PosixLogger对象。
* `NowMicros`调用库函数`gettimeofday`。
* `SleepForMicroseconds`调用函数`std::this_thread::sleep_for()`。
* `StartThread`调用`std::thread`运行一个函数，并将线程`detach`。
* `private: void BackgroundThreadMain`循环地（`while (true)`）检查`background_work_queue_`是否有可执行的任务（函数），如果无，则调用`background_work_cv_.Wait()`；否则，执行这个任务。这一段函数是用`background_work_mutex_`保护起来的。<u>总之，这是后台线程运行的主函数。</u>
* `private: static void BackgroundThreadEntryPoint(PosixEnv* env);`返回`env->BackgroundThreadMain();`
* `Schedule`。检查`started_background_thread_`是否为false，当为false时，意味着后台线程未开启，则将`started_background_thread_`设置为true，并通过`std::thread`来运行`PosixEnv::BackgroundThreadEntryPoint`，并将线程`detach`。如果`background_work_queu_`是空，则用`background_work_cv_`进行`Signal`。最后将传入的函数和函数参数放入`background_work_queue_`中。这一段函数使用`background_work_mutex_`进行保护的 。<u>总之，这个函数用来将任务（函数）添加至后台进程。</u>

#### 3.8.4 内部类

##### 3.8.4.1 private: BackgroundWorkItem

存储`Schedule()`调用过程中，可能会用到的work item，构造函数传入函数和参数，并保存起来。

### 3.9 SingletonEnv

#### 3.9.1 功能

#### 3.9.2 模板参数

* EnvType，Env类型，如PosixEnv

#### 3.9.3 主要成员变量

* `typename std::aligned_storage<sizeof(EnvType), alignof(EnvType)::type env_storage_;`
* `static std::atomic<bool> env_initialized_;`env，即`env_storage_`，是否被初始化。

#### 3.9.4 主要相关函数

* 构造函数：`env_storage_`赋值为true，并在`env_storage_`上调用`EnvType`的构造函数（placement new）。
* 默认析构函数，不允许复制构造和赋值。
* `Env* env()`返回`env_storage_`。

## 4 相关内容

### 4.1 主要相关变量

* `int g_mmap_limit;`可以Mmap的区域的个数，64位机器上1000个区域，32位机器上无区域。在运行环境下，这个相当于一个常量。
* `int g_open_read_only_file_limit;`只读文件的个数上限，初始值为-1。

### 4.2 主要相关函数

* `int MaxMmaps();`返回`g_mmap_limit`。
* `int MaxOpenFiles();`返回只读文件保持打开的上限。具体步骤如下：
  * 当`g_open_read_only_file_limit`大于0时，则表示其已经初始化过了，直接返回这个值。否则，
  * 调用库函数`getrlimit`，得到目前系统设置的上限值。
  * 调用失败时，将其默认值设为50；否则当返回的软限值为无限时，其设置为`std::numeric_limits<int>::max()`；否则，其设置为软限值的$${\frac{1}{5}}$$，即20%。
  * 返回`g_open_read_only_file_limit`。
* `int LockOrUnlock(int fd, bool lock);`根据传入参数`lock`的true或false，调用库函数`cntl`对文件`fd`进行加锁或解锁。

### 4.3 相关定义

* `using PosixDefaultEnv = SingletonEnv<PosixEnv>;`



