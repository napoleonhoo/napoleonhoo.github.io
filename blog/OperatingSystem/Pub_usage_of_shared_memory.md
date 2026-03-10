# 共享内存（shared_memory）的使用

## 1 操作系统层面上的共享内存
### 什么是共享内存？

共享内存首先是**内存**，他的硬件存储介质实际上是一块RAM，和普通的内存一样。所以它也是易失性的（volatile），一旦关机重启，上面所存储的东西就会丢失。不会像硬盘这种可以永久保存下来，即非易失性的。

另一方面，共享内存是**共享**的。大家都知道内存一般来讲是各个进程私有的。共享内存的共享就在于可以和其他进程共享。所以，这也是为什么共享内存会成为进程间通信的一种方式。

由于共享内存的特殊性质，决定了其既不像内存又不会像硬盘，而是为它专门设计了一种文件系统，叫做tmpfs（**T**e**mp**orary **F**ile **S**ystem）。中文直译叫做“临时文件系统”，也有地方翻译的更直达本意，叫做“内存文件系统”。

### tmpfs简介

tmpfs设计的目的就是像可挂载的文件系统那样，但是数据是存储在易失性介质上，而不是持久性存储设备上。是在“逻辑文件系统层”上实现的，而非是直接利用硬件实现的。

你可以在你的机器上用`df -h`命令来看下机器上的tmpfs的情况，在我的机器上输出如下：

```
$ df -h
Filesystem      Size  Used Avail Use% Mounted on
devtmpfs         94G     0   94G   0% /dev
tmpfs            94G  3.1M   94G   1% /dev/shm
/dev/vda1        99G   18G   77G  20% /
/dev/vdb1       493G  240G  228G  52% /data
```

可以看到，目录`/dev/shm`就是tmpfs格式的。所以，对共享内存的操作都要在这个目录下面。

tmpfs的底层存储介质是RAM和硬盘。因为他是直接建立在虚拟内存系统的基础上的，因为虚拟内存同时使用了RAM和硬盘（准确说是其中的交换swap区），所以tmpfs也是。可以简单的认为，它是*搭建在内存上的文件系统*。

## 2 共享内存的读写

请看如下示例代码：

``` cpp
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const size_t kTestShmSize = 4096;
  
int ShmWrite(const char *filename) {
  int shm_fd = shm_open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return 1;
  }

  if (ftruncate(shm_fd, kTestShmSize) == -1) {
    return 2;
  }

  void *shm_ptr = mmap(0, kTestShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED) {
    return 3;
  }

  const char *message = "test";
  std::memcpy(shm_ptr, message, std::strlen(message) + 1);

  if (munmap(shm_ptr, kTestShmSize) == -1) {
    return 4;
  }

  if (close(shm_fd) == -1) {
    return 5;
  }

  return 0;
}

int ShmRead(const char *filename) {
  int shm_fd = shm_open(filename, O_RDONLY, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return 1;
  }

  void *shm_ptr = mmap(0, kTestShmSize, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (shm_ptr == MAP_FAILED) {
    return 2;
  }

  char buffer[kTestShmSize];
  std::memcpy(buffer, shm_ptr, kTestShmSize);
  std::cout << "Content in shared memory is: " << buffer << std::endl;

  if (munmap(shm_ptr, kTestShmSize) == -1) {
    return 3;
  }

  if (close(shm_fd) == -1) {
    return 4;
  }

  return 0;
}

int main() {
  const char *filename = "/dev/shm/test";
  int wr_res = ShmWrite(filename);
  if (wr_res != 0) {
    return wr_res;
  }

  int rd_res = ShmRead(filename);
  if (rd_res != 0) {
    return rd_res;
  }

  return 0;
}
```

1. 无论是读还是写，第一步都是调用`shm_open()`。类似于`open()`，第一个参数是文件名，第二个参数是打开的flag，如：`O_CREAT`代表如果没有就创建，`O_RDONLY`代表只读，`O_RDWR`代表读写；第三个参数是mode，代表创建的文件的读写执行权限。上面示例中使用的`S_IRUSR | S_IWUSR`，也可以在代码中写成数字：`0600`，即owner可读写。最终，如果调用成功，这个函数返回一个文件描述符（file descriptor）。
2. 对于新建的共享内存上的文件，下一步就要调用`ftruncate()`来设置想要的文件的大小。对于现存的文件，只是进行读写的话，可以不必进行这个步骤。
3. 调用`mmap()`将文件映射到内存。
4. 直接对`mmap()`返回的指针指向的内存进行读写。
5. 调用`munmap()`、`close()`等结束操作。

另外
1. `shm_unlink()`函数做的和`shm_open()`完全相反，即删除一个由`shm_open()`创建的对象。
2. 在调用`mmap()`之后就可以关闭这个fd了，不会影响后续的操作。

## 参考资料

1. [tmpfs - wikipedia](https://en.wikipedia.org/wiki/Tmpfs)
2. [shm_open(3) - Linux Manual Page](https://man7.org/linux/man-pages/man3/shm_open.3.html)
3. [chmod(2) - Linux Manual Page](https://man7.org/linux/man-pages/man2/chmod.2.html)
4. [open(2) — Linux manual page](https://man7.org/linux/man-pages/man2/open.2.html)