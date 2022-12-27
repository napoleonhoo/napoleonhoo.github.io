---
layout: default
---

# §Ch4 多线程相关

## ChannelObject

有：
```cpp
template<class T>
using Channel = std::shared_ptr<ChannelObject<T>>;
```
Channle是ChannelObject的指针，  
功能是一个加了锁的双向队列（deque），从最前read，从最后write。  
类似于Golang中的Channel。  

## WaitGroup
类似于Golang中的WaitGroup。

## Semaphore
借用系统库`<semaphore.h>`中`sem_*`等相关函数，实现了：
- `post()`：释放一个信号量。
- `wait() try_wait()`：获取一个信号量。

## Barrier
借用系统库`<pthread.h>`中的`pthread_barrier_*`相关变量与函数，实现了wait、reest等基本操作。

## ManagedThread
类ManagedThread对线程进行了一次封装。
- 构造时，便开启了一个thread，执行`run_thread`函数。
- `run_thread`函数，在进入时，会调用`Semaphore::wait`，执行`_func`，然后调用`Semaphore::post`。
- `start`函数，将传入的函数对象赋值给`_func`，然后调用`Semaphore::post`。

总体来讲，这个类启动了一个线程，然后不停地等待需要执行的函数。

## ThreadGroup
由多个ManageThread组成的线程池。
- 构造时，根据传入的线程数量，初始化`std::vector<ManagedThread> _threads`，并设置barrier的线程数量。
- `start`函数，让`_threads`中的每个都调用`ManagedThread::start`。并设置`parent_group()`为`this`。
- `barrier_wait`函数，调用私有成员变量`Barrier::wait`函数。

## parallel_run*
### parallel_run_barrier_wait
调用传入的`ThreadGroup`参数或调用thread_local的`parrent_group()`的`barrier_wait`函数。

## abacus_parallel
