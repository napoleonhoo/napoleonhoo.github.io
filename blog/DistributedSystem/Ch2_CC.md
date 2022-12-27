---
layout: default
---
# 集合通信
英文：Collective Communication  
代表作品：
- MPI（Message Passing Interface），是一个协议。
- gloo，facebook对集合通信的实现。

Reference：
- https://scc.ustc.edu.cn/zlsc/cxyy/200910/MPICH/
- https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node64.html
- https://www.open-mpi.org/doc/current/man3/MPI_Allreduce.3.php
# 主要的操作及释义
## 图示主要操作

![](./CC.png)

## Barrier Synchronization
```c
MPI_BARRIER(comm) 
　IN　　comm　　通信子(句柄)
int MPI_Barrier(MPI_Comm comm)
```
MPI_BARRIER阻塞所有的调用者直到所有的组成员都调用了它,仅当所有的组成员都进入了这个调用后,各个进程中这个调用才可以返回.
## Broadcast
```c
MPI_BCAST(buffer,count,datatype,root,comm) 
　IN/OUT　buffer　　  通信消息缓冲区的起始地址(可变)
　IN　　　 count　  　 通信消息缓冲区中的数据个数(整型) 
　IN 　　　datatype 　通信消息缓冲区中的数据类型(句柄) 
　IN　　　 root　  　　发送广播的根的序列号(整型) 
　IN 　　　comm   　　通信子(句柄) 
int MPI_Bcast(void* buffer,int count,MPI_Datatype datatype,int root, MPI_Comm comm)
```
MPI_BCAST是从一个序列号为root的进程将一条消息广播发送到组内的所有进程,包括它本身在内.调用时组内所有成员都使用同一个comm和root,其结果是将根的通信消息缓冲区中的消息拷贝到其他所有进程中去.
## Gather
```c
MPI_GATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root , comm)
　IN　sendbuf   　发送消息缓冲区的起始地址(可变)
　IN　sendcount 　发送消息缓冲区中的数据个数(整型)
　IN　sendtype　  发送消息缓冲区中的数据类型(句柄) 
　OUT recvbuf 　　接收消息缓冲区的起始地址(可变,仅对于根进程) 
　IN　recvcount 　待接收的元素个数(整型,仅对于根进程)
　IN　recvtype 　 接收元素的数据类型(句柄,仅对于根进程)
　IN　root　　　   接收进程的序列号(整型)
　IN　comm 　　 　 通信子(句柄)
int MPI_Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype, 
               void* recvbuf, int recvcount, MPI_Datatype recvtype, 
               int root, MPI_Comm comm)
```
每个进程(包括根进程)将其发送缓冲区中的内容发送到根进程,根进程根据发送这些数据的进程的序列号将它们依次存放到自已的消息缓冲区中.其结果就象一个组中的n个进程(包括根进程在内)都执行了一个调用:
```c
MPI_Send(sendbuf, sendcount, sendtype, root, ...)
```
同时根进程执行了n次调用:
```c
MPI_Recv(recvbuf+i*recvcount*extent(recvtype), recvcount, recvtype, i,...)
```
此处extent(recvtype)是调用函数MPI_Type_extent()所返回的类型,另外一种描述就象是组中的n个进程发送的n条消息按照它们的序列号连接起来,根进程通过调用MPI_RECV(recvbuf, recvcount*n, recvtype,...) 来将结果消息接收过来.
## Scatter
```c
MPI_SCATTER(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
            root,comm)
 IN   sendbuf     发送消息缓冲区的起始地址(可变,仅对于根进程)
 IN   sendcount   发送到各个进程的数据个数(整型,仅对于根进程)
 IN   sendtype    发送消息缓冲区中的数据类型(句柄,仅对于根进程)
 OUT  recvbuf     接收消息缓冲区的起始地址(可变)
 IN   recvcount   待接收的元素个数(整型)
 IN   recvtype    接收元素的数据类型(句柄)
 IN   root        发送进程的序列号(整型)
 IN   comm        通信子(句柄)
int MPI_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
```
MPI_SCATTER是MPI_GATHER的逆操作.
其结果相当于根进程执行了n次发送操作:
```c
MPI_Send(sendbuf+i*sendcount*extent(sendtype),sendcount,sendtype,i,...)
```
然后每个进程执行一次接收操作:
```c
MPI_Recv(recvbuf,recvcount,recvtype,i,...)
```
另外一种解释是根进程通过MPI_Send(sendbuf,sendcount*n,sendtype,...)发送一条消息,这条消息被分成n等份,第i份发送给组中的第i个处理器, 然后每个处理器如上所述接收相应的消息.
## Allgather
```c
MPI_ALLGATHER(sendbuf, sendcount, sendtype, recvbuf, recvcount,
              recvtype,comm)
 IN  sendbuf     发送消息缓冲区的起始地址(可变)
 IN  sendcount   发送消息缓冲区中的数据个数(整型)
 IN  sendtype    发送消息缓冲区中的数据类型(句柄)
 OUT recvbuf     接收消息缓冲区的起始地址(可变)
 IN  recvcount   从任一进程中接收的元素个数(整型)
 IN  recvtype    接收消息缓冲区的数据类型(句柄)
 IN  comm        通信子(句柄)
int MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
```
MPI_ALLGATHER和MPI_GATHER意义相同,但此时是所有进程都将接收结果,而不是只有根进程接收结果.从每个进程发送的第j块数据将被每个进程接收,然后存放在各个进程接收消息缓冲区recvbuf的第j块.每一个进程的sendcount和sendtype的类型必须和其他所有进程的recvcount和recvtype相同.
调用MPI_ALLGATHER相当于所有进程执行了n次调用:
```c
MPI_GATHER(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,
           root,comm)
```
其中root从0到n-1.有关于MPI_ALLGATHER的正确使用方法和MPI_GATHER相同.
## Alltoall
```c
MPI_ALLTOALL(sendbuf, sendcount, sendtype, recvbuf, recvcount,
             recvtype, comm)
 IN  sendbuf     发送消息缓冲区的起始地址(可变)
 IN  sendcount   发送到每个进程的数据个数(整型)
 IN  sendtype    发送消息缓冲区中的数据类型(句柄)
 OUT recvbuf     接收消息缓冲区的起始地址(可变)
 IN  recvcount   从每个进程中接收的元素个数(整型)
 IN  recvtype    接收消息缓冲区的数据类型(句柄)
 IN  comm        通信子(句柄)
int MPI_Alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm)
```
MPI_ALLTOALL是对MPI_ALLGATHER的扩展,区别是每个进程可以向每个接收者发送数目不同的数据.第i个进程发送的第j块数据将被第j 个进程接收并存放在其接收消息缓冲区recvbuf的第i块.每个进程的sendcount和sendtype的类型必须和所有其他进程的recvcount和recvtype相同,这就意谓着在每个进程和根进程之间,发送的数据量必须和接收的数据量相等.但发送方和接收方之间的不同数据类型映射仍然是允许的.
调用MPI_ALLTOALL相当于每个进程对每个进程(包括它自身)执行了一次调用
```c
MPI_Send(sendbuf+i*sendcount*extent(sendtype),sendcount,
         sendtype,i,...)
```
然后再执行一次从所有其他进程接收数据的调用:
```c
MPI_Recv(recvbuf+i*recvcount*extent(recvtype),recvcount,i,...)
```
所有参数对每个进程都是很重要的,而且所有进程中的comm值必须一致.
## Reduce
```c
MPI_REDUCE(sendbuf,recvbuf,count,datatype,op,root,comm)
 IN   sendbuf   发送消息缓冲区的起始地址(可变)
 OUT  recvbuf   接收消息缓冲区中的地址(可变,仅对于根进程)
 IN   count     发送消息缓冲区中的数据个数(整型)
 IN   datatype  发送消息缓冲区的元素类型(句柄)
 IN   op        归约操作符(句柄)
 IN   root      根进程序列号(整型)
 IN   comm      通信子(句柄)
int MPI_Reduce(void* sendbuf, void* recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm)
```
MPI_REDUCE将组内每个进程输入缓冲区中的数据按op操作组合起来,并将其结果返回到序列号为root的进程的输出缓冲区中.输入缓冲区由参数sendbuf、count和datatype定义;输出缓冲区由参数recvbuf、count和datatype定义;两者的元素数目和类型都相同.所有组成员都用同样的参数count、datatype、op、root和comm来调用此例程,因此所有进程都提供长度相同、元素类型相同的输入和输出缓冲区.每个进程可能提供一个元素或一系列元素,组合操作针对每个元素进行.例如,如果操作是 MPI_MAX,发送缓冲区中包含两个浮点数(count=2并且datatype=MPI_FLOAT),结果recvbuf(1)存放着所有sendbuf(1)中的最大值,recvbuf(2)存放着所有sendbuf(2)中的最大值.
定义了的Reduce操作有：
```c
        MPI_MAX           最大值
        MPI_MIN           最小值
        MPI_SUM           求和
        MPI_PROD          求积
        MPI_LAND          逻辑与
        MPI_BAND          按位与
        MPI_LOR           逻辑或
        MPI_BOR           按位或
        MPI_LXOR          逻辑异或
        MPI_BXOR          按位异或
        MPI_MAXLOC        最大值且相应位置
        MPI_MINLOC        最小值且相应位置
```
## Allreduce
```c
MPI_ALLREDUCE(sendbuf, recvbuf, count, datatype, op, comm)
 IN   sendbuf     发送消息缓冲区的起始地址(可变)
 OUT  recvbuf     接收消息缓冲区的起始地址(可变)
 IN   count       发送消息缓冲区中的数据个数(整型)
 IN   datatype    发送消息缓冲区中的数据类型(句柄)
 IN   op          操作(句柄)
 IN   comm        通信子(句柄)
int MPI_Allreduce(void* sendbuf, void* recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
```
MPI中还包括对每个归约操作的变形,即将结果返回到组内的所有进程.MPI要求组内所有参与的进程都归约同一个结果.除了将结果返回给组内的所有成员外,其他同MPI_REDUCE.
## ReduceScatter
```c
MPI_REDUCE_SCATTER(sendbuf, recvbuf, recvcounts, datatype, op, comm)
 IN   sendbuf       发送消息缓冲区的起始地址(可变)
 OUT  recvbuf       接收消息缓冲区的起始地址(可变)
 IN   recvcounts    整型数组,存放着分布在每个进程的结果元素个数, 所
                    有进程的数组都必须相同.
 IN   datatype      输入缓冲区的元素类型(句柄)
 IN   op            操作(句柄)
 IN   comm          通信子(句柄)

int MPI_Reduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
```
MPI_REDUCE_SCATTER对由sendbuf、count和datatype定义的发送缓冲区中的数组逐个元素进行归约操作,这个数组的长度count = ∑irecvcount[i].然后, 这个结果向量被分成n个互不连通的部分,这里n为组中成员数.第i段中包含recvcounts[i]个元素,第i段发送到进程i并且存放在由recvbuf、recvcounts[i]和datatype定义的输入缓冲区中.
## Scan
```c
MPI_SCAN(sendbuf, recvbuf, count, datatype, op, comm)
 IN   sendbuf    发送消息缓冲区的起始地址(可变)
 OUT  recvbuf    接收消息缓冲区的起始地址(可变)
 IN   count      输入缓冲区中元素的个数(整型)
 IN   datatype   输入缓冲区中元素的类型(句柄)
 IN   op         操作(句柄)
 IN   comm       通信子(句柄)
int MPI_Scan(void* sendbuf, void* recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
```
MPI_SCAN常用于对分布于组中的数据作前置归约操作.此操作将序列号为0,... ,i(包括i)的进程发送缓冲区的值的归约结果存入序列号为i 的进程的接收消息缓冲区中,这种操作支持的类型、语义以及对发送及接收缓冲区的限制和MPI_REDUCE相同.
# 点对点通信常用函数以及内置数据类型
## Send
```c
MPI_SEND(buf,count,datatype,dest,tag,comm)
    IN buf 发送缓存的起始地址(选择型)
    IN count 发送缓存的元素的个数(非负整数)
    IN datatype 每个发送缓存元素的数据类型(句柄)
    IN dest 目的地进程号(整型)
    IN tag 消息标志(整型)
    IN comm 通信子(句柄)
int MPI_Send(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```
## Recv
```c
MPI_RECV(buf,count,datatype,source,tag,comm,status)
    OUT buf 接收缓存的起始地址(选择型)
    IN count 接收缓存中元素的个数(整型)
    IN datatype 每个接收缓存元素的数据类型(句柄)
    IN source 发送操作的进程号(整型)
    IN tag 消息的标识(整型)
    IN comm 通信组(句柄)
    OUT status 状态对象(状态)
int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```
## Datatype
```c
     MPI datatype            C datatype 
     MPI_CHAR                signed char 
     MPI_SHORT               signed short int 
     MPI_INT                 signed int 
     MPI_LONG                signed long int 
     MPI_UNSIGNED_CHAR       unsigned char 
     MPI_UNSIGNED_SHORT      unsigned short int 
     MPI_UNSIGNED            unsigned int 
     MPI_UNSIGNED_LONG       unsigned long int 
     MPI_FLOAT               float 
     MPI_DOUBLE              double
     MPI_LONG_DOUBLE         long double 
     MPI_BYTE 
     MPI_PACKED 
```
