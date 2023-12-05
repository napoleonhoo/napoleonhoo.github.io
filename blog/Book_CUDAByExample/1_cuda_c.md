# CUDA by Example 笔记(一): CUDA C介绍

*对应书本第三章*

# 1 大纲

- 基本概念
- kernel调用
- 参数传递
- device属性

# 2 内容

## 2.1 基本概念

**host**：CPU和系统的内存

**device**：GPU和“显存”

**kernel**：device上执行的函数

## 2.2 kernel调用

``` cpp
__global__ void kernel(void) {
}

int main(void) {
    kernel<<<1, 1>>>();
    printf("Hello world!\n");
    return 0;
}
```

注意到两点：

- kernel函数最前面用`__global__`修饰。这是CUDA给C标准加入的一个修饰符，这个修饰符告诉编译器，这个函数应该在device上运行，而不是host。
- kernel调用在普通的函数调用参数列表之前，有三个尖括号包围的调用kernel的参数，即`<<<1, 1>>>`

## 2.3 参数传递

``` cpp
#include "common/book.h"

__global__ void add(int a, int b, int* c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
    add<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);
    return 0;
}
```

注意到：

- 我们可以和调用CPU函数一样，调用GPU函数，即函数的传参列表
- 需要在device上分配内存

<font color=red>*问题1：为啥a和b不需要使用`cudaMalloc`分配内存呢？*</font>

### device上分配内存：`cudaMalloc`

这是在device上分配内存空间的函数。第一个参数是一个指针：指向分配的内存的指针。所以是个二级指针。第二个参数是内存的大小。

不可以在host上对device上分配的内存进行读写，但是可以传递、做数学运算（如：`++ptr`）、cast为不同类型（如`reinterpret_cast`）。简单来讲，**可以对指针本身进行运算，但是不能对指针指向的内容做运算**。

### host、device之间的内存拷贝：`cudaMemcpy`

第一个参数是目的指针，第二个参数是源指针，第三个参数是拷贝内存的大小（以byte计），第四个参数，代表拷贝种类，目前只需要掌握两种：`cudaMemcpyHostToDevice`：从host拷贝向device，`cudaMemcpyDeviceToHost`：从device拷贝向host。

这个函数用于host与device之间的内存拷贝，即不可以直接memcpy之类的C/C++上的CPU的操作。

*以后会学到更多的拷贝种类和拷贝函数。但这个函数是最基础的。*

### 释放device上的内存：`cudaFree`

使用很简单，和CPU上的`free`基本上一个意思，完全一样的用法。

## 2.4 device属性

### 获取device的个数： `cudaGetDeviceCount`

``` cpp
int count;
cudaGetDeviceCount(&count);
```

这个函数可以获取你机器上的device的个数。

### 获取device的属性：`cudaGetDeviceProperties`

``` cpp
int i;
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, i);
```

这个函数的第一个参数就是device的property结构体，是函数的返回参数；第二个参数是输入参数，代表返回的是哪个device的property。这个`i`是要小于上面得到的`count`的，从0开始计数。

`cudaDeviceProp`结构体的内容有很多，这里就不一一列举了，大家可以参考[官方文档](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)。大家可以选择一些打印出来。

## 2.5 其他

### 错误处理：`HandleError`

``` cpp
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
```

这个函数对函数的返回值做了校验，如果返回值不等于`cudaSuccess`，则用`cudaGetErrorString`获取错误说明并打印出来。是一个简单又使用的函数。建议大家写代码的时候带上。

# 3 总结

这一节的大多数都是基础知识，可能大家都有些了解。着重介绍了`cudaMalloc`函数。

# 4 英语学习

- skim：浏览
- ruse：诡计
- sinking：（情绪突然）颓废的、沮丧的、抑郁的
- vanilla：普通的、毫无特色的
- embellish：装饰，润色，渲染
- vandalize：蓄意破坏
- shorthand：速记法，简略表达式
- notwithstanding：尽管，然而
- caveat：警告
- discrete：分离的，不相关的
- ornamentation：装饰，点缀
