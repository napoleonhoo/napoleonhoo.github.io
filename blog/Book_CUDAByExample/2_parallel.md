# CUDA by Example 笔记(二): 并行编程基础

*对应书本第四章*

# 1 大纲

- cuda并行编程的一般流程
- block、grid、`blockIdx.x`
- 二维block
- kernel调用的第一个流程

#  2 内容

## 2.1 向量相加

### host上的主要代码

``` cpp
int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR( cudaMalloc((void**)&dev_a, sizeof(int) * N) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, sizeof(int) * N) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_c, sizeof(int) * N) );

    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) );

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}

```

### 总结一般的host代码最基础的写作顺序：

1. 在host上分配内存，包括host上对应device的输入、输出；
2. 在device上分配内存，包括device上的输入、输出；（函数：`cudaMalloc`）
3. 为输入赋值；
4. 将输入拷贝到device上的变量或地址；（函数：`cudaMemcpy`）
5. 执行device上的kernel；（调用方式：`<<<    >>>`）
6. 将device的输出数据拷贝到host上的地址；（函数：`cudaMemcpy`）
7. 释放在device上分配的内存；（函数：`cudaFree`）
8. 释放在host上分配的内存。

以上就是写一个比较常见的、通用的、基础的host控制代码的主要流程、顺序。上面在每一个涉及到和device交互的地方都在括号里面加了一些常见的函数、调用方式。都是比较基础的，目前学到这里就差不多了。

### 调用kernel的方式

``` cpp
add<<<N, 1>>>(a, b, c);
```

三个尖括号中的第一个参数，是我们需要多少个并行的**block**来执行我们的kernel。本例中，有N（即向量的维度的个数）个block来并行执行这个kernel函数。*备注：虽然更大的block个数会有更大的并行度，但是由于目前硬件所限，N不能大于65535*。多个并行的block组成了一个**grid**。

本例中，我们的block和grid都是一个一维的参数，因为我们使用的是一个数字（标量），自动被解读成2维。

*对的，还有更高的维度，即二维、三维。我们会在后来的例子和实际应用场景中接触到。*

### kernel代码

``` cpp
__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
```

很难不注意到一个变量**`blockIdx.x`**。它是cuda的一个内置变量，恰如前面所说，我们使用了N个block来并行执行我们的kernel，`blockIdx`就是指的block的索引（像所有编程世界的索引一样，从0开始编码）。对于每一个不同的执行这个kernel的block，都有一个固定的值，且每个block不一样。这里为什么有一个`.x`呢？因为恰如上面所说，我们可以定义更高维度的block，但目前我们只使用的是一维的block，所以只需要`x`索引，也可以认为这个是block的x*坐标*。

那么为什么这里要写`tid < N`呢，毕竟我们只用了N个block，任何的`blockIdx.x`都不会对于等于N？首先，检查索引是否越界是一个好的编程习惯，也可以说是一种*防御性编程*。书中原话：*check the results of every operation that can fail.*。另外，还有其他的原因，我们暂时先不解释。

## 2.2 julia set (朱利亚集合)

这里只会说几个比较重点的语法知识，其他的内容感兴趣的同学可以看课本或者代码库。

### 使用二维block调用kernel

``` cpp
#define DIM 1000
dim3 grid(DIM, DIM);
kernel<<<grid, 1>>>(dev_bitmap);
```

这里重点关注一下调用kernel的第一个参数，grid。这里的grid变量是一个`dim3`类型的变量。这个类型是cuda内置的一个类型，它是一个三维的tuple。（*其实，目前cuda只支持二维。这么做是为了以后的兼容吧。第三维默认为1。*）本例中，我们使用了`DIM * DIM`个block来执行我们的kernel。

### 获取二维block中的tid

对于host代码，将一个二维数组当成一维数组后，二维索引转换为一维索引的方法是：

``` cpp
for (int y = 0; y < DIM; ++y) {
    for (int x = 0; x < DIM; ++x) {
        int offset = x + y * DIM;
    }
}
```

这个是不难理解的。其中的`DIM`是一行有多少个元素。

对应到device上有：

``` cpp
int x = blockIdx.x;
int y = blockIdx.y;
int offset = x + y * gridDim.x;
```

这里的`blockIdx.y`即为block索引的第二维，即y坐标。

这里的**`gridDim.x`**是类似与`blockIdx.y`似的变量，这个值对于所有的block都是一个常量、不变的值，它表示了我们总共使用的grid的维度。本例中，它的值是`DIM`。

### 两种函数

``` cpp
struct gpu_cuComplex {
    float r, i;
    __device__ gpu_cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magtitude2(void) { return r * r + i * i; }
    __device__ gpu_cuComplex operator*(const gpu_cuComplex& a) {
        return gpu_cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ gpu_cuComplex operator+(const gpu_cuComplex& a) {
        return gpu_cuComplex(r+a.r, i+a.i);
    }
};
```

注意看下书中的这个结构体，我们可以看到另一种函数前缀修饰符`__device__`。综合来看：

- `__global__`：可以从host上调用（CPU的函数）的在device上执行的函数。
- `__device__`：只能从device上（`__host__`或`__device__`函数）调用的函数。

<font color=red>*问题2：难道`__global__`只能返回void，`__device__`才能返回int之类的值？*</font>

*注意：这里修正了书本上的一个错误，即构造函数也需要device修饰。*

### 其他

因为我的环境没有图形界面，更没有OpenGL，但是为了看一下体验一下这个过程，我自己模仿书中`CPUBitmap`的功能写了一个`MyCPUBitmap`。当然，只能输出数字，不能输出图形。需要的同学可以自行参考。

``` cpp
class MyCPUBitmap {
public:
    MyCPUBitmap(int width, int length) : _width(width), _length(length) {
        _bitmap = (int*)malloc(_width * _length * sizeof(int));
    }
    ~MyCPUBitmap() {
        free(_bitmap);
    }
    unsigned char* get_ptr() {
        return reinterpret_cast<unsigned char*>(_bitmap);
    }

    size_t image_size() {
        return _width * _length * sizeof(int);
    }

    void display_and_exit() {
        display();
    }
private:
    void display() {
        for (int i = 0; i < _width * _length; ++i) {
            int tmp = _bitmap[i];
            // unsigned char *ptr_tmp = reinterpret_cast<unsigned char*>(tmp); // Wrong !
            unsigned char *ptr_tmp = reinterpret_cast<unsigned char*>(&tmp);
            for (int j = 0; j < 4; ++j) {
                printf("%d ", ptr_tmp[j]);
            }
            printf("\t");
        }
    }

private:
    int _width;
    int _length;
    int *_bitmap;
};
```

# 3 总结

今天我们主要学习了一些基础的CPU并行变成方法，至少我们目前可以根据自己的需求，做一个比较简单的GPU程序了。

# 4 英语学习

- vien：风格，纹理
- minion：下属
- contrive：设法促成，出乎意料得做到
- convoluted：错综复杂的，弯曲的
- conscientious：勤勉认真的，煞费苦心的
- liberally：洒落
- flashy：俗艳的
- the uninitiated：无专门知识的门外汉
- fractal：分形
- lest：以免，万一
- wrinkle：皱眉，小问题

