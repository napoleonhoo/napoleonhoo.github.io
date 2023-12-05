# CUDA by Example 笔记(五): 纹理内存

*对应书本第7章*

# 1 大纲

- 纹理内存，包括一维、二维纹理内存

# 2 内容

## 2.1 纹理内存简介

- 原是为了图像处理专门设计的
- 只读内存
- 也可以用在通用计算上

像常量内存一样，纹理内存也会在芯片上缓存，所以在一些情况下，通过减少对芯片外DRAM的内存请求来提供更高的带宽。特别地，纹理缓存被设计用来做图像处理相关的应用，而这种应用一般都会有访存的“空间局部性”。

## 2.2 纹理内存使用

### 声明

``` cpp
texture<float> texConstSrc;
```

这是在全局作用域声明的。

### 绑定

``` cpp
cudaMalloc((void**)&data.dev_constSrc, imageSize);
cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, imageSize);
```

### 使用

``` cpp
float c = tex1Dfetch(texConstSrc, offset);
```

`tex1Dfetch`虽然看起来是一个函数，但其实是一个编译器指令。它指示GPU将请求转到纹理内存，而非全局内存。

### 解绑定

``` cpp
cudaUnbindTexture(texConstSrc);
```

### 二维纹理内存的使用

``` cpp
// 声明
texture<float, 2> texConstSrc;

// 绑定方式
cudaMalloc((void**)&data.dev_constSrc, imageSize);
cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM);

// 使用
float c = tex2Dfetch(texConstSrc, x, y);

// 解绑定，方式与一维纹理内存相同
cudaUnbindTexture(texConstSrc);
```

*<font color=red>问题4：`cudaBind`和`cudaUnbind`编译时会提示warning：deprecated，那么应该用啥是最新的函数呢？</font>*

## 2.3 `MyAnitBitmap`

这里实现了书本上`CPUAnimBitmap`的全部接口，对于没有图形界面或OpenGL的同学，可以用这个作为代替，当然，并不能输出图像...

``` cpp
#pragma once

#include <stdio.h>
#include <stdlib.h>

struct DataBlock;

class MyAnimBitmap{
public:
    MyAnimBitmap(int width, int length, DataBlock* data) : _width(width), _length(length), _data(data) {
        _real_data = (int*)malloc(width * length * sizeof(int));
    }

    int image_size() {
        return _width * _length * sizeof(int);
    }

    unsigned char* get_ptr() {
        return reinterpret_cast<unsigned char*>(_real_data);
    }

    ~MyAnimBitmap() {
        free(_real_data);
    }

    void anim_and_exit(void (*a)(void*, int), void (*e)(void*)) {
        _fAnim = a;
        _animExit = e;
        anim();
        cleanup();
    }

    void anim() {
        for (int i = 0; i < 8; ++i) {
            _fAnim(_data, i);
        }
    }

    void cleanup() {
        _animExit(_data);
    }
    
private:
    int _width, _length;
    DataBlock* _data;
    int* _real_data;
    void (*_fAnim)(void*, int);
    void (*_animExit)(void*);
};
```



# 3 总结

本节主要介绍了GPU中的纹理内存的使用方法，内容不是特别多，但是另一方面来讲，纹理内存并非是通用计算的主流。

# 4 英语学习

- inconsequential：无关紧要的，离题的

