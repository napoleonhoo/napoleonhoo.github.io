# § new & delete

## 1 new & delete

new主要分为三种：new operator、operator new、placement new。同样地，delete分为两种：delete operator、operator delete、placement delete。



### 1.1 new/delete operator

也可称为new/delete expression。即最常用的哪一种，如

``` cpp
A* a = new A();
delete a;
```

new operator的执行过程主要如下：

1. 调用`operator new`分配内存；
2. 调用构造函数初始化相应的内存；
3. 返回相应的指针。

另外，它是不能被重载。

delete operator的执行过程如下：

1. 调用对象的析构函数；
2. 调用`operator delete`分配内存。



### 1.2 operator new/delete

operator new、operator delete是可以被重载的，以实现不同的或定制化的分配策略。

声明如下：

``` cpp
void* operator new(std::size_t) throw (std::bad_alloc);
void* operator new[](std::size_t) throw (std::bad_alloc);
void operator delete(void*) throw();
void operator delete[](void*) throw();
void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new[](std::size_t, const std::nothrow_t&) throw();
void operator delete(void*, const std::nothrow_t&) throw();
void operator delete[](void*, const std::nothrow_t&) throw();
```

上面是很多的重载版本。`operator new`是主要用来执行内存分配任务的，会调用`malloc`。可以在类里面对其进行重载，默认调用全局`::operator new`；`operator delete`是主要用来执行释放内存操作的，会调用`free`。可以在类里面对其进行重载，默认调用全局`::operator delete`。



### 1.3 placement new/delete

它是一种operator new的重载，它不分配内存，只是在已经分配好的内存上调用构造函数。

声明定义如下：

``` cpp
// Default placement versions of operator new.
inline void* operator new(std::size_t, void* __p) throw() { return __p; }
inline void* operator new[](std::size_t, void* __p) throw() { return __p; }

// Default placement versions of operator delete.
inline void  operator delete  (void*, void*) throw() { }
inline void  operator delete[](void*, void*) throw() { }
```

使用方法如下（这其实称为placement new expression，调用的是placement new function）：

``` cpp
A *a = new (buf) A();
```

不存在一种placement delete expression，即不能显式调用placement delete。当placement new expression调用placement new function，如果构造函数函数构造的时候发生了异常，就会调用placement delete function。

