# 清空slice的两种方法

当我刚开始学习Go语言时，对slice中没有类似于C++的vector中clear()/erase()的函数而感到不知所措。后来的我却发现，Go中对清空、删除元素有着更方便的操作。

## 1. 索引操作

清空slice的第一种方法就是利用索引操作，如：

```go
a = a[:0]
```

就可以将slice a clear。这句话的实际含义是：将a中第0个元素之前的数据赋给a。这个操作：

- 直接在原来的slice上清空，不会引起Garbage Collection。
- clear之前之后的slice是一个slice。
- clear之后，length=0，capacity=原来的capacity。

实际例子：

```go
package main

import (
	"fmt"
)

func main() {
	a := []int{1, 2, 3, 4, 5, 6}
	fmt.Printf("addr of a before clear: %p\n", a)
	a = a[:0]
	fmt.Printf("before append, content of a: %v, len of a: %d, cap of a: %d\n", a, len(a), cap(a))
	fmt.Printf("addr of a after clear: %p\n", a)
	a = append(a, 7)
	fmt.Printf("after append, content of a: %v, len of a: %d, cap of a: %d\n", a, len(a), cap(a))
}
```

输出：

```
addr of a before clear: 0xc00001c180
before append, content of a: [], len of a: 0, cap of a: 6
addr of a after clear: 0xc00001c180
after append, content of a: [7], len of a: 1, cap of a: 6
```

由其中的地址打印语句即可以看出：slice a还是原来的slice a。

## 2. nil赋值

清空一个slice的第二种方法就是将slice赋一个nil值，如：

```go
b = nil
```

这句话的实际意义就是给b赋一个空指针。这个操作：

- 会引发Garbage Collection。
- clear之前和之后的slice虽然有着相同的变量名，但是已经在实际上不再是一个slice了。
- clear之后，length=0，capacity=0.

```go
package main

import (
	"fmt"
)

func main() {
	b := []int{1, 2, 3, 4, 5, 6}
	fmt.Printf("addr of b before clear: %p\n", b)
	b = nil
	fmt.Printf("before append, content of b: %v, len of b: %d, cap of b: %d\n", b, len(b), cap(b))
	fmt.Printf("addr of b after clear: %p\n", b)
	b = append(b, 7)
	fmt.Printf("after append, content of b: %v, len of b: %d, cap of a: %b\n", b, len(b), cap(b))
	fmt.Printf("addr of b after append: %p\n", b)
}
```

输出：

```
addr of b before clear: 0xc000098000
before append, content of b: [], len of b: 0, cap of b: 0
addr of b after clear: 0x0
after append, content of b: [7], len of b: 1, cap of a: 1
addr of b after append: 0xc000094020
```

从上面对地址的输出可以看到，clear之前和之后的slice已经不在指向同一个地址了，也实际上不再是同一个slice了。并且当我们append元素时，发生了类似于创建一个新slice的操作过程。