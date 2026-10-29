# Lecture 5 C to Assembly

# 课程内容
本节课主要描述了C语言到汇编语言到过程，主要是用了LLVM IR相关的一些例子。

## Clang/LLVM编译流程

| | |
|-|-|
| C source | bitarray.c |
| | Clang preprocessor |
| Preprocessed Source | bitarray.i |
| | Clang code generator |
| LLVM IR | bitarray.ll |
| | LLVM optimizer |
| Optimized LLVM IR | bitarray.ll |
| | LLVM code generator |
| Assembly | bitarray.s |

LLVM IR: LLVM Intermediate Representation （中间结果表示）

### 查看LLVM IR 的结果
``` shelll
clang -O3 fib.c -S -emit-llvm
```
-S: 产生汇编代码
-S -emit-llvm: 产生LLVM IR

### 编译LLVM IR 的结果
``` shell
clang fib.ll -S
```

## 结语

虽说汇编语言作为高级语言和机器代码的桥梁，是十分重要的一环。但是由于其并不是特别复杂，重点主要在读懂代码上，难点并非是汇编语言本身，诀窍在于勤加练习。这里就不在赘述了。接下来，敬请期待进入到多线程编程的环节。