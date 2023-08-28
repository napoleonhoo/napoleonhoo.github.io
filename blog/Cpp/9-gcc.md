---
layout: default
---


# GCC编译的基本命令

## -o：输出文件命名

```shell
gcc -o main
```

这样编译main.c之后，会得到一个名为*main*的输出文件。	

## -E：预编译

```shell
gcc -E main.c -o main.i
```

## -S: 编译（生成汇编文件）

```shell
gcc -S main.c -o main.s
```

## -c：汇编（生成目标文件.o）

```shell
gcc -c main.c -o main.o
```

## 全过程，不加参数（+链接）

```shell
gcc main.c -o main
```

直接生成一个名为*main*的可执行文件。

## -std：语言标准

```shell
gcc -std=c99
```

使用C99标准编译程序。

