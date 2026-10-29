# Lecture 4 Assembly Language and Computer Architecture
# 课程内容
这节课主要是将了汇编语言、计算机架构的基础部分。

## The Four Stages of Compilation
| | |
|-|-|
| Source | bitarray.c main.c |
| Preprocess | clang -E |
| Preprocessed source | bitarray.i main.i |
| Compile | clang -S |
| Assembly | bitarray.s main.s |
| Assemble | clang -c |
| Object File | bitarra.o main.o  Libraries |
| Link | ld |
| Binary Executable | everybit |

## Assembly Code
汇编语言提供了一个方便机器码的符号化标识。

### Disassembling
编译时使用-g标志。使用命令：
``` shell
objdump -S exe
```

### Why Assembly?
- 汇编语言揭示了编译器做了什么和没做什么。
- bug可能会在低层级上出现。比如说，<font color=red>**bug可能只会在用-O3编译的时候出现**</font>。另外，编译器可能就是bug的原因。
- 做优化的时候，可以手动更改编译。
- 逆向工程（Reverse Engineering）。

### Expectations of Students
- 了解如何使用x86指令实现C语言的结构。
- 借助手册，可以阅读x86汇编语言。
- 借助一般的编译器行为了解高层次的性能优化。
- 可以修改编译器产生的x86汇编语言。
- 使用**编译器内部函数(compiler intrinsic functions)**来使用C并不能直接支持的汇编指令。
- 手写汇编语言。

## x86-64 ISA Primer
ISA: instruction set architecture

### Registers
有两个特别需要注意的：
- 16个128位的%xmm[0-15]的XMM registers (for SSE).
- 16个256位%ymm[0-15]的YMM registers (for AVX).

### Data Types
### Opcode
### RFLAGS Register
### Direct Addressing Modes
- 需要一个时钟周期来从寄存器取数，但是需要几百个时钟周期来从内存取数。
- Instruction-pointer relative: The address is indexed relative to %rip.
``` asm
movq 172(%rip), %rdi
```
- Base Indexed Scale Displacement: `Address = Base + Index * Scale + Displacement`. Often use in stack frame.
``` asm
movq 172(%rdi, %rdx, 8), %rax
```
### Assembly Idiom
1. 对寄存器清零。
``` asm
xor %rax, %rax
```

2. 看寄存器是否是0.
``` asm
test %rcx, %rcx
```

3. No-operation (no-op) instructions
``` asm
data16 data16 data16 now %cs:0x0(%rax,%rax,1)
```
指令本身并不做任何事情。只是优化一些指令的内存（如代码大小、对齐等等）。

## Floating-Point And Vector Hardware
现代x86-64架构通过一些不同的指令集支持标量（非向量）浮点数数学计算。
- SSE和AVX指令支持单精度和双精度标量的浮点数数学计算。（即float、double）
- x87指令支持单、双、扩展精度的浮点数数学计算。（即float、double、long double）
- SSE和AVX也支持向量指令。

### SSE
编译器更喜欢使用SSE指令而不是x87指令，因为SSE指令编译、优化更简单。 

SSE使用XMM寄存器和浮点数类型。

| | |
|-|-|
| ss | One single-precision floating-point value |
| sd | One double-precision floating-point value |
| ps | Vector of single-precision floating-point values |
| pd | Vector of double-precision floating-point values |

- The first letter: s (single) or p (packed).
- The second letter: s (single-precision) or d (double-precision).

### Vector Hardware
现代微处理器一般结合向量硬件来处理数据，使用的是single-instruction stream, multiple-data stream (SIMD) 方式。

每一个向量寄存器存放k个标量整数或浮点数。向量单元存放k个向量通道（Vector Lane），每一个包含标量整数或浮点数硬件。所有向量通道在一个时钟周期内使用同样的指令和控制信号。其中，k是向量宽度。

向量指令一般是elementwise运行的：
- 一个向量的第i个元素只可以参与与另一个向量的第i个元素。
- 所有通道对相应的向量的元素执行同样的操作。
- 根据架构的不同，向量内存可能需要对齐。
- 一些架构支持通道之间的操作。

现代x86-64架构支持多个向量指令集。
- 现代SSE、AVX指令集支持在整数、单精度、双精度的浮点数向量操作。
- 现代AVX指令集支持在单精度、双精度的浮点数向量操作。
- AVX2指令集增加了对整数向量的支持。
- AVX-512 (AVX3) 指令将寄存器长度增加到512位，并且提供了新的向量操作。

AVX和AVX2指令集增加了SSE的功能：
- SSE指令使用128位XMM向量寄存器，且可以同时操作至多两个操作数。
- AVX也可以使用256位的YMM向量寄存器，并且可以操作3个操作数：2个源操作数，1个目的操作数。
``` asm
Vaddpd %ymm0, %ymm1, %ymm2
```

Example Opcodes:

| | SSE | AVX/AVX2 |
|-|-|-|
| Floating-point | addpd | vaddpd |
| Integer | paddq | vpaddq |

- p 区别是整数还是浮点数。
- v 区别是SSE还是AVX/AVX2.

## Overview of Computer Architecture
### 5-Stage Processor
``` mermaid
graph LR;
A[IF] --> B[ID] --> C[EX] --> D[MA] --> E[WB]
```

``` mermaid
graph LR;
A[Fetch Unit] --> B [Decode -> GPRS] --> C[ALU] --> D[Data Mem.] --> E[WB]
```

### 现代处理器的一些设计特点
- 向量硬件
- 超标量处理
- 乱序执行
- 分支预测

# 其他
- 教授的话：If you really want to understand something, you need to understand it to the level that's necessary and the one level below that. It's not that you'll necessarily use that one level below it, but that gives you insight as to why that layer is what it is and what's really going on.
- GPR: General Purpose Register
