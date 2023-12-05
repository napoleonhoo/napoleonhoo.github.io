---
layout: default
---

# SpinLock

## 简介

reference: [wikipedia - Spinlock](./https://en.wikipedia.org/wiki/Spinlock)

1. 一个线程在获取锁的过程中是循环等待的（spin），一直不断地来检查所是否可用。
2. busy waiting 忙等待
3. 因为它避免了操作系统进程重新调度或者上下文切换的开销，所以，当线程很可能**只等待很短的一段时间**的时候，使用spinlock是很高效的。由于这个原因，操作系统kernel经常使用spinlock。
4. 然而，当需要等待很长一段时间的时候，spinlock就很浪费了。因为其它的线程需要一直spin。
5. 实现的时候，要考虑race condition。
6. 实现一般需要使用汇编指令，如atomic的test-and-set指令等。