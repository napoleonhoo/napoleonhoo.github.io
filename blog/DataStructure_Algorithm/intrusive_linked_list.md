---
layout: default
---

# 侵入式链表（Intrusive Linked List）

参考链接[Intrusive linked lists](https://www.data-structures-in-practice.com/intrusive-linked-lists/)

## 一般链表

``` cpp
struct list {
    int* data;
    struct list *next;
};
```

## 侵入式链表

``` cpp
struct list {
    struct list* next;
};

struct item {
    int val;
    struct list items;
};
```

实际上感觉大多数人用的就是侵入式链表啊。不知道为啥有个专门的名字。