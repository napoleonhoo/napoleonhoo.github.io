---
layout: default
---

# Ex6 从尾到头打印链表

## 题目描述

>输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

## 问题分析 & 解题思路

刚做过[Ex5二维数组中的查找](./Ex4.html)的问题，很快就可以联想到是不是可以用从后向前插入的方法写入链表上的元素，需要一次遍历。考虑到这样需要预先知道链表的长度，需要一次遍历。

## 代码

```c
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     struct ListNode *next;
 * };
 */
/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int *
reversePrint (struct ListNode *head, int *returnSize)
{
  *returnSize = 0;
  struct ListNode *list = head;
  while (list)
    {
      ++*returnSize;
      list = list->next;
    }
  int *array = (int *)malloc (sizeof (int) * (*returnSize));
  list = head;
  int i = *returnSize - 1;
  while (list)
    {
      array[i--] = list->val;
      list = list->next;
    }
  return array;
}
```

## 运行结果

>执行结果：通过
>
>执行用时：8 ms, 在所有 C 提交中击败了89.46%的用户
>
>内存消耗：6.8 MB, 在所有 C 提交中击败了53.73%的用户

## 备注

- 需要注意一点，题目中的描述没有说清楚的是：returnSize这个参数需要作为输出数据，内容是数组的长度，（当然也是链表的长度）
- 当然也可以用递归写出更简洁的代码，这里就不再赘述。

