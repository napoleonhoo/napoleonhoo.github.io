---
layout: default
---

# Ex5 替换空格

## 题目描述

> 请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

## 问题分析 & 解题思路

这个问题的关键点还是在申请新的字符串空间时，给"%20"这个字符串预留下位置。**从后向前**遍历，遇到非空格字符就复制，遇到空格字符就用"%20"代替。仅此而已。

## 代码

```c
char *
replaceSpace (char *s)
{
  if (!s)
    return NULL;
  int number_of_space = 0;
  for (int i = 0; i < strlen (s); ++i)
    {
      if (s[i] == ' ')
        {
          ++number_of_space;
        }
    }
  int new_length = strlen (s) + 1 + number_of_space * 2;
  char *str = (char *)malloc (sizeof (char) * new_length);
  memset (str, '\0', new_length);
  int j = new_length - 1;
  for (int i = strlen (s) - 1; i >= 0; --i)
    {
      if (s[i] != ' ')
        {
          str[--j] = s[i];
        }
      else
        {
          str[--j] = '0';
          str[--j] = '2';
          str[--j] = '%';
        }
    }
  return str;
}
```

## 运行结果

>执行结果：通过
>
>执行用时：0 ms, 在所有 C 提交中击败了100.00%的用户
>
>内存消耗：5.3 MB, 在所有 C 提交中击败了28.67%的用户

好吧，LeetCode上的用时还是挺令人着迷的。