---
layout: default
---

# Ex4 二维数组中的查找

## 题目描述

> 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
>

## 问题分析

示例矩阵：

>[
>  [1,   4,  7, 11, 15],
>  [2,   5,  8, 12, 19],
>  [3,   6,  9, 16, 22],
>  [10, 13, 14, 17, 24],
>  [18, 21, 23, 26, 30]
>]

通过观察，我们发现矩阵**右上角**的数字是一行中最大的，一列中最小的数字。意识到这个问题是我们解决这个问题的重要一步。

## 解题思路

从矩阵的**右上角**这一点入手，当这个数字小于要找的数字时，我们向下移动一行；当这个数字大于要找的数字时，我们向左（前）移动一列。从这一点出发，我们就很容易写出代码。

## 代码

```c
bool
findNumberIn2DArray (int **matrix, int matrixSize, int *matrixColSize,
                     int target)
{
  if (matrix == NULL || matrixSize <= 0 || matrixColSize <= 0)
    return false;
  bool found = false;
  int i = 0, j = *matrixColSize - 1;
  while (i < matrixSize && j >= 0)
    {
      if (matrix[i][j] == target)
        {
          return true;
        }
      else if (matrix[i][j] > target)
        {
          --j;
        }
      else
        {
          ++i;
        }
    }
  return false;
}
```

## 运行结果

> 执行结果：通过
>
> 执行用时：56 ms, 在所有 C 提交中击败了10.35%的用户
>
> 内存消耗：8.1 MB, 在所有 C 提交中击败了56.64%的用户

好吧，略差。。。

## 备注

- LeetCode在编译时有个需要注意的，上述代码中作为安全检查的第5行，即

  ```C
  if (matrix == NULL || matrixSize <= 0 || matrixColSize <= 0)
  ```

  在写成如下形式时，

  ```C
  if (matrix == NULL || *matrix == NULL || matrixSize <= 0 || matrixColSize <= 0)
  ```

  当输入的矩阵为空时，LeetCode编译使用的AddressSanitizer会报错。

- 发现**右上角**这个关键位置是做这一个题的关键，否则很难了。当然参考LeetCode上面其他通过的提交，遍历、std::find_if之类的也可以过。好吧。