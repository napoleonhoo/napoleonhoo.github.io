---
layout: default
---

# Ex7 重建二叉树

## 问题描述

> 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
>
> 例如，给出
>
> > 前序遍历 preorder = [3,9,20,15,7]
> > 中序遍历 inorder = [9,3,15,20,7]
>
> 返回如下的二叉树：
>
> > ​    3
> >
> >    / \
> >   9  20
> >     /  \
> >    15   7

## 解题思路

对于前序遍历来讲，第一个node是根结点，然后是左子树、右子树，根结点在最**前**面；对于中序遍历，先是左子树，然后是根结点，最后是右子树，根结点在**中**间。要想重建一颗二叉树，只需要知道根结点、左子树、右子树分别是哪些结点，但单单只根据前序遍历或中序遍历得不到这些结点，所以需要两种遍历方式。

以题中为例，preorder的第一位（3）就是根结点，用这一个来将中序遍历的左子树（9）和右子树（15，20，7）分开，然后得到左子树和右子树的长度，就可以得到前序遍历的左子树（9）和右子树（20，15，7），递归地，就可以得到一个二叉树。

## 代码

```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

struct TreeNode *
buildTree (int *preorder, int preorderSize, int *inorder, int inorderSize)
{
  if (!preorder || !inorder || preorderSize <= 0 || inorderSize <= 0)
    {
      return NULL;
    }
  struct TreeNode *head
      = (struct TreeNode *)malloc (sizeof (struct TreeNode) * 1);
  head->val = preorder[0];
  int index = 0;
  for (int i = 0; i < inorderSize; ++i)
    {
      if (inorder[i] == head->val)
        {
          index = i;
          break;
        }
    }
  int length_left = index, length_right = inorderSize - index - 1;
  head->left = buildTree (preorder + 1, length_left, inorder, length_left);
  head->right = buildTree (preorder + length_left + 1, length_right,
                           inorder + length_left + 1, length_right);
  return head;
}
```

## 结果

> 执行结果：通过
>
> 执行用时：20 ms, 在所有 C 提交中击败了52.81%的用户
>
> 内存消耗：11.4 MB, 在所有 C 提交中击败了66.26%的用户

## 备注

- 后序遍历先是左子树，然后是右子树，最后是根结点，根结点在最**后**。
- 可以得到一颗完整二叉树的遍历方式至少需要有两种，并且包含中序遍历，如前序和中序、后序和中序，使用前序和后序遍历得不到一颗完整的二叉树。
- 对于树来讲，递归是一个必须要掌握的技能，递归重在理解，在理解的基础上写代码就十分简单了。

