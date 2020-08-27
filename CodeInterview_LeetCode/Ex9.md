---
layout: default
---

# Ex9 用两个栈实现队列

## 问题描述

>用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
>

## 解题思路

假设输入是10、100，如果输入到队列（**FILO**，First In Last Out，先进后出）的话，输出应该是10、100；但输入到栈（**FIFO**，First In First Out，先进先出）的时候，输出应该是100、10。

我们可以这么来做，在输入（appendTail）的时候，将输入插入到stack1中，插入顺序10、100，如下图：

当输出（deleteHead）时（暂不考虑队列中无元素）：

- 将stack1中的top元素push到stack2中，然后pop，直到stack1为空，如下图：

- 将stack2中的top元素存储起来（为rtn），以便返回使用；
- 将stack2中的top元素push到stack1中，然后pop，直到stack2为空，和第一步恰巧相反，如下图：
- 返回rtn。

## 代码

```c++
class CQueue
{
public:
  CQueue () {}

  void
  appendTail (int value)
  {
    stack1_.push (value);
  }

  int
  deleteHead ()
  {
    if (stack1_.empty ())
      return -1;
    while (!stack1_.empty ())
      {
        stack2_.push (stack1_.top ());
        stack1_.pop ();
      }
    int rtn = stack2_.top ();
    stack2_.pop ();
    while (!stack2_.empty ())
      {
        stack1_.push (stack2_.top ());
        stack2_.pop ();
      }
    return rtn;
  }

private:
  std::stack<int> stack1_, stack2_;
};
```



## 结果

>执行结果：通过
>
>执行用时：1224 ms, 在所有 C++ 提交中击败了8.62%的用户
>
>内存消耗：111 MB, 在所有 C++ 提交中击败了20.57%的用户

## 备注

为了更符合题目的要求，不同以往，我们使用了C++中的`std::stack`，而没有使用C中的数组。