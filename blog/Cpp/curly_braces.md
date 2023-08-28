---
layout: default
---

# 用花括号可以更简洁的地方

Reference:
- [5 Ways Using Braces Can Make Your C++ Code More Expressive](https://www.fluentcpp.com/2019/11/15/5-ways-cpp-braces-will-make-your-code-more-expressive/)

## 1 填充各类容器

```cpp
// vector
std::vector<std::string> words = {"the", "mortar", "hoding", "code", "together"};

// pair
std::pair answer = {"forty-two", 42};

// tuple
std::tuple cue = {3, 2, 1, "go!"};

// map
std::map<int, std::string> numbers = {{1, "one"}, {2, "two"}, {3, "three"}};
```

## 2 向函数传递复合参数

```cpp
// function 1
void display(std::vector<int> const& values);

// function 2
template<typename T>
void display(std::vector<T> const& values);

// call
display({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
```

## 3 从函数返回复合值或对象

```cpp
std::vector<int> numbers() {
	return {0, 1, 2, 3, 4, 5};
}
```

## 4 聚合初始化

```cpp
struct Point { int x, y, z };

Point p = {1, 2, 3};
```

## 5 RAII
使用大括号来圈起来一段作用域。
```cpp
{
	std::unique_ptr<X> myResource = nullptr;
}
```
