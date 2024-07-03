# 阅读笔记：Netflix推荐系统设计（2013）

## 一、Netflix Recommendations: Beyond the 5 stars (Part 1)

Reference: [Netflix Recommendations: Beyond the 5 stars (Part 1)](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)

这片文章比较简单，主要讲诉的是之前举办的Netflix Prize的获奖算法。另外还有一丢丢的Netflix的发展。

### 1 Netflix Prize的获奖算法

举办比赛的目的是找到一些推荐给用户的新算法，化解成一个算法问题就是如何降低RMSE（Root Mean Square Error），将目前的0.9525的RMSE降低到0.8572或更低。

第一个获得Progress Prize的有8.43%的提升。主要算法的内容是Matrix Factorization（一般称作SVD，Single Value Decomposistion）和受限玻尔兹曼机（Restricted Boltzmann Machines，RBM）。只使用SVD是0.8914的RMSE，只是用RBM是0.8990的RMSE。二者的线性结合结果降低至0.88。两者的结合仍是今天推荐引擎的主要算法之一。

Netflix Prize的最终大奖于两年后颁发，赢得了$1M奖金。但那是公司的总体战略发生了重大的转变。

### 2 Netflix的发展

简单来说，Netflix从原来租借DVD的公司，变成了一个流媒体公司。它原来的业务是通过用户在网上订阅，然后通过邮件邮寄DVD到用户家里。现在它是一个流媒体公司，线上点播。

感兴趣的花可以阅读：[Netflix - Wikipedia](https://en.wikipedia.org/wiki/Netflix)


## 二、Netflix Recommendations: Beyond the 5 stars (Part 2)

Reference：[Netflix Recommendations: Beyond the 5 stars (Part 2)](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5)

这一篇着重讲诉了Netflix的模型的选择、使用、评估到最终上线的方法等。这里重点介绍文章中的“Data and Models”部分。

### 用到的数据

- 用户的评分。
- 单品的流行度。有很多评价流行度的方法，可以在不同的时间维度上计算，如小时、天、周等级别。或者，可以将用户按照地域或其他类似的指标分类，然后计算这一群人中的流行度。
- 播放情况，如时长、播放的时间、设备类型。
- 会员将视频放入队列的情况。
- 丰富的元数据，如演员、导演、体裁、父母的打分、评价。
- 根据将视频展现给用户的方式，观察用户的行为，如鼠标滚动、悬停、点击，或者是用户在页面停留的时间。
- 社交数据
- 搜索数据
- 除上面提及的内部数据以外，还有一些外部数据。如票房、评价等。
- 其他，如人口特征、地域、语言、时间等。

### 用到的部分算法

- Linear regression
- Logistic regression
- Elastic nets
- Singular Value Decomposition
- Restricted Boltzmann Machines
- Markov Chains
- Latent Dirichlet Allocation
- Association Rules
- Gradient Boosted Decision Trees
- Random Forests
- Clustering techniques from the simple k-means to novel graphical approaches such as Affinity Propagation
- Matrix factorization

### Consumer Data Science

简要的一套评估流程：
1. 离线实验，拿到离线指标收益的结论。
2. 在线实验，拿到实验数据收益的结论。
3. 推全。

适用于无论是新加特征还是新的算法。

![简要流程](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*30KWZ38MzCXDB9AXjMtGjA.png)

## 三、System Architectures for Personalization and Recommendation

Reference: [System Architectures for Personalization and Recommendation](https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8)

本文主要讲述在为Netflix个性化和推荐服务的系统架构的设计。这篇笔记主要内容：
- 重点学习在线、离线、近线计算的设计；
- 顺带讲述文中模型与信号、事件和数据分发、推荐结果的内容。

### 1 简介

**在线（Online）计算**可以比较好的响应最近的事件（event）和用户交互，但是必须是实时（real-time）的进行。但是，它的计算复杂度和使用的数据量都是受限的。

**离线（Offline）计算**是批量形式运行，且时间要求比较低，所以它的计算复杂度和使用的数据量的限制比较少。但是，它在每个更新之间是比较稳定的，因为最近的数据是没有被使用的。*即，不能根据用户的反馈做出较快的反应。*

结合管理在离线计算的核心办法：**近线（Nearline）计算**。近线计算是在线和离线计算的折中方法，它可以进行类似在线的计算，但是又不要求其实时响应。

![3-1 总体架构图](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*qqTSkHNOzukJ5r-b54-wJQ.png)

### 2 离线、近线、在线计算

![3-2](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bm0urYNwRhvPap19IkWfVA.png)

算法的计算结果可以通过实时的在线计算、批量的离线计算、或介于两者之间的近线计算。每种方法都有自己的优势和劣势，需要在不同的使用场景下分别考虑。

**在线计算**可以迅速地对事件做出反应，并使用最新的数据。它受限于可用性和响应延迟时间的SLA（Sevice Level Agreements）。所以使用复杂的和计算代价昂贵的算法是比较困难的。另外，在某些特定的情况下，单纯的在线计算可能也难以满足SLA，所以要有一个快速的fallback机制，比如说revert到一些提前计算好的结果上。在线计算也意味着各种各样的数据需要在线访问得到，这也需要更高数量的基础设施。

**离线计算**允许使用更复杂的算法和更多的数据。它对响应时间SLA的要求也比较宽松。一个新的算法可以不用考虑太多的性能优化就可以在生产环境部署。这对敏捷创新形成有力的支撑。*作者举例，在Netflix，如果一个算法执行起来比较慢，可以简单地给加一些机器。而不是耗费宝贵的工程时间来给他调优，然后实验结果发现它并不是一个具有业务价值的算法。*因为离线没有比较强的延迟要求，它不会对上下文的变化或新的数据做出快速的反应。离线计算也需要存储、计算、访问大规模预计算结果的基础设施。

**近线计算**按照在线的情景进行计算。但是，不需要立即将计算的记过进行返回，而是将它们存储上，异步地处理。针对用户事件，来进行近线计算，所以系统可以在请求之间更加地反应敏捷。这就为我们打开了对每个事件进行更进行更复杂计算的大门。*举个例子，一个用户刚开始观看一个电影时，就会推荐就会更新来反应这个事件。*这些结果可以被存储在临时的缓存或者后端的存储上。近线计算也是是应用<font color=red>**增量学习算法（incremental learning algorithm）**</font>的天然设定。

使用在线、近线、离线方法并不是一个选择题，而是应该将它们结合起来。比如，在离线将一些结果提前计算出来，而将不是特别复杂的或者上下文敏感的信息进行在线计算。

即使是模型也可以在离线结合起来。对传统的有监督的分类算法不合适，但是像<font color=red>**矩阵分解（Matrix Factorization）**</font>这样的算法更适合将在离线结合起来：一些factor离线计算出来结果，然后其他的在线更新来得到一个更新的结果。另外一个例子是<font color=red>**无监督的方法，比如聚类**</font>。这些例子指明了可以讲我们的模型训练分为大规模的、可能复杂的全局的模型训练，和可以在线计算的一个轻量级的、根据特定用户的模型训练或跟新。

### 3 离线任务

主要是两种任务：模型训练和中间或最终结果的批量计算。

模型训练一般是离线的批量完成，但是也有一些**在线学习（Online Learning）**的技巧，在线进行增量训练。结果的批量计算是利用产出的模型和响应的输入，来计算将来计算将来在线处理用到的数据或是直接给用户的数据。

#### 体会

区分在离线的主要目的，模型训练等的任务比较耗时、复杂。

#### Netflix技术栈

大数据、分布式系统：Hadoop + Hive/Pig

发布数据的要求：
1. 告诉订阅者结果ready了。
2. 支持不同的仓库，如HDFS、S3、Cassandra
3. 处理错误，如监控与报警。

Netflix.Hermes（工具名称）

![Netflix.Hermes](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8lw3s6XZxOUTGM0yVJLJxA.png)

### 4 信号与模型

这里的信号指的是输入算法的新鲜数据。这些数据是通过在线服务获得的，并且是由用户相关信息组成的，*如用户最近看了啥，session、设备、日期、事件等上下文数据*。

### 5 事件与数据分发

事件指的是一些时间敏感的小单元，需要以极低的延迟来处理的。这些事件用来出发一系列接下来的动作或处理，如近线结果等。

数据则是信息密集度很高的单元，也许需要处理然后为将来的使用而存储起来。这里延迟就不如信息的数量、质量那么重要了。

#### 技术栈

准实时的事件流：Netflix.Manhattan。它是一个分布式计算系统，是推荐算法架构的中心。

数据流由使用Chukwa到Haddop的日志流来管理，用于初步的处理。接下来使用Hermes来作为发布-订阅机制。

![Netflix.Manhattan](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*dzpP6wfwejPQ-xKUDJlukg.png)

### 6 推荐结果

#### Netflix技术栈

使用多种仓库来保存离线和中间结果，主要的数据存储是Cassandra、EVCache、MySQL：
- MySQL存储结构化的关系数据，可能需要通过通用目的查询来进行处理。但是，通用性的代价是在分布式环境下可扩展性的问题。
- Cassandra和EVCache都是KV数据库。当是使用分布式、no-SQL存储，它就是众所周知的标准答案。
- 在需要高强度和持续的写操作的情况下，EVCache更好。

核心观点：不是要找出将数据存储在哪里，而是要平衡各种有冲突的目标的要求，如查询复杂度、读写延迟、事物一致性等，最终对每一种使用情况都找到一个最优解。

![Serve](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Y1oZvmQnwc7lJTL07Q4eXg.png)

## 四、后记

此篇笔记是阅读了2013年Netflix在网上发布的几篇技术博客。可以看到，其中的一些内容已经成了基础知识或者行业共识。甚至有一些技术已经略显过时。但是这里学习的原因就是了解一下它的总体系统架构的设计，和一些关键思路想法的总结。希望也能给读者带来一些启发。