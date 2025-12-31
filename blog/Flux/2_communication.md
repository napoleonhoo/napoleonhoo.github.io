# 第二章 通信

这一章从coordinator启动开始，讲述coordinator及相关组件所有的通信，由于coordinator是整个训练流程的调度者，这就意味着所有通信的最先发起者必是coordinator。

## 1. 以coordinator为中心的通信

1. coordinator向elastic server提交对worker、server的请求。

- coordinator=>elastic server：worker、server的节点数、资源quota、启动命令等。
- elastic server=>coordinator：返回status，0表示请求成功。

2. coordinator要求worker加载小模型。

- coordinator=>single worker：参数包含no path、DNN PLUGIN MODEL。
- single worker=>all server：dense的维度。
- server=>worker: 返回status，0代表push的dense的维度，和load的dense table维度一致。
- worker=>coordinator：返回status，0代表push成功。

3. coordinator向elastic server提交自己的ip。（why？）

- coordiantor=>elastic server：coordinator的ip。
- elastic=>coordinator：status=0代表验证成功。

4. coordinator要求worker预取数据。

- coordinator=>all worker：要读取的dataset的相关信息。
- worker=>coordinator：status=0代表预取成功。

5. coordinator要求server加载模型。

- coordinator=>server：参数包含model path，BATCH MODEL。
- server=>coordinator：status=0代表加载成功。

6. coordinator要求worker将dense参数push到server上。逻辑同2中所述。
7. coordinator要求worker训练样本。

- coordinator=>all worker：训练样本的路径。（如果没有预取完，还会按照4中流程继续预取）
- worker=>all server：请求dense参数。
- server=>worker：status=0，代表请求成功，并返回参数。
- worker=>all server：请求sparse参数。
- server=>worker：status=0，代表请求成功，并返回参数。
- worker=>all server：推送sparse梯度。
- server=>worker：status=0，代表推送成功。
- worker=>all server：推送dense梯度。
- server=>worker：status=0，代表推送成功。
- worker=>coordinator：status=0，代表训练成功。

8. coordinator要求保存xbox patch model。

- coordinator=>server：save XBOX PATCH MODEL。
- server=>coordinator：status=0，代表save成功。
- coordiantor=>server：获取cache threshold。
- server=>coordinator：返回cache threshold。
- coordinator=>server：save XBOX CACHE BATCH MODEL，同时开启stream RPC receiver。
- server=>coordinator：sparse cache的具体内容。
- coordinator：接收完毕关闭stream。
- coordinator=>worker：save DNN PLUGIN MODEL。
- worker=>coordinator：status=0，代表save成功。

9. coordinator要求server保存batch model。

- coordinator=>server：save BATCH MODEL。
- server=>coordinator：staus=0，代表save成功。

10. coordinator要求保存XBOX BASE MODEL。流程上和8相同。
10. coordinator请求elastic server迁移某个实例。

- coordinator=>elastic server：要迁移的实例的id。
- elastic server=>coordinator：status=0，代表请求成功。

## 2. worker主动发起的通信

1. worker向coordinator请求server的ip列表。
2. worker向coordinator请求注册自己为worker。
3. worker向coordinator发送heartbeat。

## 3. server主动发起的通信

1. server向coordinator请求注册自己为server。
2. server向coordinator发送heartbeat。

## 4. 一些注意点和疑问

1. coordinator向elastic server申请节点，返回的status只代表是否请求成功。并不返回具体的slave的节点ip等信息。
2. worker、server是在**本机的环境变量**中找到coordinator的ip。
3. coordinator为什么要向elastic server通报自己的ip。

> dump_address是调用了app_master提供的一个简易的kv服务，server/worker启动时会调用这个接口从app_master上获取coordinator地址信息。

