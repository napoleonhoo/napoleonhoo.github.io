# § Ch 1. coordinator

## 名词简介

1. working root：存储在AFS上的模型相关文件的存储目录。
2. donefile：存储在AFS的batch model（checkpoint）的相关信息，如模型路径等。
3. batch model：离线训练的checkpoint，模型训练任务失败后可以从这个点重启。
4. base：xbox（cube）配送的全量模型，供在线服务（predictor）使用，一般是一天一个base。
5. patch：xbox（cube）配送的增量模型，供在线服务使用，一般是一天多个patch。

## 主流程

1. `OnlineRunner::run()`，入口函数。*注意协程（coroutine）的使用。注意elastic库中相关东西。*
2. `_dataset_pass_id`初始化，赋值为0。
3. `check_path_exists()`，检查working root、donefile是否存在。*注意fs_utils相关内容。*
4. `SlaveManager::start()`
   1. 向elastic server请求指定数量的worker，并等待最小数量（coordinator.yaml配置决定）的worker调度成功。*注意ElasticSlaveScheduler类。另外，这里的elastic Server指的是MPI 集群的APP master。*
   2. 向elastic server请求指定数量的server，并等待全部数量的server调度成功。后台持续检测server是否alive。检测标准为，上一次的<font color=red>heartbeat时间</font>是否超过了配置的`heartbeat_timeout_s`，如果超过了的话，会abort整个任务。
5. 启动brpc server，这个使用来和server、Worker通信的。
6. `get_last_status`，从donefile读取最后一行数据。donefile的主要内容有以下几部分，从左到右分别是：day（保存checkpoint的日期）、时间戳（也是xbox的key，指明了xbox对应的checkpoint）、checkpoint的AFS路径、产出此checkpoint的delta id、batch model id、patch model id。<font color=red>确认？</font>需要patch model id等于batch model id。
7. 训练数据预取。*注意coroutine::Channel*根据配置的路径相关信息，生成要取的训练数据的路径，并查看数据是否准备完成。如果已经准备完了的话，调用`SlaveManager::prefetch()`。
8. 加载模型checkpoint。和上一步是同步进行的。

