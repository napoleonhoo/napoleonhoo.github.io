# 做训练

## 1 名词解释

- working root：存储在AFS上的模型相关文件的存储目录。

- donefile：存储在AFS的batch model（checkpoint）的相关信息，如模型路径等。

- batch model：离线训练的checkpoint，模型训练任务失败后可以从这个点重启。

- base：xbox（cube）配送的全量模型，供在线服务（predictor）使用，一般是一天一个base。

- patch：xbox（cube）配送的增量模型，供在线服务使用，一般是一天多个patch。
