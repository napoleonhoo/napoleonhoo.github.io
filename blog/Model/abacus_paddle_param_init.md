# abacus与paddle的参数初始化方式

这里主要讲的是FC(Fully Connected，全连接)层的weight和bias的初始化方式。

## 1 abacus

### 1.1 weight初始化

以**标准正态分布**随机初始化为基础，并乘以相应的scale。具体代码如下：

``` cpp
// in file: dnn/param_layer.cpp

double init_range = conf["init_range"].as<double>() * global_init_range;
if (conf["scale_by_rown"].as<bool>()) {
    init_range /= sqrt(double(conf["rown"].as<int>()));
}
int len = conf["rown"].as<int>() * conf["coln"].as<int>();
std::vector<float> w;
w.clear();
std::normal_distribution<float> ndistr(0.0, 1.0);
for (int i = 0; i < len; ++i) {
    w.push_back(ndistr(abacus::local_random_engine()) * init_range);
}

```

一般配置有：

``` yaml
init_range: 1
global_init_range: 0.2
scale_by_rown: true
```

总结公式有：

$$
w[i] = \mathcal{N}(0.0, 1.0) \times init\_range \times global\_init\_range \times \sqrt{row\_number}
$$

其中，$row\_number$为参数的行数。

### 1.2 bias初始化

直接初始化为**1**。具体代码如下：

``` cpp
// in learner/dnn_plugin.cpp

Eigen::MatrixXf* bias_mat = get_value_mat(dnn_ins, "bias_input");
if (bias_mat) bias_mat->setOnes(thr->ins_num, 1);
```

## 2 paddle

可以在组网中自定义，参考文档[fc的使用](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/fc_cn.html#fc)和[create_parameter的使用](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/create_parameter_cn.html#create-parameter)。*上述文档为paddle1.8版本*

这里节选部分模型的参数初始化代码，如下：

``` python
    # 一跳精排，其中init_range = 0.2
    def common_tower(self, bn_video, bn_news, gate_merge, prefix):
        """common tower"""
        input_orig = paddle.concat([bn_video, bn_news], axis=1)
        input_tmp = paddle.multiply(input_orig, gate_merge) 
        fc_layers_input = [input_tmp]
        fc_layers_size = [128, 1]
        fc_layers_act = ["relu"] * (len(fc_layers_size) - 1)  + [None]
        scales_tmp = [bn_video.shape[1] + bn_news.shape[1]] + fc_layers_size
        scales = []
        for i in range(len(scales_tmp)):
            scales.append(self._config.init_range / (scales_tmp[i] ** 0.5))
        for i in range(len(fc_layers_size)):
            name = prefix + "_fc_" + str(i)
            logits = paddle.static.nn.fc(
                    x = fc_layers_input[-1],
                    size = fc_layers_size[i],
                    weight_attr = paddle.ParamAttr(learning_rate=1.0, \
                        initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 * scales[i])),
                    bias_attr = paddle.ParamAttr(learning_rate=1.0, \
                        initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 * scales[i])),
                    name=name)
            if fc_layers_act[i]:
                name = prefix + "_ln_" + str(i)
                bn = paddle.static.nn.layer_norm(input=logits, begin_norm_axis=1, name=name)
                output = F.relu(bn, name="ctx_dnn_embedding")
            else:
                output = logits
            fc_layers_input.append(output)
    
    # 统一粗排，其中_init_range = 0.2
    def deep_net_origin(self, input, fc_layers_size, prefix, last_None=False, lr_x=1.0):
        """ 重复函数，待合并 """
        fc_layers_input = [input]
        if last_None:
            fc_layers_act = ["relu"] * (len(fc_layers_size) - 1) + [None]
        else:
            fc_layers_act = ["relu"] * (len(fc_layers_size))
        scales_tmp = [input.shape[1]] + fc_layers_size
        scales = []
        for i in range(len(scales_tmp)):
            scales.append(self._init_range / (scales_tmp[i] ** 0.5))
        for i in range(len(fc_layers_size)):
            name = prefix + "_" + str(i)
            fc = fluid.layers.fc(input = fc_layers_input[-1], size = fc_layers_size[i], act = fc_layers_act[i],
                                param_attr = fluid.ParamAttr(learning_rate=lr_x, \
                                    initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                                bias_attr = fluid.ParamAttr(learning_rate=lr_x, \
                                    initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                                name=name)
            fc_layers_input.append(fc)

        return fc_layers_input[-1]
```

可以看到，在具体实践上，初始化方式基本上都是**正态分布**，均值为0，标准差做了一些scale。且weight和bias的参数初始化方式相同。可以总结基本公式：

$$
w[i] = \mathcal{N}(0.0, 1.0 \times \frac{init\_range}{shape^{0.5}})
$$

其中，$shape$为输入的shape、fc的大小组成的数组。*具体参见代码*

## 3 异同点

### 相同点

都是以均值为1的正态分布作为基础。

### 不同点

abacus是标准正态分布随机值之后进行scale，paddle是对正态分布的标准差进行scale。

*<font color=red>具体为啥这么搞？暂时还没弄明白。</font>*