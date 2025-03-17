## 这个项目中云端只是用来聚合模型
本地还有DEV测试集来计算本地精确度


### 配置文件
1. 模型名称，用来选择使用哪一种模型
2. tpye:数据集名称 原来使用的是cifar 现模型使用Movielens
3. k是随机采样的客户端数量 后续可以使用强化学习来选择客户端
4. momentum是动量， 是否采用该优化方法
5. lambda: 是更新时候乘以的一个超参数,update_per_layer = weight_accumulator[name] * self.conf["lambda"]
6. 数据集名称可以是100K,1M,DVD,一共三个数据集

_developed by Reece_