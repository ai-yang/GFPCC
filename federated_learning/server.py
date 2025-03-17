import torch
from torch_sparse import SparseTensor
import evaluate
from utils import *


class Server:
    # 定义构造函数
    def __init__(self, model, device, conf, train_edge_index, test_edge_index, num_users, num_movies, rl_model):
        # 导入配置文件
        self.conf = conf
        self.device = device
        self.num_movies = num_movies
        # 根据配置获取模型文件
        self.global_model = model.to(self.device)
        # 生成一个测试集合加载器
        self.test_edge_index = test_edge_index
        self.train_edge_index = train_edge_index
        self.test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                                   sparse_sizes=(num_users + num_movies, num_users + num_movies))

    # 全局聚合模型
    # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一层乘上学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]  # 这里是原始的fed_avg函数
            # 累加和
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 评估函数
    def model_eval(self):
        self.global_model.to(self.device).eval()  # 开启模型评估模式（不修改参数）
        val_loss, recall, precision, ndcg = evaluate.evaluation(
            self.global_model, self.test_edge_index, self.test_sparse_edge_index,
            [self.train_edge_index], self.conf['recommend_m'], self.conf['LAMBDA'], self.num_movies)
        print(f"[GLOBAL-MODEL-DEV]"
              f"val_loss: {round(val_loss, 5)}, val_recall@{self.conf['recommend_m']}: {round(recall, 5)}, "
              f"val_precision@{self.conf['recommend_m']}: {round(precision, 5)}, val_ndcg@{self.conf['recommend_m']}: {round(ndcg, 5)}")

    # 计算基站需要缓存的recommend_k个资源
    def cache_judge(self, client_recommend):
        # 得到一个需要缓存在服务器端的推荐列表
        top_k_recommend = top_k_frequent_numbers(client_recommend, self.conf["recommend_k"])
        return top_k_recommend
