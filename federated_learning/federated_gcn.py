import json
import random
from models import LightGCN_model
from client import *
from server import *
import data_pre
import matplotlib.pyplot as plt
from models.A2C_model import *

if __name__ == '__main__':
    embed_ek = [0.21, 0.23, 0.25, 0.27, 0.29]
    local_epoch = [500, 600, 700, 800, 900]
    init_embed = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # 读取配置文件
    with open("../conf.json", 'r') as f:
        conf = json.load(f)
    device = torch.device('cpu')  # 选择设备，cpu或者gpu

    # 数据预处理 以及测试集中电影的长度
    num_users, num_movies, train_edge_index, test_edge_index = data_pre.pre_datasets(conf)
    # 获取所有不重复的元素
    unique_elements = torch.unique(test_edge_index[1])
    # 计算不重复元素的数量
    num_test_movies = unique_elements.size(0)
    print(num_test_movies, num_movies)
    num_test_movies = test_edge_index.shape[1]
    # Oracle 最优方法 推荐测试集中出现次数最多的电影
    oracle_top_k = top_k_frequent_numbers(test_edge_index[1], conf['recommend_k'])
    oracle_cache_efficiency = compute_cache_efficiency(test_edge_index, num_test_movies, oracle_top_k)

    # Random 最差方法 使用随机的方法来生成recommend_k个推荐项目，缓存到服务器中
    random_top_k = torch.randint(1, num_movies + 1, (1, conf['recommend_k']))
    random_cache_efficiency = compute_cache_efficiency(test_edge_index, num_test_movies, random_top_k)

    print(oracle_cache_efficiency, random_cache_efficiency)  # 输出最佳方法和一般方法之间的对比

    # 开启服务器
    server = Server(LightGCN_model.LightGCN(device, num_users, num_movies, embedd_ek=init_embed), device, conf,
                    train_edge_index, test_edge_index, num_users, num_movies, rl_model=ActorCritic())
    # 客户端列表
    clients = []

    # 添加n_models个客户端到列表
    for c in range(conf["no_models"]):
        clients.append(
            Client(device, conf, LightGCN_model.LightGCN(device, num_users, num_movies, embedd_ek=init_embed),
                   train_edge_index, test_edge_index, num_users, num_movies,
                   idx=c, local_epoch=400, embedd_ek=init_embed, rl_model=ActorCritic()))  # 在这里调整本地循环次数
    print("\n\n")

    # 用于记录每一层的缓存效率，将结果存储到一个列表中
    global_cache_efficiency = []

    # 全局模型训练
    for e in range(1, conf["global_epochs"] + 1):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样no_client个进行本轮训练
        candidates = random.sample(clients, conf["no_client"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id, end=' ')
        print('\n')

        # 权重累计和本地推荐集
        weight_accumulator = {}
        client_recommend = torch.empty((0, conf["recommend_m"]))

        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)

        # 遍历客户端，每个客户端本地训练模型，获取推荐列表
        for c in candidates:
            diff, top_m_recommend = c.local_train(server.global_model)  # 获取每一个推荐电影的列表
            client_recommend = torch.cat((client_recommend, top_m_recommend), dim=0)  # 将它们整合
            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 计算云端需要缓存的recommend_k个资源
        server_top_k = server.cache_judge(client_recommend)
        cache_efficiency = compute_cache_efficiency(test_edge_index, test_edge_index.shape[1], server_top_k)
        print("[GLOBAL-MODEL-DEV]Cache efficiency:", cache_efficiency, "Total num:", num_test_movies)

        # 添加到列表中用来绘制图形
        global_cache_efficiency.append(cache_efficiency)

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)
        server.model_eval()

    # 绘制图像，以及调用测试函数
    plt.plot(global_cache_efficiency)
    plt.plot([oracle_cache_efficiency] * conf["global_epochs"])
    plt.show()
