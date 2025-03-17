import torch
from torch_sparse import SparseTensor
from models.LightGCN_model import bpr_loss
from models.A2C_model import *
import dataloader
import evaluate
import tqdm


class Client:
    # 构造函数
    def __init__(self, device, conf, model, train_edge_index, test_edge_index, num_users, num_movies,
                 local_epoch, embedd_ek, idx=1):
        # 本地强化学习模型
        self.local_rl = rl_model
        # 配置文件
        self.conf = conf
        self.num_movies = num_movies
        self.num_users = num_users
        # 判断是在GPU还是CPU上训练
        self.device = device
        # 客户端本地模型(一般由服务器传输)
        self.local_model = model.to(self.device)
        # 客户端ID
        self.client_id = idx
        # 本地的循环次数
        self.local_epoch = local_epoch
        # 本地的三层权重
        self.embedd_ek = embedd_ek
        # 按ID对训练集合的拆分 得到本地训练集
        len_local_train = train_edge_index.size()[1]  # 本地数据集的长度
        all_range = list(range(len_local_train))  # all_range的是一个所有用户的编号表
        data_len = int(len_local_train / self.conf['no_models'])  # 计算出单个服务器所需要的用户数量
        indices = all_range[idx * data_len: (idx + 1) * data_len]  # 将这个服务器需要的用户编号，并保存到indices
        self.train_edge_index = train_edge_index[:, indices]
        # 本地测试集
        self.test_edge_index = test_edge_index
        # 本地训练集和测试集的离散化矩阵
        self.train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
            num_users + num_movies, num_users + num_movies)).to(self.device)
        self.val_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
            num_users + num_movies, num_users + num_movies)).to(self.device)
        # 用来存储训练的loss
        self.train_losses = []
        self.val_losses = []
        self.action_space = [i for i in range(self.conf["action_space"])]
        self.rl_model = ActorCritic(num_inputs=128, num_outputs=self.conf["action_space"],)

    # 模型本地训练函数
    def local_train(self, model):
        # 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到
        for name, param in model.state_dict().items():
            # 客户端首先用服务器端下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())

        # 定义最优化函数器用于本地模型训练
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        # 本地训练模型
        self.local_model.train()  # 设置开启模型训练

        # 开始训练模型
        for iters in tqdm.tqdm(range(1, self.local_epoch + 1)):
            # 先进行随机采样 生成负标签
            user_indices, pos_item_indices, neg_item_indices = \
                dataloader.sample_mini_batch(self.conf["batch_size"], self.train_edge_index, self.num_movies)
            # 将数据传入设备中
            user_indices, pos_item_indices, neg_item_indices = \
                user_indices.to(self.device), pos_item_indices.to(self.device), neg_item_indices.to(self.device)
            # 将数据传入模型 得到全局embedding
            users_emb_final, users_emb_0, items_emb_final, items_emb_0, lr_input = \
                self.local_model.forward(self.train_sparse_edge_index)
            # 得到本次训练的embedding
            users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
            # print(users_emb_final.shape, items_emb_final.shape)
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
            # 计算函数损失，并更新参数
            train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                  pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, self.conf["LAMBDA"])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # 如果训练到达ITERS_PER_EVAL次，本次进行测试
            if iters % self.conf["ITERS_PER_EVAL"] == 0:
                self.local_dev(train_loss, iters)

        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        for name, datas in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (datas - model.state_dict()[name])
        print("Client %d local train done" % self.client_id)
        # 得到本地推荐列表并上传到基站
        top_m_recommend = self.find_top_m()
        # 客户端返回差值
        return diff, top_m_recommend

    # 本地模型测试函数
    def local_dev(self, train_loss, iters):
        self.local_model.eval()
        val_loss, recall, precision, ndcg = evaluate.evaluation(
            self.local_model, self.test_edge_index, self.val_sparse_edge_index,
            [self.train_edge_index], self.conf['recommend_m'], self.conf['LAMBDA'], self.num_movies)
        print(f"[Iteration {iters}/{self.conf['local_epochs']}] train_loss: {round(train_loss.item(), 5)}, "
              f"val_loss: {round(val_loss, 5)}, val_recall@{self.conf['recommend_m']}: {round(recall, 5)}, "
              f"val_precision@{self.conf['recommend_m']}: {round(precision, 5)}, val_ndcg@{self.conf['recommend_m']}: {round(ndcg, 5)}")
        self.train_losses.append(train_loss.item())
        self.val_losses.append(val_loss)
        self.local_model.train()

    # 生成一个本地推荐列表，向服务器推荐m2个电影
    def find_top_m(self):
        # 运行模型，得到用户嵌入和物品嵌入
        users_emb_final, users_emb_0, items_emb_final, items_emb_0, lr_input = \
            self.local_model.forward(self.train_sparse_edge_index)
        user_embedding = users_emb_final
        item_embedding = items_emb_final

        # 通过相乘得到每个用户对电影的评分矩阵
        rating = torch.matmul(user_embedding, item_embedding.T)

        for edge_index in [self.train_edge_index]:
            # 得到每个用户喜欢的电影（先前缓存的电影）
            user_pos_items = evaluate.get_user_positive_items(edge_index)
            # 将原来已经有的电影去除
            exclude_users = []
            exclude_items = []
            for user, items in user_pos_items.items():
                exclude_users.extend([user] * len(items))  # 2*N的格式 [0]为用户 [1]为对应的电影
                exclude_items.extend(items)

            # 将已经观看的电影缓存去掉（设置一个很小的数）
            rating[exclude_users, exclude_items] = -(1 << 10)

        # 得到每个用户新的recommend_m1个推荐的电影
        _, top_k_items = torch.topk(rating, k=self.conf["recommend_m"])
        return top_k_items

    def choose_action(self, observation):
        probs, _ = self.local_rl(observation)
        action = probs
