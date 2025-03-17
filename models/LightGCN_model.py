import torch
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing


class LightGCN(MessagePassing):
    def __init__(self, device, num_users, num_items, embedd_ek, embedding_dim=64, K=3, add_self_loops=False, **kwargs):
        """
        Args:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int, optional): 嵌入维度，设置为64，后续可以调整观察效果
            K (int, optional): 传递层数，设置为3，后续可以调整观察效果
            add_self_loops (bool, optional): 传递时加不加自身节点，设置为不加
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops  # 是否需要加入自身节点，原来的LightGCN中是不加入
        self.device = device
        # 使用强化学习来学习embedd_ek也就是三个聚合时候的权重
        self.embedd_ek = embedd_ek
        # 使用embedding来生成图嵌入
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)  # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """
        Args:
            edge_index (SparseTensor): 邻接矩阵
        Returns:
            tuple (Tensor): e_u%^k, e_u^0, e_i^k, e_i^0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
        embs = [emb_0]  # 之所以数量是4是因为一开始把初始化的emb也加入进去了
        emb_k = emb_0

        # 多尺度扩散
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        # print(embs.shape)
        # emb_final = torch.mean(embs, dim=1)
        # emb_final = torch.matmul(embs, self.embedd_ek)
        weighted_tensor = embs * self.embedd_ek[:, None]  # E^K 使用强化学习得到的权重进行加权相加
        # 沿着 dim=1 对加权后的张量求和，得到形状为 [33736, 64] 的结果
        emb_final = torch.sum(weighted_tensor, dim=1)
        # print(emb_final.shape)
        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        first_row_tensor1 = users_emb_final[0, :]
        first_row_tensor2 = items_emb_final[0, :]
        # 将两行拼接成一列
        concatenated = torch.cat((first_row_tensor1.unsqueeze(1), first_row_tensor2.unsqueeze(1)), dim=0)

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight, concatenated

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t.to(self.device), x)

    def upgrade_em(self, embed_ek):
        self.embedd_ek = embed_ek


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    """
    Args:
        users_emb_final (torch.Tensor): e_u^k
        users_emb_0 (torch.Tensor): e_u^0
        pos_items_emb_final (torch.Tensor): positive e_i^k
        pos_items_emb_0 (torch.Tensor): positive e_i^0
        neg_items_emb_final (torch.Tensor): negative e_i^k
        neg_items_emb_0 (torch.Tensor): negative e_i^0
        lambda_val (float): λ的值
    Returns:
        torch.Tensor: loss值
    """

    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2 loss L2范数是指向量各元素的平方和然后求平方根

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # 正采样预测分数
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # 负采样预测分数

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss
