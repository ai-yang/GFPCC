import pandas as pd
import torch
from torch_geometric.utils import structured_negative_sampling
import random


def load_node_csv(path, index_col):
    """
    Args:
        path (str): 数据集路径
        index_col (str): 数据集文件里的列索引
    Returns:
        dict: 列号和用户ID的索引、列号和电影ID的索引
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}  # enumerate()索引函数,默认索引从0开始
    return mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    """
    Args:
        path (str): 数据集路径
        src_index_col (str): 用户列名
        src_mapping (dict): 行号和用户ID的映射
        dst_index_col (str): 电影列名
        dst_mapping (dict): 行号和电影ID的映射
        link_index_col (str): 交互的列名(一般是评分，如果大于rating_threshold则看成是一条边)
        rating_threshold (int, optional): 决定选取多少评分交互的阈值，设置为4分
    Returns:
        torch.Tensor: 2*N的用户电影交互节点图
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(
        torch.long) >= rating_threshold  # 将数组转化为tensor张量  edge_attr储存的是布尔类型，用于判断是否要将当前边加入训练边集
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:  # 如果这个边的评分大于4则加入边集
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


def sample_mini_batch(batch_size, edge_index, num_items):
    """
    Args:
        :param batch_size: 批大小
        :param edge_index: 2*N的边列表
        :param num_items: 所有物品的数量
    """
    edges = structured_negative_sampling(edge_index, num_items)  # 返回的是三个元组 所以需要stack来组合起来
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices
