import torch
import numpy as np


def top_k_frequent_numbers(matrix, num_k):
    """
    :param matrix: 输入的矩阵
    :param num_k: 提取出num_k个最常出现的数字
    :return: 返回一个列表，存储最常出现的num_k个数字
    """
    flattened = torch.flatten(matrix)
    unique, counts = torch.unique(flattened, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    top_k_indices = sorted_indices[:num_k]
    top_k_numbers = unique[top_k_indices]
    return top_k_numbers


def count_occurrences(array1, array2):
    """
    计算第一个数组中第二个数组中数字出现的次数
    参数:
        array1: torch.tensor, 第一个数组，包含很多数字
        array2: torch.tensor, 第二个数组，包含一部分数字
    返回值:
        tot_hit: 返回命中物品的数量
    """

    # 得到一个物品是否命中的列表，其中值为true和false
    indices = torch.isin(array1, array2)
    # 计算命中多少个物品
    tot_hit = torch.sum(indices)
    return tot_hit


# 计算缓存效率并返回
def compute_cache_efficiency(test_edge_index, num_test_movies, item_top_k):
    # 命中次数
    hit_num = count_occurrences(test_edge_index[1], item_top_k)
    # 计算缓存效率 命中电影数量/总的请求电影数量
    cache_efficiency = hit_num/num_test_movies
    return cache_efficiency


def top_n_indices(arr, n):
    """
    返回数组前n个最大值的下标
    """
    top_indices = np.argpartition(arr, -n)
    top_indices_n = top_indices[0, -n:]
    # 返回前n个最大值的下标
    return top_indices_n


def top_n_indices1(arr, n):
    """
    返回数组前n个最大值的下标
    """
    top_indices = np.argpartition(arr, n)
    top_indices_n = top_indices[-n:]
    # 返回前n个最大值的下标
    return top_indices_n
