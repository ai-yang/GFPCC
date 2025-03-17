import pandas as pd
import random
import tqdm
from utils import *
from data_pre import pre_datasets
import json

#  加载配置文件和数据集
with open("../conf.json", 'r') as f:
    conf = json.load(f)

# 加载movielens100K数据集
ratings_data = pd.read_csv('../database/movielens_1M/ratings.csv')
num_users, num_movies, train_edge_index, test_edge_index = pre_datasets(conf)

# 计算每个项目的平均评分和总评分数
ratings_mean = ratings_data.groupby(['movieId'], as_index=False)['rating'].mean()
ratings_count = ratings_data.groupby(['movieId'], as_index=False)['rating'].count()


# 定义Thompson sampling方法
def thompson_sampling(alpha, beta):
    # 从Beta分布中随机取样
    return np.random.beta(alpha, beta)


# 定义推荐函数
def recommend(ratings_data, n, epochs):
    # 初始化每个项目的正向和反向计数器
    positive_counts = np.ones(num_movies)
    negative_counts = np.ones(num_movies)

    # 进行epochs次Thompson sampling
    for i in tqdm.tqdm(range(epochs)):
        # 随机选择用户和项目
        user_id = random.randint(1, num_users+1)
        item_id = random.randint(1, num_movies+1)

        # 获取用户对项目的评分
        rating = ratings_data[(ratings_data['userId'] == user_id) & (ratings_data['movieId'] == item_id)][
            'rating'].values

        # 如果用户评分存在，则更新正向或反向计数器
        if len(rating) > 0:
            rating = rating[0]
            if rating >= 4.0:
                positive_counts[item_id] += 1
            else:
                negative_counts[item_id] += 1

    # 从Beta分布中采样每个项目的胜利概率
    victory_probabilities = thompson_sampling(positive_counts, negative_counts)

    # 获取前N个胜利概率最高的项目的ID列表
    top_item_ids = ratings_data.iloc[np.argsort(-victory_probabilities)][:n]['movieId'].tolist()

    return top_item_ids


# 超参数
epoch = 100000
# 推荐前recommend_k个最受欢迎的项目的ID列表
tf_top_k = torch.Tensor(recommend(ratings_data, conf['recommend_k'], epoch))
tf_cache_efficiency = compute_cache_efficiency(test_edge_index, test_edge_index.shape[1], tf_top_k)
print(tf_top_k, tf_cache_efficiency)
