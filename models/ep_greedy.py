import pandas as pd
from data_pre import pre_datasets
import json
from utils import *
import tqdm


# 定义epsilon-greedy算法函数
def epsilon_greedy(rewards, ep):
    if np.random.rand() > ep:
        return np.random.choice(np.arange(len(rewards)))
    else:
        return np.argmax(rewards)


#  加载配置文件和数据集
with open("../conf.json", 'r') as f:
    conf = json.load(f)

# 读入包含用户评分信息的DataFrame
ratings_df = pd.read_csv('../database/movielens_1M/ratings.csv')
num_users, num_movies, train_edge_index, test_edge_index = pre_datasets(conf)

# 基于用户评分数据生成电影评分矩阵
movie_ratings = pd.pivot_table(ratings_df, values='rating', index='userId', columns='movieId')

# 将电影评分矩阵转换为numpy数组
movie_ratings_array = np.nan_to_num(movie_ratings.values)

# 初始化epsilon-greedy算法参数
epsilon = 0.1
num_episodes = 200000
num_actions = movie_ratings_array.shape[1]

# 初始化Q值
Q = np.zeros(num_actions)

# 迭代num_episodes次
for episode in tqdm.tqdm(range(num_episodes)):
    # 随机选择一个用户
    user_id = np.random.choice(num_users)

    # 选择一个行动
    action = epsilon_greedy(Q, epsilon)

    # 获得此用户对此电影的评分
    reward = movie_ratings_array[user_id, action]

    # 更新Q值
    Q[action] = Q[action] + 0.2 * (reward - Q[action])


# 转换为元组列表
top_N_movies = torch.Tensor(top_n_indices1(Q, conf['recommend_k']))
print(top_N_movies)
hit_num = count_occurrences(test_edge_index[1], top_N_movies)
print(hit_num)
ep_cache_efficiency = hit_num/test_edge_index.shape[1]
print(ep_cache_efficiency)
