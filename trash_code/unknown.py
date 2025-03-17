import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import json
from sklearn.model_selection import train_test_split
from data_pre import pre_datasets
from utils import *


#  加载配置文件和数据集
with open("../conf.json", 'r') as f:
    conf = json.load(f)

num_users, num_movies, train_edge_index, test_edge_index = pre_datasets(conf)
ratings = pd.read_csv('../database/movielens_1M/ratings.csv')
# ratings = ratings[ratings['rating'] >= 4]  # 只取评分大于4的

num_interactions = ratings.shape[0]  # 计算出可以用于训练的数据长度
all_indices = [i for i in range(num_interactions)]  # 所有索引
train_indices, test_indices = train_test_split(all_indices, test_size=0.8, random_state=conf['random_seed'])
train_data = ratings.iloc[train_indices]

# 将数据转换成 用户-电影 二维表 纵坐标为用户，横坐标为电影
pivoted_data = pd.pivot_table(train_data, values='rating', index='userId', columns='movieId')

# 将用户未评分的电影分值设置为0
pivoted_data = pivoted_data.fillna(0)

# 一些超参数
alpha = 0.5  # constant
K = pivoted_data.shape[1]  # the number of arm（电影数量）
d = pivoted_data.shape[0]  # the number of feature（用户数量，当作电影特征）
t = 3000  # epoch
N = 100  # change the mu after N trials

# 初始化操作
B = np.identity(d)
mu = np.zeros((d, 1))
f = np.zeros((d, 1))
graph = pd.DataFrame(None, columns=['time', 'regret'])
cnt = 0


movies_count = dict.fromkeys(pivoted_data.columns, 0)
mu_true = np.random.rand(d, 1)
# 重复t次
for i in tqdm.tqdm(range(t)):
    cnt = cnt + 1
    # change the mu
    if cnt % N == 0:
        mu_true = np.random.rand(d, 1)

    # sample from distribution
    mu_est = random.multivariate_normal(mu.reshape(1, d).tolist()[0], np.linalg.inv(B)).reshape(d, 1)

    # select the argmax arm
    bb = np.random.rand(K, d)
    arg = np.dot(bb, mu_est)
    a_t = bb[arg.argmax(), :]
    reward = np.dot(a_t, mu_true)

    # update
    B = B + np.dot(a_t.reshape(d, 1), a_t.reshape(1, d))
    f = f + (a_t.reshape(d, 1) * reward)
    mu = np.dot(np.linalg.inv(B), f)

    # record movie count
    movie_id = pivoted_data.columns[arg.argmax()]
    movies_count[movie_id] += 1

    # regret
    arg_r = np.dot(bb, mu_true)
    a_t_r = bb[arg_r.argmax(), :]
    reward_true = np.dot(a_t_r, mu_true)

    regret = reward_true - reward
    df1 = pd.DataFrame({'time': [i], 'regret': [regret[0]]})
    graph = pd.concat([graph, df1], ignore_index=True)

# recommend top recommend_k movies
top_n_movies = sorted(movies_count.items(), key=lambda x: x[1], reverse=True)[:conf['recommend_k']]
top_k = torch.Tensor([t[0] for t in top_n_movies])  # top_k个电影
tf_cache_efficiency = compute_cache_efficiency(test_edge_index, test_edge_index.shape[1], top_k)
print(top_k)
print(tf_cache_efficiency)

# drawing the graph
graph['accumulated_regret'] = graph['regret'].cumsum()

x = graph['time']
y = graph['accumulated_regret']

plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('accumulated regret')
plt.title('Thompson Sampling')
plt.show()
