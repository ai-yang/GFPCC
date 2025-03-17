import numpy as np
import pandas as pd

# 加载movielens数据集
ratings_data = pd.read_csv('../database/movielens_100K/ratings.csv')

# 创建电影字典
movies_dict = {}
for index, row in pd.read_csv('../database/movielens_100K/ratings.csv').iterrows():
    movies_dict[row['movieId']] = row['title']

# 计算每个电影的平均得分
movie_ratings = pd.DataFrame(ratings_data.groupby('movieId')['rating'].mean())
movie_ratings['num_ratings'] = pd.DataFrame(ratings_data.groupby('movieId')['rating'].count())

# 计算用户的评分平均值
user_ratings = pd.DataFrame(ratings_data.groupby('userId')['rating'].mean())

# 将评分数据转换为评分矩阵
ratings_matrix = pd.pivot_table(ratings_data, values='rating', index='userId', columns='movieId')

# 计算电影和用户的相似度矩阵
movie_similarity = np.corrcoef(movie_ratings.fillna(0).T)
user_similarity = np.corrcoef(ratings_matrix.fillna(0))


# 混合过滤函数
def hybrid_filter(userId=9, N=50):
    # 计算每个电影的得分
    movie_ratings['score'] = movie_ratings['rating'] * movie_ratings['num_ratings'] / (movie_ratings['num_ratings'] + 5)

    # 计算用户的相似度向量
    user_sim_vector = user_similarity[userId - 1]
    user_sim_vector[userId - 1] = -1

    # 计算每个电影的加权得分
    movie_ratings['hybrid_score'] = np.dot(movie_similarity, user_sim_vector) * movie_ratings['score']

    # 按加权得分对电影进行排序，并返回前N个电影的ID
    recommendations = movie_ratings.sort_values('hybrid_score', ascending=False)[:N].index.tolist()

    # 输出推荐结果
    print(f'推荐给用户{userId}的电影为：')
    for movie_id in recommendations:
        print(movies_dict[movie_id])


hybrid_filter(userId=9, N=50)
