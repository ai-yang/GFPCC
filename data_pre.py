import dataloader
from sklearn.model_selection import train_test_split


def pre_datasets(conf):
    # 判断输入数据集名称
    data_name = conf['data_name']
    movie_path = ""
    rating_path = ""
    if data_name == '100K':
        movie_path = "../database/movielens_100K/ratings.csv"
        rating_path = "../database/movielens_100K/ratings.csv"
    elif data_name == '1M':
        movie_path = "../database/movielens_1M/movies.csv"
        rating_path = "../database/movielens_1M/ratings.csv"
    elif data_name == 'DVD':
        movie_path = "../database/CiaoDVD/ratings.csv"
        rating_path = "../database/CiaoDVD/ratings.csv"
    # 加载数据集
    user_mapping = dataloader.load_node_csv(rating_path, index_col='userId')  # 处理出所有不同的用户
    movie_mapping = dataloader.load_node_csv(movie_path, index_col='movieId')  # 处理出所有不同的电影
    edge_index = dataloader.load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        link_index_col='rating',
        rating_threshold=4,
    )  # 根据唯一的点和唯一的边处理出边集

    # 用来提取出所有用户数量和电影数量存入num_users和num_movies
    num_users, num_movies = len(user_mapping), len(movie_mapping)
    num_interactions = edge_index.shape[1]  # 计算出可以用于训练的数据长度
    all_indices = [i for i in range(num_interactions)]  # 所有索引
    '''
    这里面有三个数据集，分别如下：
    这里面保存的是需要读取的数据下标
    训练集 train_indices
    测试集：test_indices
    验证集: val_indices
    '''
    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=conf['random_seed'])  # 将数据集划分成80:20的训练集:测试集
    print(edge_index.shape)
    train_edge_index = edge_index[:, train_indices]
    test_edge_index = edge_index[:, test_indices]
    # print(test_edge_index.shape)
    return num_users, num_movies, train_edge_index, test_edge_index
