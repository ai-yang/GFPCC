import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from operator import itemgetter  # 用来使用sorted() function
from data_pre import pre_datasets
from utils import *
from tqdm import tqdm

# 载入配置文件和数据集
with open("../conf.json", 'r') as f:
    conf = json.load(f)

ratings = pd.read_csv('../database/movielens_100K/ratings.csv')
num_users, _, train_edge_index, test_edge_index = pre_datasets(conf)
# 模型超参数
epochs = 30
it_per_evl = 10
lr = 0.01
weight_decay = 0.5

# # 分割数据集和训练集
# num_interactions = ratings.shape[0]  # 计算出可以用于训练的数据长度
# all_indices = [i for i in range(num_interactions)]  # 所有索引
# train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=conf['random_seed'])
# train_data = ratings.iloc[train_indices]
# test_data = ratings.iloc[test_indices]
# # 将数据转换成 用户-电影 二维表 纵坐标为用户，横坐标为电影
train_data_2 = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
test_data_2 = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
# 将用户未评分的电影分值设置为0
training_set = train_data_2.fillna(0)
test_set = test_data_2.fillna(0)

# Converting the data into Torch tensors
training_set = torch.tensor(training_set.values, dtype=torch.float32)
test_set = torch.tensor(test_set.values, dtype=torch.float32)
num_movies = training_set.shape[1]


class SAE(nn.Module):
    def __init__(self, ):
        # use super() function to optimize the SAE
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(num_movies, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 40)
        self.fc4 = nn.Linear(40, num_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # need to modify or update x after each encoding and decoding
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # start decoding it
        x = self.activation(self.fc3(x))
        # the final part of the decoding doesn't need to apply the activation function, we directly use full coneection fc4 function
        x = self.fc4(x)
        # x is our vector of predicted ratings
        return x

    def predict(self, x):  # x: visible nodes
        x = self.forward(x)
        return x


# 定义prediction function
def Prediction(user_id, nb_recommend):
    user_input = Variable(test_set[user_id]).unsqueeze(0)
    predict_output = sae.predict(user_input)
    predict_output = predict_output.data.numpy()
    # predicted_result = np.vstack([user_input, predict_output])
    recommend = top_n_indices(predict_output, nb_recommend)
    return recommend
    # train_movie_id = np.array([i for i in range(1, num_movies + 1)])
    # # create a temporary index for movies since we are going to delete some movies that the user had seen, 创建一个类似id的index，排序用
    # recommend = np.array(predicted_result)
    # recommend = np.row_stack((recommend, trian_movie_id))  # insert that index into the result array, 把index插入结果
    # recommend = recommend.T  # transpose row and col 数组的行列倒置
    # recommend = recommend.tolist()  # transfer into list for further process转化为list以便处理
    #
    # movie_not_seen = []  # delete the rows containing the movies that the user had seen 删除users看过的电影
    # for i in range(len(recommend)):
    #     # if recommend[i][0] == 0.0:
    #     movie_not_seen.append(recommend[i])
    #
    # movie_not_seen = sorted(movie_not_seen, key=itemgetter(1), reverse=True)  # sort the movies by mark 按照预测的分数降序排序
    #
    # recommend_movie = []  # create list for recommended movies with the index we created 推荐的top20
    # for i in range(0, nb_recommend):
    #     recommend_movie.append(movie_not_seen[i][2])

    # recommend_index = []  # get the real index in the original file of 'movies.dat' by using the temporary index这20部电影在原movies文件里面真正的index
    # for i in range(len(recommend_movie)):
    #     recommend_index.append(movies[(movies.iloc[:, 0] == recommend_movie[i])].index.tolist())
    #
    # recommend_movie_name = []  # get a list of movie names using the real index将对应的index输入并导出movie names
    # for i in range(len(recommend_index)):
    #     np_movie = movies.iloc[recommend_index[i], 1].values  # transefer to np.array
    #     list_movie = np_movie.tolist()  # transfer to list
    #     recommend_movie_name.append(list_movie)
    #
    # print('Highly Recommended Movies for You:\n')
    # for i in range(len(recommend_movie_name)):
    #     print(str(recommend_movie_name[i]))


# load the trained model
# sae = torch.load('AutoEncoder.pkl')


sae = SAE()
# criterion that will need for the training, for the loss function MSE
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=lr, weight_decay=weight_decay)

# in each epoch, we will loop over all our observation that is users, than to loop each epoch, so there are 2 loop
for epoch in tqdm(range(1, epochs + 1)):
    # the loss variable = 0 since before start the training, loss = 0, loss will increase as it finds some errors
    train_loss = 0
    # s: counter. we need to normalize the train loss, so need to divide the train loss by this counter   s=0. means the type of counter is float
    s = 0.
    for id_user in range(num_users):
        # since num_users and training_set start from 0, so don't need to modify the range to (1, num_users1)
        # training_set[id_user] is a vector, but a network in pytorch or even in keras can not accept a single vector of one dimension, what they rather accept is a batch of input vectors,
        # so when we apply different functions of the network for example forward function, the function will not take single vector of 1D as input, so we need to add an additional dimension like a fake dimension which will correpondent to the batch,
        # so using Variable().unsqueeze(0), we will put this dimension into the first dimension, so this value will be 0, we now create a batch having a single input vector, but at the same time, a batch can have several input vectors (batch learning)
        input = Variable(training_set[id_user]).unsqueeze(0)
        # now take care of the target, we have separate variables between input vector and target, so basically the target is the same as the input vector, since we are going to modify the input, which will be a clone of the input vector, using clone()
        target = input.clone()
        # to optimize the memory
        # target.data will take all the values of target which is input vectors, that will be all the rating of this user at the loop
        # so this we will make sure the observation contains at least 1 rating, which means take the users who have rated at least 1 movie
        if torch.sum(target.data > 0) > 0:
            # first get our vector of predicted ratings that is output
            output = sae(input)
            # second, again, to optimize the memory and the computation
            # when we apply stochastic gradient descent, we want to make sure the gradient is computed only with respect to the input and not the target, to do this we will use require_grad = False, whcih will reduce the computation and save up memory, that will make sure will not compute the gradient of the targets
            target.require_grad = False
            # also the optimization, we only want to include the computations the non-zero values, we don't want to use the movie that users didn't rate, we have already set the non-zero input target, but we haven't set the non-zero output, so set it
            # equal to 0 means that it will not add up to the error that will affect the weight updating
            output[target == 0] = 0
            # compute the loss
            loss = criterion(output, target)
            # compute the mean corrector, number of movies/the number of movies predicted possitive, + 1e-10 means we want to make sure the 分母是非空的，所以加上1^-10，一个很小的数在分母上
            # mean_corrector represent the average of the error by only considering the movies that were rated
            mean_corrector = num_movies / float(torch.sum(target.data > 0) + 1e-10)
            # backward method for the loss, which will tell in which direction we need to update the different weights
            loss.backward()
            # update the train_loss and s
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            # differece between backward and optimizer.step, backward decides the dirction to which  the weight will be updated, optimizer.step decides intensity of the updates that is the amount by which the weights will be updated
            optimizer.step()

    if epoch % it_per_evl == 0:
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

movie_for_you = []
# recommendation for target user's id
for user_id in range(1, num_users):
    # the number of movies recommended for the user
    movie_for_you.append(Prediction(user_id=user_id, nb_recommend=conf['recommend_m']))

recommend_list = torch.tensor(movie_for_you)
top_k = top_k_frequent_numbers(recommend_list, conf['recommend_k'])
top_k = top_k + 1
sae_cache_efficiency = compute_cache_efficiency(test_edge_index, test_edge_index.shape[1], top_k)
print(sae_cache_efficiency)
