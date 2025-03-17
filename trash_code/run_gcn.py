from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import dataloader
from evaluate import evaluation
from models.LightGCN_model import LightGCN, bpr_loss

"""
这一个文件用来集中训练LightGCN模型
去除了联邦框架，将所有数据集直接喂入模型中训练
用来验证模型的可行性
"""

# 读取数据集
movie_path = 'database/CiaoDVD/ratings.csv'
rating_path = 'database/CiaoDVD/ratings.csv'
user_mapping = dataloader.load_node_csv(rating_path, index_col='userId')
movie_mapping = dataloader.load_node_csv(movie_path, index_col='movieId')
edge_index = dataloader.load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    link_index_col='rating',
    rating_threshold=4,
)
device = torch.device('cuda')  # 判断是否能够在GPU上训练
num_users, num_movies = len(user_mapping), len(movie_mapping)
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]  # 所有索引

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, random_state=1)  # 将数据集划分成80:10的训练集:测试集
val_indices, test_indices = train_test_split(
    test_indices, test_size=0.5, random_state=1)  # 将测试集划分成10:10的验证集:测试集,最后的比例就是80:10:10

train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]


# 超参数在这里调整
ITERATIONS = 5200
BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
K = 20
LAMBDA = 1e-6

# SparseTensor 是稀疏化存储矩阵
model = LightGCN(device, num_users, num_movies)
model = model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
edge_index = edge_index.to(device)

train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                       sparse_sizes=(num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1],
                                     sparse_sizes=(num_users + num_movies, num_users + num_movies))
train_edge_index = train_edge_index.to(device)
train_sparse_edge_index = train_sparse_edge_index.to(device)
val_edge_index = val_edge_index.to(device)
val_sparse_edge_index = val_sparse_edge_index.to(device)
train_losses = []
val_losses = []

for iter in range(ITERATIONS):
    # forward propagation
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        train_sparse_edge_index)

    # mini batching
    user_indices, pos_item_indices, neg_item_indices = dataloader.sample_mini_batch(BATCH_SIZE, train_edge_index, num_movies)
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    # loss computation
    train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                          pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if iter % ITERS_PER_EVAL == 0:
        model.eval()
        val_loss, recall, precision, ndcg, top_k = evaluation(
            model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA, num_movies)  # 推荐前K个内容
        print(
            f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
        train_losses.append(train_loss.item())
        val_losses.append(val_loss)
        model.train()

    if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
        scheduler.step()

iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.show()
