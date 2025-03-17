import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# 定义SAE网络
class SAE(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_input)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        r = self.fc2(h)
        return r, h


# 定义混合过滤系统
class Hybrid(nn.Module):
    def __init__(self, n_users, n_movies, n_hidden):
        super(Hybrid, self).__init__()
        self.sae = SAE(n_movies, n_hidden)
        self.fc = nn.Linear(n_hidden + n_users, n_movies)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        v, h = self.sae(x)
        v = self.dropout(v)
        x = torch.cat([v, x], 1)
        x = self.fc(x)
        return x


# 定义训练函数
def train(model, optimizer, criterion, train_data, epochs=10):
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        inputs = train_data
        target = inputs.clone()
        target.require_grad = False
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch {epoch}, loss: {running_loss / len(train_data)}")


# 定义预测函数
def predict(model, test_data):
    with torch.no_grad():
        predictions = np.array([])
        for data in test_data:
            inputs, _ = data
            outputs = model(inputs)
            predictions = np.append(predictions, outputs.cpu().numpy())
    return predictions



ratings = pd.read_csv('../database/movielens_100K/ratings.csv')
train_data_2 = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
# 加载数据集

# 将评分矩阵分为训练集和测试集
train_data = torch.tensor(train_data_2.values, dtype=torch.float32)
test_data = torch.tensor(train_data_2.values, dtype=torch.float32)

# 定义SAE深度自编码器
n_hidden = 2
sae = SAE(ratings.shape[1], n_hidden)

# 训练SAE
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
train(sae, optimizer, criterion, train_data, epochs=10)

# 得到SAE的隐含特征表示
with torch.no_grad():
    train_features = sae(torch.Tensor(train_data[:, :-1])).numpy()
    test_features = sae(torch.Tensor(test_data[:, :-1])).numpy()

# 计算物品相似度和用户相似度
item_similarities = np.dot(train_features.T, train_features)
user_similarities = np.dot(train_features, train_features.T)

# 定义混合过滤模型
n_users = ratings.shape[0]
n_movies = ratings.shape[1]
hybrid = Hybrid(n_users, n_movies, n_hidden)
train_ratings = torch.Tensor(train_data[:, :-1])
train_targets = torch.Tensor(train_data[:, -1])
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(hybrid.parameters(), lr=0.01, weight_decay=0.5)
train(hybrid, optimizer, criterion, zip(train_ratings, train_targets), epochs=10)

# 预测测试集评分并计算均方根误差（RMSE）
test_ratings = torch.Tensor(test_data[:, :-1])
test_targets = torch.Tensor(test_data[:, -1])
predictions = predict(hybrid, zip(test_ratings, test_targets))
rmse = np.sqrt(np.mean((predictions - test_targets.numpy()) ** 2))
print(f"Test RMSE: {rmse}")
