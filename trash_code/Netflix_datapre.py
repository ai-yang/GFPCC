from datetime import datetime
import pandas as pd
import numpy as np

"""
用来预处理Netflix数据集
转化为moivelens格式
但是因为Netflix数据集太大，训练开销太大
于是寻找另一个数据集代替
该函数被弃用
"""

df1 = pd.read_csv('Netflix/combined_data_1.txt', header=None, names=['user_id', 'rating', 'timestamp'],
                  usecols=[0, 1, 2])  # 读入combined_data_1
df2 = pd.read_csv('Netflix/combined_data_2.txt', header=None, names=['user_id', 'rating', 'timestamp'],
                  usecols=[0, 1, 2])  # 读入combined_data_2
df3 = pd.read_csv('Netflix/combined_data_3.txt', header=None, names=['user_id', 'rating', 'timestamp'],
                  usecols=[0, 1, 2])  # 读入combined_data_3
df4 = pd.read_csv('Netflix/combined_data_4.txt', header=None, names=['user_id', 'rating', 'timestamp'],
                  usecols=[0, 1, 2])  # 读入combined_data_4

df1['rating'] = df1['rating'].astype(float)
# df2['rating'] = df2['rating'].astype(float)
# df3['rating'] = df3['rating'].astype(float)
# df4['rating'] = df4['rating'].astype(float)

print('Dataset 1 shape: {}'.format(df1.shape))
# print('Dataset 2 shape: {}'.format(df2.shape))
# print('Dataset 3 shape: {}'.format(df3.shape))
# print('Dataset 4 shape: {}'.format(df4.shape))
print('-Dataset examples-')
print(df1.iloc[::5000000, :])

df = df1
# df.append(df2)
# df.append(df3)
# df.append(df4)
df.index = np.arange(0, len(df))
print('Full dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::5000000, :])
df_nan = pd.DataFrame(pd.isnull(df.rating))
df_nan = df_nan[df_nan['rating'] == True]
df_nan = df_nan.reset_index()

item_np = []
item_id = 1

for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
    # 使用numpy
    temp = np.full((1, i - j - 1), item_id)
    item_np = np.append(item_np, temp)
    item_id += 1

# 考虑最后一条记录和其长度
# 使用numpy
last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), item_id)
item_np = np.append(item_np, last_record)
print('Item numpy: {}'.format(item_np))
print('Length: {}'.format(len(item_np)))


def time2stamp(cmnttime):  # 时间转时间戳函数
    cmnttime = datetime.strptime(cmnttime, '%Y-%m-%d')
    stamp = int(datetime.timestamp(cmnttime))
    return stamp


df = df[pd.notnull(df['rating'])].copy()
df['item_id'] = item_np.astype(int)
df['user_id'] = df['user_id'].astype(int)
df = df.loc[:, ['user_id', 'item_id', 'rating', 'timestamp']]  # 交换两列位置
df['timestamp'] = df['timestamp'].astype(str).apply(time2stamp)  # 时间转成时间戳
print('-Dataset examples-')
print(df.iloc[::5000000, :])
# df.sort_values(by=["user_id", "timestamp"], ascending=[True, True]) # 先按用户id排序，然后按时间戳排序
df.to_csv('ratings1.dat', sep=',', index=0, header=0)
