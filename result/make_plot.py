import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
data = pd.read_csv('1M_10clients.csv')

# 提取组名和对应的值
group_names = data['cache_size'].tolist()
values = data.iloc[:, 1:].values
label_name = ['Oracle', 'GFPC', 'FPCC', 'm-$\epsilon$-Greedy', 'Thompson Sampling', 'Random']
# 设置柱状图的宽度
bar_width = 0.1

# 创建x轴位置
index = np.arange(len(group_names))

# 绘制柱状图
for i in range(values.shape[1]):
    plt.bar(index + i * bar_width, values[:, i], bar_width, label=label_name[i], ec='k', ls='-', lw=0.7)

# 添加x轴标签、标题和图例
plt.xlabel('Cache Size')
plt.ylabel('Cache Efficiency(%)')
plt.title('')
plt.xticks(index + 2 * bar_width, group_names)
plt.legend()
plt.savefig("output.png", dpi=300)
# 展示图形
plt.show()
