import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = pd.read_csv("100K_10clients.csv", header=0, index_col=False)
labels = f['cache_size'][0:2].values
a = f['Oracle'][0:2].values
b = f['GFPC'][0:2].values
c = f['FPC'][0:2].values
d = f['Greedy'][0:2].values
e = f['tp sampling'][0:2].values
ff = f['random']
ways = ['Oracle', 'GFPC', 'FPC', 'Greedy', 'Thompson Sampling', 'random']

x = np.arange(len(labels))  # 标签位置
width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

fig, ax = plt.subplots()
rects1 = ax.bar(x - width * 2, a, width, label=ways[0])
rects2 = ax.bar(x - width + 0.01, b, width, label=ways[1])
rects3 = ax.bar(x + 0.02, c, width, label=ways[2])
rects4 = ax.bar(x + width + 0.03, d, width, label=ways[3])
rects5 = ax.bar(x + width * 2 + 0.04, e, width, label=ways[4])

# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('Cache Efficiency', fontsize=12)
ax.set_xlabel('Cache size', fontsize=12)
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# def autolabel(rects):
#     """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3点垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)

fig.tight_layout()
plt.show()
