import random

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 获取1000个随机散点图数据
data = np.random.rand(1000, 2)
# print(data)

# K-Means聚类
km = KMeans(n_clusters=5) #聚5类
result = km.fit(data)
labels = result.labels_  # 获取聚类标签
print(labels)
centroids = result.cluster_centers_  # 获取聚类中心
print(centroids)
# inertia = result.inertia_  # 获取聚类准则的总和
# print(inertia)
# color = ['red', 'black', 'green', 'blue', 'yellow']
# X = data.T[0]
# Y = data.T[1]
# # plt.scatter(X,Y)
# label_iter = 0  # 计数器
# for label in labels:
#     plt.scatter(X[label_iter], Y[label_iter], color=color[label])
#     label_iter = label_iter + 1
# plt.show()

