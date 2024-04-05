# import numpy as np
#
# # 构造数据
# import time
# d = 50                           # dimension
# nb = 1000                     # database size
# # nq = 1000000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')  #xb=1000*50
# xb[:, 0] += np.arange(nb) / 10.
# # xq = np.random.random((nq, d)).astype('float32')
# # xq[:, 0] += np.arange(nq) / 1000.
#
# print(xb[:1])
#
# # 写入文件中
# # file = open('data.txt', 'w')
# np.savetxt('./data.txt', xb)

import numpy as np
import faiss

# 读取文件形成numpy矩阵
data = []
with open('data.txt', 'rb') as f:
    for line in f:
        temp = line.split()
        data.append(temp)
print("data[0]:\n")
print(data[0])

# 训练与需要计算的数据
dataArray = np.array(data).astype('float32')
print("dataArray[0]:\n")
print(dataArray[0])

print("dataArray.shape[1]:\n")
print(dataArray.shape[1])
# 获取数据的维度
d = dataArray.shape[1]

# IndexFlatL2索引方式
# # 为向量集构建IndexFlatL2索引，它是最简单的索引类型，只执行强力L2距离搜索
index = faiss.IndexFlatL2(d)   # build the index
index.add(dataArray)                  # add vectors to the index
#
# # we want to see 4 nearest neighbors
k = 11
# # search
D, I = index.search(dataArray, k)
#
# # neighbors of the 5 first queries
print(I[:5])