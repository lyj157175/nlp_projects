import pickle
import torch
# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
# f = open('THUCNews/data/dataset.pkl','rb')
# data = pickle.load(f)
# print(data['test'])
# ([1, 132, 664, 19, 18, 833, 26, 21, 401, 166, 556, 228, 2211, 1568, 1860, 
# 2785, 171, 650, 1625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 19, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

import torch
a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
print(torch.max(a, 1)[1])


