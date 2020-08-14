import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



dtype = torch.FloatTensor
sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = ' '.join(sentences).split()
word_list = list(set(word_list))
word2idx = {w:i for i, w in enumerate(word_list)}
idx2word = {i:w for i, w in enumerate(word_list)}
num_class = len(word_list)

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for seq in sentences:
        input = [word2idx[i] for i in seq.split(' ')[:-1]]
        target = word2idx[seq.split(' ')[-1]]
        input_batch.append(input)
        target_batch.append(target)
    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # nn.Parameter()来转换一个固定的权重数值，使的其可以跟着网络训练一直调优下去，学习到一个最适合的权重值。
        self.C = nn.Embedding(num_class, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, num_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, num_class).type(dtype))
        self.b = nn.Parameter(torch.randn(num_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output
    
model = NNLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

for epoch in range(5000):
    optimizer.zero_grad()
    out = model(input_batch)
    loss = criterion(out, target_batch)
    if epoch % 1000 == 0:
        print('epoch: {}  loss: {}'.format(epoch, loss))
    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])