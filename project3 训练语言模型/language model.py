import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000


# torchtext预处理流程：
# 定义Field：声明如何处理数据
# 定义Dataset：得到数据集，此时数据集里每一个样本是一个 经过 Field声明的预处理 预处理后的 wordlist
# 建立vocab：在这一步建立词汇表，词向量(word embeddings)
# 构造迭代器：构造迭代器，用来分批次训练模型
TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", 
                    train="text8.train.txt", 
                    validation="text8.train.txt", 
                    test="text8.train.txt", 
                    text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))  
# TEXT.vacob.itos[:10]
# TEXT.vacob.stoi['the']
VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
                        (train, val, test), 
                        batch_size=BATCH_SIZE, 
                        device=-1, 
                        bptt_len=32, 
                        repeat=False, 
                        shuffle=True)  #bptt_len

it = iter(train_iter)
batch = next(it)
# batch
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))

# 定义模型
import torch
import torch.nn as nn

class RNNModel(nn.Module): 
    """ 一个简单的循环神经网络"""
    def __init__(self, vocab_size, embed_size, hidden_size, nlayers):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()  
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        
        
    def forward(self, text, hidden):
        # text: seq_len * batct_size 
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.embed(text)  # seq_len * batch_size * embed_size 
        output, hidden = self.rnn(emb, hidden)  
        # output: seq_len * batch_size * hidden_size    
        # hidden: (num_layers * num_directions, batch_size, hidden_size)  n为lstm的层数
        
        out_vocab = self.linear(output.view(-1, output.size(2)))   #（seq_len * batch_size） * hidden_size
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1)) #转为原来的形状
        return out_vocab, hidden

    def init_hidden(self, batch_size, requires_grad=True):   #初始化， 返回一个cell state，一个hidden state 
        weight = next(self.parameters()) 
#         return (weight.new_zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=True),
        return weight.new_zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=True)


#初始化模型
model = RNNModel(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 1)
if USE_CUDA:
    model = model.cuda()


def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size()) 
            
    loss = total_loss / total_count
    model.train()
    return loss

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()  #tensor的起点
    else:
        return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)   #降低lr

import copy
GRAD_CLIP = 1.   
NUM_EPOCHS = 1

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()  
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        data, target = data.cuda(), target.cuda()
        
        #不然会爆内存
        hidden = repackage_hidden(hidden)
        
        optimizer.zero_grad()      
        output, hidden = model(data, hidden)  #backpropgate throgh all iter 
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   
        optimizer.step()
    
        if i % 100 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item())
    
        if i % 100 == 0:
            val_loss = evaluate(model, val_iter)
            
            if len(val_losses) == 0 or val_loss < min(val_losses):
                print("best model, val loss: ", val_loss)
                torch.save(model.state_dict(), "lm-best.th")      #保存模型
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_losses.append(val_loss)


best_model = RNNModel(VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 1)
if USE_CUDA:
    best_model = best_model.cuda()
best_model.load_state_dict(torch.load("lm-best.th"))

# 使用最好的模型在valid数据上计算perplexity
val_loss = evaluate(best_model, val_iter)
print("perplexity: ", np.exp(val_loss))

# 使用最好的模型在测试数据上计算perplexity
test_loss = evaluate(best_model, test_iter)
print("perplexity: ", np.exp(test_loss))


#使用训练好的模型生成一些句子
hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))