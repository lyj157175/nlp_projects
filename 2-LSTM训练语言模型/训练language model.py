#!/usr/bin/env python
# coding: utf-8

'''LSTM训练语言模型, 用来预测下一个词'''


import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors
import numpy as np
import random



class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, nlayers, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)  # (1000, 50002)
        self.init_weights()

        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):   
        # input: seq_len * batch = [50, 32]，可以在LSTM定义batch_first = True
        # hidden = (nlayers * b * hidden_size)
        # hidden是个元组，输入有两个参数，一个是刚开始的隐藏层h的维度，一个是刚开始的用于记忆的c的维度，
        embed = self.drop(self.embedding(input))  #seq_len * b * embedding_size
        output, hidden = self.rnn(embed, hidden) 
        # output.shape = seq_len * b * hidden_size 
        # hidden元组 = (h层：nlayers * 32 * hidden_size, c层：nlayers * 32 * hidden_size)
        output = self.drop(output)
        linear = self.linear(output.view(-1, output.size(2)))  
        # [seq_len*batch, hidden_size] -> [seq_len*batch, vocab_size]
        
        return linear.view(output.size(0), output.size(1), linear.size(1)), hidden
               # 输出恢复维度 :[seq_len, b, vocab_size]
               # hidden = (h层维度：nlayers * b * hidden_size, c层维度：nlayers * b * hidden_size)

            
    def init_hidden(self, batch_size, requires_grad=True):
        # 最初隐藏层参数的初始化
        weight = next(self.parameters())
        # weight = torch.Size([50002, 500])是所有参数的第一个参数
        # 所有参数self.parameters()，是个生成器

        return (weight.new_zeros((self.nlayers, batch_size, self.hidden_size), 
                                    requires_grad=requires_grad),
                weight.new_zeros((self.nlayers, batch_size, self.hidden_size), 
                                    requires_grad=requires_grad))
                # return = (2 * 32 * 1000, 2 * 32 * 1000)
                # 这里不明白为什么需要weight.new_zeros，我估计是想整个计算图能链接起来
                # 这里特别注意hidden的输入不是model的参数，不参与更新，就跟输入数据x一样                 
       


# - 我们首先定义评估模型的代码。
# - 模型的评估和模型的训练逻辑基本相同，唯一的区别是我们只需要forward pass，不需要backward pass

def evaluate(model, dev_iter):
    model.eval() # 预测模式
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
# 这里不管是训练模式还是预测模式，h层的输入都是初始化为0，hidden的输入不是model的参数
# 这里model里的model.parameters()已经是训练过的参数。
        for i, batch in enumerate(dev_iter):
            data, target = batch.text, batch.target
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            hidden = repackage_hidden(hidden)   # 截断计算图
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())  # 得到该batch的单词总数，*data.szie进行拆包，得到seq_len * batch = 50*32
            total_loss += loss.item()*np.multiply(*data.size())  # 一次batch总的损失

    loss = total_loss / total_count # 整个验证集总的损失除以总的单词数
    model.train() 
    return loss


# #### 定义一个function，把一个hidden state和计算图之前的历史分离。


# 将当前隐藏层hidden与之前进行截断
def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor): 
        # 这个是GRU的截断，因为只有一个隐藏层
        return hidden.detach() # 截断计算图，h是全的计算图的开始，只是保留了h的值
    else: # 这个是LSTM的截断，有两个隐藏层，格式是元组
        return tuple(repackage_hidden(v) for v in hidden)



if __name__ == '__main__':
    use_cuda= torch.cuda.is_available()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if use_cuda:
        torch.cuda.manual_seed(1234)

    # 超参数
    BATCH_SIZE = 32 
    EMBEDDING_SIZE = 500  
    MAX_VOCAB_SIZE = 50000
    hidden_size = 1000 
    learning_rate = 0.001
    GRAD_CLIP = 1.
    NUM_EPOCHS = 2


    TEXT = torchtext.data.Field(lower=True)   
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
                        path=".", 
                        train="text8.train.txt", 
                        validation="text8.dev.txt", 
                        test="text8.test.txt", 
                        text_field=TEXT)

    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
    print("vocabulary size: {}".format(len(TEXT.vocab)))

    VOCAB_SIZE = len(TEXT.vocab)  # 50002
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
                                (train, val, test), 
                                batch_size=BATCH_SIZE, 
                                device=-1, 
                                bptt_len=50, 
                                repeat=False, 
                                shuffle=True)

    model = LanguageModel(VOCAB_SIZE, EMBEDDING_SIZE, hidden_size, 2, dropout=0.5)
    if use_cuda:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    val_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train() 
        hidden = model.init_hidden(BATCH_SIZE)  # 隐藏层初始化
    
        for i, batch in enumerate(train_iter):
            data, target = batch.text, batch.target
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
        # 语言模型每个batch的隐藏层的输出值是要继续作为下一个batch的隐藏层的输入的,
        # 因为batch数量很多，如果一直往后传，会造成整个计算图很庞大，反向传播会内存崩溃。
        # 所有每次一个batch的计算图迭代完成后，需要把计算图截断，只保留隐藏层的输出值。
        # 不过只有语言模型才这么干，其他比如翻译模型不需要这么做。
        # repackage_hidden自定义函数用来截断计算图的。
            optimizer.zero_grad()   
            output, hidden = model(data, hidden)  # output = (50,32,50002)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))   #loss_fn((num, class), num)
            loss.backward()
            # 防止梯度爆炸，设定阈值，当梯度大于阈值时，更新的梯度为阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) 
            optimizer.step()
            
            if i % 1000 == 0:
                print("epoch", epoch, "iter", i, "loss", loss.item())
        
            if i % 10000 == 0:
                val_loss = evaluate(model, val_iter)  #在val_iter上进行验证 
                
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print("best model, val_loss: ", val_loss)
                    torch.save(model.state_dict(), "LanguageModel.th")
                else:
                    scheduler.step()  # 自动调整学习率
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    # 学习率调整后需要更新optimizer，下次训练就用更新后的
                val_losses.append(val_loss) # 保存每10000次迭代后的验证集损失损失

    # 模型评估
    # 加载保存好的模型参数
    best_model = LanguageModel(VOCAB_SIZE, EMBEDDING_SIZE, hidden_size, 2, dropout=0.5)
    if use_cuda:
        best_model = best_model.cuda()
    best_model.load_state_dict(torch.load("LanguageModel.th"))

    # 使用最好的模型在valid和test数据上计算perplexity
    val_loss = evaluate(best_model, val_iter)
    print("perplexity: ", np.exp(val_loss))   # 语言模型指标：困惑度，越小越好
    test_loss = evaluate(best_model, test_iter)
    print("perplexity: ", np.exp(test_loss))


    # 使用训练好的模型生成一些句子
    hidden = best_model.init_hidden(1) # batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
    # (1,1)表示（seq_len, batch_size）输出格式是1行1列的2维tensor，VOCAB_SIZE表示随机取的值小于VOCAB_SIZE=50002
    # 我们input相当于取的是一个单词
    words = []
    for i in range(100):
        output, hidden = best_model(input, hidden)
        # output.shape = 1 * 1 * 50002
        # hidden = (2 * 1 * 1000, 2 * 1 * 1000)
        word_weights = output.squeeze().exp().cpu()
        # .exp()的两个作用：一是把概率更大的变得更大，二是把负数经过e后变成正数，下面.multinomial参数需要正数
        word_idx = torch.multinomial(word_weights, 1)[0]   #得到的是概率最大的单词的索引,[0]相当于拿到数      
        # 按照word_weights里面的概率随机的取值，概率大的取到的机会大。
        # torch.multinomial看这个博客理解：https://blog.csdn.net/monchin/article/details/79787621
        # 这里如果选择概率最大的，会每次生成重复的句子。
        input.fill_(word_idx) # 预测的单词index是word_idx，然后把word_idx作为下一个循环预测的input输入
        word = TEXT.vocab.itos[word_idx] # 根据word_idx取出对应的单词
        words.append(word) 
    print(" ".join(words))