#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq + Attention
# 
# 在这份notebook当中，尽可能复现Luong的attention模型
# 
# 数据集非常小，只有一万多个句子的训练数据，所以训练出来的模型效果并不好。如果想训练一个好一点的模型，可以参考下面的资料。

# ## 更多阅读
# 
# #### 课件
# - [cs224d](http://cs224d.stanford.edu/lectures/CS224d-Lecture15.pdf)
# 
# 
# #### 论文
# - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
# - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025?context=cs)
# - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1406.1078)
# 
# 
# #### PyTorch代码
# - [seq2seq-tutorial](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# - [Tutorial from Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
# - [IBM seq2seq](https://github.com/IBM/pytorch-seq2seq)
# - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) 较好
# 
# 
# #### 更多关于Machine Translation
# - [Beam Search](https://www.coursera.org/lecture/nlp-sequence-models/beam-search-4EtHZ)
# - Pointer network 文本摘要
# - Copy Mechanism 文本摘要
# - Converage Loss 
# - ConvSeq2Seq
# - Transformer
# - Tensor2Tensor
# 
# #### TODO
# - 尝试对中文进行分词
# 
# #### NER
# - https://github.com/allenai/allennlp/tree/master/allennlp
# 




import os
import sys
import math
from collections import Counter 
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk


# 读入中英文数据
# - 英文使用nltk的word tokenizer来分词，并且使用小写字母
# - 中文使用单个汉字作为基本单元




def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("\t") 
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])           
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en, cn
 
train_file = "nmt/nmt/en-cn/train.txt"
dev_file = "nmt/nmt/en-cn/dev.txt"
train_en, train_cn = load_data(train_file)
dev_en, dev_cn = load_data(dev_file)





print(train_en[:10])
print(train_cn[:10])




UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences, max_words=50000):
    word_count = Counter() 
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1  
    ls = word_count.most_common(max_words) 
    print(len(ls)) #train_en：5491，train_cn=3193
    
    total_words = len(ls) + 2   
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
    
    word_dict["UNK"] = UNK_IDX 
    word_dict["PAD"] = PAD_IDX 
    return word_dict, total_words

en_dict, en_total_words = build_dict(train_en) 
cn_dict, cn_total_words = build_dict(train_cn)

inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}





print(en_total_words)
print(list(en_dict.items())[:10]) # 取出前10个
print(list(en_dict.items())[-10:]) # 取出后10个，可以看到"unk"和"pad"在最后
print("---"*20)
print(cn_total_words)
print(list(cn_dict.items())[:10]) # 查看中文
print(list(cn_dict.items())[-10:]) 
print("---"*20)
print(list(inv_en_dict.items())[:10]) # 键值对调换
print(list(inv_cn_dict.items())[:10]) 





def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
  
    length = len(en_sentences)
    # en_dict.get(w, 0)，返回键w对应的值,没有为0
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences] 
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
        # 按seq[x]的长度排序，最短句子的索引排在最前面
    
    # sort sentences by english lengths
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]    
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
    
    return out_en_sentences, out_cn_sentences

train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)





# sorted示例
seq = [5,4,6,9,10]
print(sorted(range(5), key=lambda x: seq[x])) 
print(sorted(range(4), key=lambda x: seq[x]))



print(train_en[:10])
print(train_cn[:10])
print("---"*20)
k=10000 # 
print([inv_cn_dict[i] for i in train_cn[k]]) 
print([inv_en_dict[i] for i in train_en[k]])
print(" ".join([inv_cn_dict[i] for i in train_cn[k]])) 
print(" ".join([inv_en_dict[i] for i in train_en[k]])) 




print(np.arange(0, 100, 15))
print(np.arange(0, 15))




# 按句子的数量自制batch
def get_batches(n, batch_size, shuffle=True):
    idx_list = np.arange(0, n, batch_size) 
    if shuffle:
        np.random.shuffle(idx_list) #打乱数据
    batches = []
    for idx in idx_list:
        batches.append(np.arange(idx, min(idx + batch_size, n)))
        # 所有batch放在一个大列表里
    return batches




get_batches(100,15) #随机打乱的




# 对句子做padding
def sent_padding(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs) 
    max_len = np.max(lengths) # 取出最长的的语句长度
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    
    # x: padding后的句子
    # x_lengths:每句话的length
    return x, x_lengths 

def gen_examples(en_sentences, cn_sentences, batch_size):
    batches = get_batches(len(en_sentences), batch_size)
    all_ex = []
    for batch in batches: 
        mb_en_sentences = [en_sentences[t] for t in batch]        
        mb_cn_sentences = [cn_sentences[t] for t in batch]
        # padding
        mb_x, mb_x_len = sent_padding(mb_en_sentences)
        mb_y, mb_y_len = sent_padding(mb_cn_sentences)
        
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
        # （英文句子，英文句子长度，中文句子，中文句子长度） 
    return all_ex


batch_size = 64
train_data = gen_examples(train_en, train_cn, batch_size)  # (mb_x, mb_x_len, mb_y, mb_y_len)
random.shuffle(train_data) 
dev_data = gen_examples(dev_en, dev_cn, batch_size) 




# 打印第一个batch的信息
print(train_data[0][0].shape) # 一个batch英文句子维度
print(train_data[0][1].shape) # 一个batch英文句子长度维度
print(train_data[0][2].shape) # 一个batch中文句子维度
print(train_data[0][3].shape) # 一个batch中文句子长度维度
print(train_data[0])




# ### 没有Attention的Encoder-Decoder版本
class PlainEncoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        #以英文为例，vocab_size=5493, hidden_size=100, dropout=0.2
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)      
        #第一个参数为input_size： embedding_dim
        #第二个参数为hidden_size：隐藏层维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths): 
        # pack,padded的操作需要句子排序，降序排列
        sorted_len, sorted_idx = lengths.sort(0, descending=True)  
        x_sorted = x[sorted_idx.long()]      
        embedded = self.dropout(self.embed(x_sorted))   # embedded：[64, 10, 100]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)  # hid: [1, 64, 100]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [64, 10, 100]

        _, original_idx = sorted_idx.sort(0, descending=False)
        # 上面lengths.sort()过，已经打乱了batch的句子的顺序，所以得恢复原位置，不然跟中文对不上
        out = out[original_idx.long()].contiguous()    #out:[64, 10, 100]
        hid = hid[:, original_idx.long()].contiguous()   #hid:[1, 64, 100], 在batch的维度上进行排序还原
  
        return out, hid[[-1]]  # hid取出最后一层




class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y, y_lengths, hid):
        # y: [64, 12]
        # hid: [1, 64, 100]
        # 中文的y和y_lengths
        
        # 中文句子的长度也不一样，也要和上面一样，不同长度的句子，也应该去掉没用的神经元
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()] #隐藏层也要排序
        # hid是Encoder的输出，和y_sorted都作为输入进入decoder层
        y_sorted = self.dropout(self.embed(y_sorted)) 
        # batch_size, output_length, embed_size
        
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)  # hid：[1, 64, 100], 默认传入0向量
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        # output_seq：[64, 12, 100]
        hid = hid[:, original_idx.long()].contiguous()
        # hid：[1, 64, 100]
        
        output = F.log_softmax(self.out(output_seq), -1)
        # output：[64, 12, 3195]
        
        return output, hid




class PlainSeq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(y, y_lengths, hid)      
        return output, None

    def translate(self, x, x_lengths, y, max_length=10):      
        encoder_out, hid = self.encoder(x, x_lengths)
        # encoder_out.shape=torch.Size([1, 7, 100])，1是batch_size,7是句子长度
        # hid.shape=torch.Size([1, 1, 100])，
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            # 训练的时候y是一个句子，一起decoder训练
            # 测试的时候y是个一个词一个词生成的，所以这里的y是传入的第一个单词，这里是bos
            # 同理y_lengths也是1
            output, hid = self.decoder(y=y,
                    y_lengths=torch.ones(batch_size).long().to(y.device),
                    hid=hid)         
            #刚开始循环bos作为模型的首个输入单词，后续更新y，下个预测单词的输入是上个输出单词
            # output.shape = torch.Size([1, 1, 3195])
            # hid.shape = torch.Size([1, 1, 100])

            y = output.max(2)[1].view(batch_size, 1)
            # .max(2)在第三个维度上取最大值,返回最大值和对应的位置索引，[1]取出最大值所在的索引
            preds.append(y) # 每次循环输出的y值就是预测值
            # preds = [tensor([[5]], device='cuda:0'), tensor([[24]], device='cuda:0'), ... tensor([[4]], device='cuda:0')]
            # torch.cat(preds, 1) = tensor([[ 5, 24,  6, 22,  7,  4,  3,  4,  3,  4]], device='cuda:0')
        return torch.cat(preds, 1), None  



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100



# 实例model
encoder = PlainEncoder(vocab_size=en_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
decoder = PlainDecoder(vocab_size=cn_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
model = PlainSeq2Seq(encoder, decoder)



# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: [64, 12, 3195] target: [64, 12]  mask: [64, 12]
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: batch_size * seq_len
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask  # 将input在1维，把target当索引进行取值
#这里算得就是交叉熵损失，前面已经算了F.log_softmax
#output.shape=torch.Size([768, 1])
#因为input.gather时，target为0的地方不是零了，mask作用是把padding为0的地方重置为零，
#因为在volab里0代表的也是一个单词，但是我们这里target尾部的0代表的不是单词
        output = torch.sum(output) / torch.sum(mask)
        # 均值损失，output前已经加了负号，所以这里还是最小化
        return output




model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())




def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():#不需要更新模型，不需要梯度
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)




def train(model, data, num_epochs=2):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            #（英文batch，英文长度，中文batch，中文长度）         
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()  
            # 前n-1个单词作为输入，后n-1个单词作为输出，因为输入的前一个单词要预测后一个单词
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
           
            mb_y_len[mb_y_len<=0] = 1
            
            optimizer.zero_grad()
            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            
            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            # None：在这个位置上增加一个维度
            #  tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            #         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            #         [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            #         [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
            #         [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
            #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]])
            mb_out_mask = mb_out_mask.float()  # 下三角矩阵

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)
            
            num_words = torch.sum(mb_y_len).item()  # 一个batch里多少个单词 
            total_loss += loss.item() * num_words 
            total_num_words += num_words
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            #为了防止梯度过大，设置梯度的阈值 
            optimizer.step()
            
            if it % 100 == 0:
                print("Epoch", epoch, "iteration", it, "loss", loss.item())

        print("Epoch", epoch, "Training loss", total_loss/total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data) 
            
train(model, train_data, num_epochs=20)




# 翻译个句子试试
def translate_dev(i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)   
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    # 这里传入bos作为首个单词的输入
    #translation=tensor([[ 8,  6, 11, 25, 22, 57, 10,  5,  6,  4]], device='cuda:0')
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
     
    trans = []
    for word in translation:
        if word != "EOS": 
            trans.append(word) 
        else:
            break
    print("".join(trans))
    
for i in range(500,520):
    translate_dev(i)
    print()




# ## Seq2Seq + attention版本
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        # hid: [2, batch_size, enc_hidden_size]
        
        hid = torch.cat([hid[-2], hid[-1]], dim=1) # 将最后一层的hid的双向拼接
        # hid: [batch_size, 2*enc_hidden_size]
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        # hid: [1, batch_size, dec_hidden_size]
        # out: [batch_size, seq_len, 2*enc_hidden_size]
        return out, hid



# #### Luong Attention
# - 根据context vectors和当前的输出hidden states，计算输出
# 这里我们计算第二种score的计算方法
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        # enc_hidden_size跟Encoder的一样
        super(Attention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        # mask = batch_size, output_len, context_len     # mask在Decoder中创建好了
        # output: batch_size, output_len, dec_hidden_size，就是Decoder的output
        # context: batch_size, context_len, 2*enc_hidden_size，就是Encoder的output
        # 这里Encoder网络是双向的，Decoder是单向的
    
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1) # input_len = context_len
        
        # 开始计算score，用到了第二种公式计算方式，先看懂这个网址：https://zhuanlan.zhihu.com/p/40920384
        # 通过decoder的hidden states加上encoder的hidden states来计算一个分数，用于计算权重
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                
            batch_size, input_len, -1) # batch_size, context_len, dec_hidden_size
        # 第一步，公式里的Wa先与hs做点乘，把Encoder output的enc_hidden_size换成dec_hidden_size。
        
        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len 
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) 
        # batch_size, output_len, context_len
        # 第二步，ht与上一步结果点乘，得到score

        attn.data.masked_fill(mask, -1e6)
        # .masked_fill作用请看这个链接：https://blog.csdn.net/candy134834/article/details/84594754
        # mask的维度必须和attn维度相同，mask为1的位置对应attn的位置的值替换成-1e6，
        # mask为1的意义需要看Decoder函数里面的定义

        attn = F.softmax(attn, dim=2) 
        # batch_size, output_len, context_len
        # 这个dim=2到底是怎么softmax的看下下面单元格例子
        # 第三步，计算每一个encoder的hidden states对应的权重。
        
        # context: batch_size, context_len, 2*enc_hidden_size，
        context = torch.bmm(attn, context) 
        # batch_size, output_len, 2*enc_hidden_size
        # 第四步，得出context vector是一个对于encoder输出的hidden states的一个加权平均
        
        # output: batch_size, output_len, dec_hidden_size
        output = torch.cat((context, output), dim=2) 
        # output：batch_size, output_len, 2*enc_hidden_size+dec_hidden_size
        # 第五步，将context vector和 decoder的hidden states 串起来。
        
        output = output.view(batch_size*output_len, -1)
        # output.shape = (batch_size*output_len, 2*enc_hidden_size+dec_hidden_size)
        output = torch.tanh(self.linear_out(output)) 
        # output.shape=(batch_size*output_len, dec_hidden_size)
        output = output.view(batch_size, output_len, -1)
        # output.shape=(batch_size, output_len, dec_hidden_size)
        # attn.shape = batch_size, output_len, context_len
        return output, attn


# #### Decoder
# - decoder会根据已经翻译的句子内容，和context vectors，来决定下一个输出的单词
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # x_len 是一个batch中文句子的长度列表
        # y_len 是一个batch英文句子的长度列表
        # a mask of shape x_len * y_len
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        
        x_mask = torch.arange(max_x_len, device=device)[None, :] < x_len[:, None]
        # print(x_mask.shape) = (batch_size, output_len) # 中文句子的mask
        y_mask = torch.arange(max_y_len, device=device)[None, :] < y_len[:, None]
        # print(y_mask.shape) = (batch_size, context_len) # 英文句子的mask
        
        mask = ( ~ x_mask[:, :, None] * y_mask[:, None, :]).byte()
        # mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        # 1-说明取反
        # x_mask[:, :, None] = (batch_size, output_len, 1)
        # y_mask[:, None, :] =  (batch_size, 1, context_len)
        # print(mask.shape) = (batch_size, output_len, context_len)
        # 注意这个例子的*相乘不是torch.bmm矩阵点乘，只是用到了广播机制而已。
        return mask
    
    def forward(self, encoder_out, x_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        
        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, x_lengths) # 这里真是坑，第一个参数位置是中文句子的长度列表

        output, attn = self.attention(output_seq, encoder_out, mask) 
        # output.shape=(batch_size, output_len, dec_hidden_size)
        # attn.shape = batch_size, output_len, context_len
        
        # self.out = nn.Linear(dec_hidden_size, vocab_size)
        output = F.log_softmax(self.out(output), -1) # 计算最后的输出概率
        # output =(batch_size, output_len, vocab_size)
        # 最后一个vocab_size维度 log_softmax
        # hid.shape = (1, batch_size, dec_hidden_size)
        return output, hid, attn


# #### Seq2Seq
# - 最后我们构建Seq2Seq模型把encoder, attention, decoder串到一起
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        # print(hid.shape)=torch.Size([1, batch_size, dec_hidden_size])
        # print(out.shape)=torch.Size([batch_size, seq_len, 2*enc_hidden_size])
        output, hid, attn = self.decoder(encoder_out=encoder_out, 
                    x_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        # output =(batch_size, output_len, vocab_size)
        # hid.shape = (1, batch_size, dec_hidden_size)
        # attn.shape = (batch_size, output_len, context_len)
        return output, attn
    

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(encoder_out=encoder_out, 
                    x_lengths=x_lengths,
                    y=y,
                    y_lengths=torch.ones(batch_size).long().to(y.device),
                    hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


# 训练
dropout = 0.2
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=en_total_words,
                    embed_size=embed_size,
                    enc_hidden_size=hidden_size,
                    dec_hidden_size=hidden_size,
                    dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words,
                    embed_size=embed_size,
                    enc_hidden_size=hidden_size,
                    dec_hidden_size=hidden_size,
                    dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())

train(model, train_data, num_epochs=30)



for i in range(100,120):
    translate_dev(i)
    print()

