import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud  
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# 设定一些超参数
K = 100 # number of negative samples
C = 3 # nearby words threshold
NUM_EPOCHS = 2 # The number of epochs of training
MAX_VOCAB_SIZE = 30000 # the vocabulary size
BATCH_SIZE = 128 # the batch size
LEARNING_RATE = 0.2 # the initial learning rate
EMBEDDING_SIZE = 100
        
LOG_FILE = "word-embedding.log"


# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()


with open("project1_word_embedding/data/text8_train.txt", "r") as f:
    text = f.read()
text = [w for w in word_tokenize(text.lower())]  #将每个单词切分并放入列表
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))   #统计最前面的出现的29999个单词,还有一个为unk
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))   #将text中没有算的词标为unk
idx_to_word = [word for word in vocab.keys()]   #将word依次从高到低排列
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}  #词表
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)   #词频
word_freqs = word_freqs ** (3./4.)    
word_freqs = word_freqs / np.sum(word_freqs) # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)
# print(VOCAB_SIZE)



#实现dataloader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        #dict.get(key, default=None)，返回指定键的值，如果值不存在在返回default
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE-1) for t in text]   #将text的词依次给一个id，找不到的给29999
        self.text_encoded = torch.Tensor(self.text_encoded).long()       
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)
        
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]       #每个batch_size = 128
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))    #中心词的周围单词的index
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]    #防止超出text长度
        pos_words = self.text_encoded[pos_indices]    #周围单词，6个
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)  #负采样
        
        return center_word, pos_words, neg_words 


#创建dataset和dataloader
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

#迭代每个batch的数据
# next(iter(dataloader))



#定义网络模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
#         initrange = 0.5 / self.embed_size

        # self.in_embed = nn.Embedding(30000, 100)  # 30000个单词，维度100
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        
#         self.in_embed.weight.data.uniform_(-initrange, initrange)
#         self.out_embed.weight.data.uniform_(-initrange, initrange)
           
                    
    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels:   中心词周围的单词 [batch_size * (C * 2)]
        neg_labelss:  中心词周围没有出现的单词 [batch_size, (C * 2 * K)]
        
        return: loss, [batch_size]
        '''
        batch_size = input_labels.size(0)
        #3个tensor输入
        input_embedding = self.in_embed(input_labels) # [batch_size , embed_size]
        pos_embedding = self.in_embed(pos_labels) # [batch_size , (2*C) , embed_size]
        neg_embedding = self.in_embed(neg_labels) # [batch_szie , (2*C * K) , embed_size]
      
        #计算loss
        #torch.bmm((b,n.m)*(b,m,p)) = (b,n.p)
        #input_embedding.unsqueeze(2) =>  [batch_size , embed_size, 1] 
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze() # [batch_size , (2*C)]
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze() # [batch_size, (2*C*K)]

        log_pos = F.logsigmoid(log_pos).sum(1) # batch_size
        log_neg = F.logsigmoid(log_neg).sum(1) # batch_size
       
        loss = log_pos + log_neg
        
        return -loss
    
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()



def evaluate(filename, embedding_weights): 
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity

def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]



optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        # TODO
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
            
        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
            
        
        if i % 2000 == 0:
            embedding_weights = model.input_embeddings()
            sim_simlex = evaluate("simlex-999.txt", embedding_weights)
            sim_men = evaluate("men.txt", embedding_weights)
            sim_353 = evaluate("wordsim353.csv", embedding_weights)
            with open(LOG_FILE, "a") as fout:
                print("epoch: {}, iter: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                    e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                fout.write("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                    e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                
    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))


# 在 MEN 和 Simplex-999 数据集上做评估
embedding_weights = model.input_embeddings()
print("simlex-999", evaluate("simlex-999.txt", embedding_weights))
print("men", evaluate("men.txt", embedding_weights))
print("wordsim353", evaluate("wordsim353.csv", embedding_weights))


# 寻找nearest neighbors
for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word))

## 单词之间的关系
man_idx = word_to_idx["man"] 
king_idx = word_to_idx["king"] 
woman_idx = word_to_idx["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(idx_to_word[i])