import torch
from torchtext import data
import utlis
import torch.nn as nn
import torch.nn.functional as F

#创建两个Field 对象
TEXT = data.Field(tokenize='spacy',tokenizer_language='en_core_web_sm')
#torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
LABEL = data.LabelField(dtype=torch.float)


from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


import random
train_data, valid_data = train_data.split(random_state=random.seed(123)) 

TEXT.vocab = TEXT.build_vocab(train_data, max_size=25000)
#从预训练的词向量（vectors） 中，将当前(corpus语料库)词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）。
#预训练的 vectors 来自glove模型，每个单词有100维。glove模型训练的词向量参数来自很大的语料库，
#而我们的电影评论的语料库小一点，所以词向量需要更新，glove的词向量适合用做初始化参数。
LABEL.vocab = LABEL.build_vocab(train_data) 


BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#相当于把样本划分batch，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                        (train_data, valid_data, test_data), 
                                                        batch_size=BATCH_SIZE,
                                                        device=device)





class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        #vocab_size=25002，embedding_dim=100，padding_idx：遇到pad的单词用0填充     
        self.fc = nn.Linear(embedding_dim, output_dim) #output_dim=1
        
    def forward(self, text): # text: [seq_len,batch_size]
        embedded = self.embedding(text)  # embedded = [seq_len, batch_size, embedding_dim] 
        embedded = embedded.permute(1, 0, 2) # [batch_size, seq_len, embedding_dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)   
        # kernel size=(embedded.shape[1], 1), 会将句子的长度压缩到1，[batch_size, 1, embedding_dim]最终将每句话用一个向量表示
        # [batch_size, embedding_dim] 把句子长度的维度压扁为1，并降维
        
        return self.fc(pooled)  #（batch_size, embedding_dim）*（embedding_dim, output_dim）=（batch_size,output_dim）



VOCAB_SZIE = len(TEXT.vocab) #25002
EMBEDDING_DIM = 100
OUTPUT_DIM = 1 
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 


model = WordAVGModel(VOCAB_SZIE, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


import torch.optim as optim

optimizer = optim.Adam(model.parameters()) 
loss_fn = nn.BCEWithLogitsLoss()  
model = model.to(device) 
loss_fn = loss_fn.to(device) 


# kaggleGPU跑的花了2分钟
N_EPOCHS = 20
best_valid_loss = float('inf') 

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn) # 得到训练集每个epoch的平均损失和准确率
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_fn) # 验证集每个epoch的平均损失和准确率，model传入的是训练完的参数
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss: 
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pth')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')





