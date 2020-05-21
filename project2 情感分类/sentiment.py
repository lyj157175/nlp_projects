import torch
from torchtext import data
import spacy
nlp = spacy.load("en_core_web_sm")

SEED = 1234

torch.manual_seed(SEED) #为CPU设置随机种子
torch.cuda.manual_seed(SEED)#为GPU设置随机种子
torch.backends.cudnn.deterministic = True  #在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。


# 第一步：导入豆瓣电影数据集，只有训练集和测试集
TEXT = data.Field(tokenize='spacy')#torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
LABEL = data.LabelField(dtype=torch.float)
#首先，我们要创建两个Field 对象：这两个对象包含了我们打算如何预处理文本数据的信息。
#spaCy:英语分词器,类似于NLTK库，如果没有传递tokenize参数，则默认只是在空格上拆分字符串。
#LabelField是Field类的一个特殊子集，专门用于处理标签。 

from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


print(vars(train_data.examples[0])) #可以查看数据集长啥样子


# 第二步：训练集划分为训练集和验证集
import random
train_data, valid_data = train_data.split(random_state=random.seed(SEED)) #默认split_ratio=0.7

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 第三步：用训练集建立vocabulary，就是把每个单词一一映射到一个数字。
# TEXT.build_vocab(train_data, max_size=25000)
# LABEL.build_vocab(train_data)
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
#从预训练的词向量（vectors） 中，将当前(corpus语料库)词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）。
#预训练的 vectors 来自glove模型，每个单词有100维。glove模型训练的词向量参数来自很大的语料库，
#而我们的电影评论的语料库小一点，所以词向量需要更新，glove的词向量适合用做初始化参数。
LABEL.build_vocab(train_data) 

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.stoi) #语料库单词频率越高，索引越靠前。前两个默认为unk和pad。
print(TEXT.vocab.itos[:10]) #查看TEXT单词表

# 第四步：创建iterators，每个itartion都会返回一个batch的样本
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#相当于把样本划分batch，只是多做了一步，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),  
    batch_size=BATCH_SIZE,
    device=device)

#第五步：创建Word Averaging模型
import torch.nn as nn
import torch.nn.functional as F

class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        #初始化参数，
        super(WordAVGModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        #vocab_size=词汇表长度，embedding_dim=每个单词的维度
        #padding_idx：如果提供的话，输出遇到此下标时用零填充。这里如果遇到padding的单词就用0填充。
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        #output_dim输出的维度，一个数就可以了，=1
        
    def forward(self, text):
        embedded = self.embedding(text) 
        #text下面会指定，为一个batch的数据
        # embedded = [sent len, batch size, emb dim] 
        # sent len：一条评论的单词数 
        # batch size：一个batch有多少条评论
        # emb dim：一个单词的维度
        # 假设[sent len, batch size, emb dim]=（1000，64，100）
        #这个代码我猜测进行了运算：（text：1000，64，25000）*（self.embedding：1000，25000，100）= （1000，64，100）
        
        embedded = embedded.permute(1, 0, 2) 
        # [batch size, sent len, emb dim]更换顺序
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        # [batch size, embedding_dim] 把单词长度的维度压扁为1，并降维
        
        return self.fc(pooled)  
        #（batch size, embedding_dim）*（embedding_dim, output_dim）=（batch size,output_dim）

INPUT_DIM = len(TEXT.vocab) #25002
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 
#PAD_IDX = 1 为pad的索引

model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

def count_parameters(model): #统计参数，可以不用管
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

## 第六步：初始化参数
pretrained_embeddings = TEXT.vocab.vectors    #(25002, 100)
model.embedding.weight.data.copy_(pretrained_embeddings) #遇到_的语句直接替换，不需要另外赋值=
#把上面vectors="glove.6B.100d"取出的词向量作为初始化参数，数量为25000*100个参数

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token] #UNK_IDX=0

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#词汇表25002个单词，前两个unk和pad也需要初始化


# 第七步：训练模型
import torch.optim as optim 

optimizer = optim.Adam(model.parameters()) #定义优化器
criterion = nn.BCEWithLogitsLoss()  #定义损失函数，这个BCEWithLogitsLoss特殊情况，二分类损失函数
model = model.to(device) #送到gpu上去
criterion = criterion.to(device) #送到gpu上去

def binary_accuracy(preds, y): #计算准确率
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    #.round函数：四舍五入
    
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion): 
   
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train() #model.train()代表了训练模式
    #这步一定要加，是为了区分model训练和测试的模式的。
    #有时候训练时会用到dropout、归一化等方法，但是测试的时候不能用dropout等方法。
    
    
    for batch in iterator: #iterator为train_iterator
        optimizer.zero_grad() #加这步防止梯度叠加
        
        predictions = model(batch.text).squeeze(1)
        #batch.text 就是上面forward函数的参数text
        #压缩维度，不然跟batch.label维度对不上
        
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        optimizer.zero_grad()
        loss.backward() #反向传播
        optimizer.step() #梯度下降
        
        epoch_loss += loss.item() * len(batch.label)
        #loss.item()已经本身除以了len(batch.label)
        #所以得再乘一次，得到一个batch的损失，累加得到所有样本损失。
        
        epoch_acc += acc.item() * len(batch.label)
        #（acc.item()：一个batch的正确率） *batch数 = 正确数
        #train_iterator所有batch的正确数累加。
        
        total_len += len(batch.label)
        #计算train_iterator所有样本的数量，不出意外应该是17500
        
    return epoch_loss / total_len, epoch_acc / total_len
    #epoch_loss / total_len ：train_iterator所有batch的损失
    #epoch_acc / total_len ：train_iterator所有batch的正确率


def evaluate(model, iterator, criterion):
      
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    
    model.eval()
    
    #转换成测试模式，冻结dropout层或其他层。
    with torch.no_grad():
        for batch in iterator: 
            #iterator为valid_iterator
            
            #没有反向传播和梯度下降
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            
            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)
            
    model.train() #调回训练模式   
    
    return epoch_loss / total_len, epoch_acc / total_len


import time 
def epoch_time(start_time, end_time):  #查看每个epoch的时间
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 第八步：查看模型运行结果
N_EPOCHS = 10

best_valid_loss = float('inf') #无穷大

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss: #只要模型效果变好，就存模型
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# 第九步：预测结果
model.load_state_dict(torch.load("wordavg-model.pt"))
#用保存的模型参数预测数据

import spacy  #分词工具，跟NLTK类似
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]#分词
    indexed = [TEXT.vocab.stoi[t] for t in tokenized] 
    #sentence的索引
    
    tensor = torch.LongTensor(indexed).to(device) #seq_len
    tensor = tensor.unsqueeze(1) 
    #seq_len * batch_size（1）
    
    prediction = torch.sigmoid(model(tensor))
    #tensor与text一样的tensor
    
    return prediction.item()

predict_sentiment("I love This film bad")
predict_sentiment("This film is great")