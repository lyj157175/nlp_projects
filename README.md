# NLP_projects
NLP_projects：


# 1. 《Bert模型实现文本分类》项目： 
    -主要技术：Bert / Pytorch / Sklearn / Numpy / python
    -项目介绍：对18万的训练集，1万的验证集和1万的测试集进行数据处理，实现Dataloader，搭建Bert模型，完成模型的训练和数据的评估

# 2. 《情感分类》项目： 
    -主要技术：Pytorch / Torchtext / Spacy / RNN / LSTM
    -项目介绍：用LSTM模型和TorchText做情感分析，来检测一段文本的情感是正面的还是负面的。主要使用IMDb 数据集，即电影的评论 

# 3. 《训练语言模型》项目：
    -主要技术：Pytorch / Torchtext / RNN / LSTM  
    -项目介绍：用Torchtext创建vocab，BPTTIterator来迭代数据集，选择LSTM模型训练并保存最好的模型，最后用Perplexity来评估语言模型

# 4. word-embedding项目： pytorch训练词嵌入
    -skip-gram预测周围词并不是目的，目的是训练词向量参数，找到两个矩阵
    input embedding: 30000*100
    output embedding: 30000*100

    学习词向量的概念
    用Skip-thought模型训练词向量
    学习使用PyTorch dataset和dataloader
    学习定义PyTorch模型
    学习torch.nn中常见的Module
    Embedding
    学习常见的PyTorch operations
    bmm
    logsigmoid
    保存和读取PyTorch模型

