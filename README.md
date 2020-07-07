# My_NLP_projects: 动手NLP项目汇总




### 1. 《Bert模型实现文本分类》项目： 
    -主要技术：Bert / Pytorch / Sklearn / Numpy / python
    -项目介绍：对18万的训练集，1万的验证集和1万的测试集进行数据处理，实现Dataloader，搭建Bert模型，完成模型的训练和数据的评估

### 2. 《训练词向量》项目：

    -主要技术：Pytorch / numpy / sklearn
    对kim的“Distributed Representations of Words and Phrases and their Compositionality”论文进行复现。实现skip-gram模型，用text8数据集来训练输入输出两个词向量矩阵，保存输入词向量矩阵并在simlex-999、men、wordsim353三个数据集上进行词向量的评估。

### 3. 《训练语言模型》项目：

    -主要技术：Pytorch / Torchtext / RNN / LSTM  
    -项目介绍：使用text8作为数据集，利用Torchtext创建vocab，BPTTIterator来迭代数据集，选择LSTM模型训练并保存最好的语言模型，用Perplexity对语言模型评估

### 4. 《情感分类》项目： 

    -主要技术：Pytorch / Torchtext / Spacy / RNN / LSTM
    -项目介绍：使用IMDb电影评论数据集并用torchtext做数据预处理，分别用Word Averaging/RNN/CNN三种模型来做情感分析，检测一段文本的情感是正面还是负面的，保存三种最好的训练模型结果并对其进行评估。

5.《机器翻译》项目：



