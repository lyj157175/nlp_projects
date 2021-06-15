## nlp_projects: 
> NLP项目汇总，邮箱：lyj157175@163.com

------

### 1-训练词向量
- “Distributed Representations of Words and Phrases and their Compositionality”论文的简单复现，实现skip-gram模型。用text8数据集来训练输入输出两个词向量矩阵，保存输入词向量矩阵后在simlex-999、men、wordsim353三个数据集上进行词向量的评估
-------

### 2-训练语言模型
- 使用text8作为数据集，选择LSTM模型训练并保存语言模型，在测试集上用Perplexity对语言模型评估
---------

### 3-情感分类 
- 使用IMDb电影评论数据集并用torchtext做数据预处理，分别用Word Averaging/RNN/CNN三种模型来做情感分析，检测一段文本的情感是正面还是负面的，保存三种最好的训练模型结果并对其进行评估
---------

### 4-机器翻译

- 没有attention的encoder-decoder模型实现机器翻译

- encoder-decoder+attention实现机器翻译

------

### 5-Bert文本分类

- THUCNews数据集，包括18万条训练集，1万条验证集和1万的测试集，利用Bert预训练模型，完成各种Bert+模型的训练和文本分类

提供预训练模型下载（模型来自https://github.com/ymcui/Chinese-BERT-wwm里的RoBERTa-wwm-ext-large, Chinese）：

链接：https://pan.baidu.com/s/1LonaTPprR6q9x4zPhj9uKQ     提取码：t93l 

模型下载后放在  ‘bert_pretrained/roberta’  文件夹下即可

------

### 6-文本问答系统

- 数据集：SQuAD1.0

BiDAF模型完成阅读理解任务

