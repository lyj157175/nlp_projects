import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, datapath):
        self.model_name = 'LiuBertModel'
        self.train_path = datapath + '/data/train.txt'
        self.dev_path = datapath + '/data/dev.txt'
        self.test_path = datapath + '/data/test.txt'
        self.datasetpkl = datapath + '/data/dataset.pkl'
        # self.class_list = [x.strip() for x in open(datapath + '/data/class.txt').readlines()]
        self.class_list = ['finance','realty','stocks','education','science','society','politics','sports','game','entertainment']

        #保存模型训练结果的位置
        self.save_path = datapath + '/saved_dict/' + self.model_name + '.ckpt'
        #设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #若超过1000个bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000

        self.num_class = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128 
        self.learning_rate = 1e-5
        #每句话处理的长度(短填，长切）
        self.pad_size = 32

        #bert预训练模型位置
        self.bert_path = 'bert_pretrain'
        #bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        #bert隐层层个数
        self.hidden_size = 768
        

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        #从预训练好的模型中加入bertmodel
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 可以对bert的参数进行微调,对业务灵活操作
        for param in self.bert.parameters():
            param.requires_grad = True 
        self.fc = nn.Linear(config.hidden_size, config.num_class)
    
    def forward(self, x):   
        # bertmodel的输入x[ids, seq_len, mask]
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out 

