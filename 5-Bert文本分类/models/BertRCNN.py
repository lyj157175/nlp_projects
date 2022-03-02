import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    def __init__(self, dataset):
        self.model_name = "BertRCNN"
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.datasetpkl = dataset + '/data/dataset.pkl'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))
        self.hidden_size = 768
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)


    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask = mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


