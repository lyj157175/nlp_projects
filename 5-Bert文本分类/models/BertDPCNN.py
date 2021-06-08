#!/usr/bin/python
# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    def __init__(self, dataset):
        self.model_name = "BertDPCNN"
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
        self.num_filters = 250
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size))

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3,1), stride=2)

        self.padd1 = nn.ZeroPad2d((0,0,1,1))
        self.padd2 = nn.ZeroPad2d((0,0,0,1))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask = mask, output_all_encoded_layers = False)
        out = encoder_out.unsqueeze(1) #[batch_size, 1, seq_len, embed]
        out = self.conv_region(out) #[batch_size, 250, seq_len-3+1, 1]

        out = self.padd1(out) #[batch_size, 250, seq_len,1]
        out = self.relu(out)
        out = self.conv(out) #[batch_size, 250, seq_len-3+1,1]
        out = self.padd1(out)  # [batch_size, 250, seq_len,1]
        out = self.relu(out)
        out = self.conv(out)  # [batch_size, 250, seq_len-3+1,1]
        while out.size()[2] > 2:
            out = self._block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out

    def _block(self, x):
        x = self.padd2(x)
        px = self.max_pool(x)
        x = self.padd1(px)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padd1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        return x


























