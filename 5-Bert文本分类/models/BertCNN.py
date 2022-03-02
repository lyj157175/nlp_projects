import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    def __init__(self, dataset):
        self.model_name="BertCNN"
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.datasetpkl = dataset + '/data/dataset.pkl'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement= 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))
        self.hidden_size = 768
        self.filter_sizes = (2,3,4)
        self.num_filters = 256
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        #调用Conv2d()做卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )

        self.droptout = nn.Dropout(config.dropout)
        #[卷积核数量*卷积种类，输出类别]
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x


    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0] #对应输入的句子 shape[128,32]
        mask = x[2] #对padding部分进行mask shape[128,32]
        encoder_out, pooled = self.bert(context, attention_mask = mask, output_all_encoded_layers = False) #shape [128,768]
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)for conv in self.convs], 1)
        out = self.droptout(out)
        out = self.fc(out)
        return out

