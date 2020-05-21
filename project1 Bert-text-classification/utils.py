from tqdm import tqdm   # 显示进度
import torch
import time
from datetime import timedelta   
import pickle as pkl
import os

PAD, CLS = '[PAD]', '[CLS]'

#处理train.txt, dev.txt, test.txt的原始文件为模型可读
def load_dataset(file_path, config):
    '''
    return [ids, label, ids_len, mask]
    '''
    contents = []
    with open(file_path, 'r', encoding = 'UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            seq_len = len(token)
            mask = []
            
            pad_size = config.pad_size
            if pad_size:
                if seq_len < pad_size:
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    token_ids = token_ids + [0] * (pad_size - seq_len)
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size 
            content = contents.append((token_ids, label, seq_len, mask))
    return contents       



#处理dataset.pkl为模型可读
def build_dataset(config):
    '''
    return train_data, dev_data, test_data
    '''
    #如果datasetpkl存在则直接取出返回train, dev, test
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    #如果datasetpkl不存在则利用load_dataset处理train.txt, dev.txt, test.txt
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))  #写入datasetpkl文件

    return train, dev, test 

class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size 
        self.dataset = dataset
        self.device = device
        self.num_batches = len(self.dataset) // self.batch_size  # dataset有几个batch_size
        self.residue = False    #记录batch是否有残余
        if len(self.dataset) % self.num_batches != 0:
            self.residue = True
        self.index = 0
    
    def _to_tensor(self, batch):
        x = torch.LongTensor([item[0] for item in batch]).to(self.device)
        y = torch.LongTensor([item[1] for item in batch]).to(self.device)
        seq_len = torch.LongTensor([item[2] for item in batch]).to(self.device)
        mask = torch.LongTensor([item[3] for item in batch]).to(self.device)
        return (x, seq_len, mask), y
    
    def __next__(self):
        if self.residue and self.index == self.num_batches:
            batches = self.dataset[self.index * self.batch_size:]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1 
            batches = self._to_tensor(batches)
            return batches 
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches


#处理数据为dataloader, 返回数据形式为 (x, seq_len, mask), y
def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter 


def get_time_dif(start_time):
    '''
    获取使用时间
    '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds = int(round(time_dif)))
