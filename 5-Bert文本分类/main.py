import time
import torch
import numpy as np
from importlib import import_module   
import argparse
import utils
import train




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text-Classsification')
    parser.add_argument('--model_name', type=str, default='Bert')
    args = parser.parse_args()

    np.random.seed(1234)      
    torch.manual_seed(1234)       
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True 

    dataset = 'THUCNews'    
    model_path = import_module('models.' + args.model_name) 
    config = model_path.Config(dataset)

    train_data, dev_data, test_data = utils.bulid_dataset(config)
    train_iter = utils.bulid_dataloader(train_data, config)
    dev_iter = utils.bulid_dataloader(dev_data, config)
    test_iter = utils.bulid_dataloader(test_data, config)

    model = model_path.Model(config).to(config.device)
    # 模型训练，评估与测试
    train.train(config, model, train_iter, dev_iter, test_iter)
    train.test(config, model, test_iter)
