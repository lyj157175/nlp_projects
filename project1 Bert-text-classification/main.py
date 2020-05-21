import time 
import torch
import numpy as np
# from importlib import import_module #动态导入对象
import argparse  #python自带的参数解析包
import train
import utils
import LiuBertModel


# parser = argparse.ArgumentParser(description = 'Bert-text-classification') 
# parser.add_argument('--model', type = str, default = 'BertModel', help = 'chosse a model')
# args = parser.parse_args()


if __name__ == '__main__':
    datapath = 'THUCNews'  #数据集地址
    # model_name = args.model 
    # LiuBertModel = import_module(model_name)   #x:实例化BertModel文件，包括Config和Model类
    config = LiuBertModel.Config(datapath) #配置实例化

    #初始化保证每次运行结果一样
    np.random.seed(1)      
    torch.manual_seed(1)       
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True   

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)
    train_iter = utils.build_iterator(train_data, config)  #(ids*b, seq_len*b, mask*b), y*b)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)
    
    #获取使用的时间
    time_dif = utils.get_time_dif(start_time)   
    print("模型开始之前，准备数据时间：", time_dif)
    
    #模型训练
    model = LiuBertModel.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    # train.test(config, model, test_iter)
    

