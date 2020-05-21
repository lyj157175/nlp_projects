import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import  BertAdam


def train(config, model, train_iter, dev_iter, test_iter):

    start_time = time.time()
    #启动 BatchNormalization 和 dropout
    model.train()
    #拿到所有mode种的参数
    param_optimizer = list(model.named_parameters())
    #不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        #需要衰减 
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        #不需要衰减
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}     
    ]

    optimizer = BertAdam(params = optimizer_grouped_parameters,
                        lr = config.learning_rate,
                        warmup = 0.5,
                        t_total = len(train_iter) * config.num_epochs)

    total_batch = 0 #记录进行多少batch
    dev_best_loss = float('inf') #记录校验集合最好的loss
    last_improve = 0 #记录上次校验集loss下降的batch数
    flag = False #记录是否很久没有效果提升，停止训练
    model.train()

    for epoch in range(config.num_epochs):
        print('epoch: {}/{}'.format((epoch+1), config.num_epochs))
        # it = iter(train_iter)
        for i,(trains, labels) in enumerate(train_iter): #labels.shape [128]
            outputs = model(trains)  #shape [128, 10]
            model.zero_grad
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                label = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cup()  #max()[1]最大的索引
                train_acc = metrics.accuracy_scorce(label, predict)
                
                #校验集
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'  #loss在下降每步输出就打星
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1     
            if total_batch - last_improve > config.require_improvement:
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def evaluate(config, model, iters, test=False):
    model.eval()
    loss_total = 0
    labels_all = np.array([], dtype = int)
    predict_all = np.array([], dtype = int)
    with torch.no_grad():
        for iter, labels in iters:
            outputs = model(iter)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss 
            label = labels.data.cpu().numpy()
            predict = torch.max(outputs, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits = 4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total/len(iters), report, confusion

    return acc, loss_total/len(iters) 


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：",time_dif)



