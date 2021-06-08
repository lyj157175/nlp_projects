import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam



def train(config, model, train_iter, dev_iter, test_iter):
    param_optimizer = list(model.named_parameters())
    #定义不需要衰减的参数，bert是基于transformer的encoder，add&norm层的参数不需要衰减
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        #需要衰减
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        #不需要衰减
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0
    dev_best_loss = float('inf')
    last_imporve = 0  # 上次校验集loss下降的batch数
    flag = False      # 记录是否有效果提升
    model.train()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}'.format(epoch+1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predit = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predit)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ''
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, imporve))
                model.train()
            total_batch = total_batch + 1
            if total_batch - last_imporve > config.require_improvement:
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def evaluate(config, model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)

