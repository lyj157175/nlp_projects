
def binary_accuracy(preds, y): 
    '''计算准确度，即预测和实际标签的相匹配的个数'''
    rounded_preds = torch.round(torch.sigmoid(preds)) #.round函数：四舍五入[neg: 0, pos: 1]
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train()  #训练时会用到dropout、归一化等方法，但测试的时候不能用dropout等方法
    
    for batch in iterator: 
        optimizer.zero_grad() 
        
        preds = model(batch.text).squeeze(1) # squeeze(1)压缩维度，和batch.label维度对上 
        loss = loss_fn(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)
             
        loss.backward() 
        optimizer.step() 
        
        epoch_loss += loss.item() * len(batch.label)
        #得到一个batch累加得到所有样本损失
        
        epoch_acc += acc.item() * len(batch.label)
        #（acc.item()：一个batch的正确率） *batch数 = 正确数, 累加得到所有训练样本正确数。
        
        total_len += len(batch.label)
        #计算train_iterator所有样本的数量，不出意外应该是17500
        
    return epoch_loss / total_len, epoch_acc / total_len #得到所有的batch的平均loss和平均acc




def evaluate(model, iterator, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.eval()
    #转换成测试模式，冻结dropout层或其他层。
    
    with torch.no_grad():  
        for batch in iterator: 
            
            preds = model(batch.text).squeeze(1)
            loss = loss_fn(preds, batch.label)
            acc = binary_accuracy(preds, batch.label)
            
            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len




import time 

def epoch_time(start_time, end_time): 
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
