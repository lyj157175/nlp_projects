class PlainEncoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        #以英文为例，vocab_size=5493, hidden_size=100, dropout=0.2
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)      
        #第一个参数为input_size： embedding_dim
        #第二个参数为hidden_size：隐藏层维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths): 
        # pack,padded的操作需要句子排序，降序排列
        sorted_len, sorted_idx = lengths.sort(0, descending=True)  
        x_sorted = x[sorted_idx.long()]      
        embedded = self.dropout(self.embed(x_sorted))   # embedded：[64, 10, 100]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)  # hid: [1, 64, 100]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [64, 10, 100]

        _, original_idx = sorted_idx.sort(0, descending=False)
        # 上面lengths.sort()过，已经打乱了batch的句子的顺序，所以得恢复原位置，不然跟中文对不上
        out = out[original_idx.long()].contiguous()    #out:[64, 10, 100]
        hid = hid[:, original_idx.long()].contiguous()   #hid:[1, 64, 100], 在batch的维度上进行排序还原
  
        return out, hid[[-1]]  # hid取出最后一层



class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y, y_lengths, hid):
        # y: [64, 12]
        # hid: [1, 64, 100]
        # 中文的y和y_lengths
        
        # 中文句子的长度也不一样，也要和上面一样，不同长度的句子，也应该去掉没用的神经元
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()] #隐藏层也要排序
        # hid是Encoder的输出，和y_sorted都作为输入进入decoder层
        y_sorted = self.dropout(self.embed(y_sorted)) 
        # batch_size, output_length, embed_size
        
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)  # hid：[1, 64, 100], 默认传入0向量
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        # output_seq：[64, 12, 100]
        hid = hid[:, original_idx.long()].contiguous()
        # hid：[1, 64, 100]
        
        output = F.log_softmax(self.out(output_seq), -1)
        # output：[64, 12, 3195]
        
        return output, hid



class PlainSeq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid = self.decoder(y, y_lengths, hid)      
        return output, None

    def translate(self, x, x_lengths, y, max_length=10):
        #x是一个句子，用数值表示
        #y是句子的长度
        #y是“bos”的数值索引=2
        
        encoder_out, hid = self.encoder(x, x_lengths)
        # encoder_out.shape=torch.Size([1, 7, 100])，1是batch_size,7是句子长度
        # hid.shape=torch.Size([1, 1, 100])，
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            # 训练的时候y是一个句子，一起decoder训练
            # 测试的时候y是个一个词一个词生成的，所以这里的y是传入的第一个单词，这里是bos
            # 同理y_lengths也是1
            output, hid = self.decoder(y=y,
                    y_lengths=torch.ones(batch_size).long().to(y.device),
                    hid=hid)         
            #刚开始循环bos作为模型的首个输入单词，后续更新y，下个预测单词的输入是上个输出单词
            # output.shape = torch.Size([1, 1, 3195])
            # hid.shape = torch.Size([1, 1, 100])
            

            y = output.max(2)[1].view(batch_size, 1)
            # .max(2)在第三个维度上取最大值,返回最大值和对应的位置索引，[1]取出最大值所在的索引
            preds.append(y) # 每次循环输出的y值就是预测值
            # preds = [tensor([[5]], device='cuda:0'), tensor([[24]], device='cuda:0'), ... tensor([[4]], device='cuda:0')]
            # torch.cat(preds, 1) = tensor([[ 5, 24,  6, 22,  7,  4,  3,  4,  3,  4]], device='cuda:0')
        return torch.cat(preds, 1), None  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100

# 实例model
encoder = PlainEncoder(vocab_size=en_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
decoder = PlainDecoder(vocab_size=cn_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
model = PlainSeq2Seq(encoder, decoder)