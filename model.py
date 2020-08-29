# RNN, LSTM, GRU等rnn模型的数据格式：
# 输入维度: [bseq_len, batch_size, input_dim]


### rnn
rnn = nn.RNN(input_dim, hidden_dim, num_layers, bidirectional=False, batch_first=False)
x = torch.randn(seq_len, batch_size, input_dim)
out, ht = rnn(x) 
# out: [seq_len, batch_size, num_directions * hidden_dim]
# ht: [num_layers*num_directions, batch_size, hidden_dim]

out和ht[-1]相等，隐藏单元就是输出的最后一个单元


### lstm
lstm = nn.LSTM()
# 输入维度 50，隐层100维，两层
lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
# 输入序列seq= 10，batch =3，输入维度=50
x = torch.randn(seq_len, batch_size, input_dim)
out, (hn, cn) = lstm(x) # 使用默认的全 0 隐藏状态
# out: [seq_len, baych_size, num_directions * hidden_dim]
# hn=hc: [num_layers*num_directions, batch_size, hidden_dim]

out[-1,:,:]和hn[-1,:,:]相等

### gru和rnn一样
gru = nn.GRU(input_dim, hidden_dim, num_layers) 
x = torch.randn(seq_len, batch_size, input_dim)
out, hn = gru(x)


