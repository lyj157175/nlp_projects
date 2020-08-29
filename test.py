class Encoder(nn.Module):

	def __init__(self, vocab_size, hidden_size, dropout=0.2):
		super(Encoder, self).__init__()
		self.embed = nn.Embedding(vocab_size, hidden_size)
		self.rnn = nn.GRU(hidden_size, hiden_size, batch_first=True)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths):
		sorted_len, sorted_idx = lengths.sort(0, descending=True)
		x_sorted = x[sorted_idx.long()]
		embed = self.dropout(self.embed(x_sorted))

		packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_len.long().cpu().data.numpy(), batch_first=True)
		out, hid = self.rnn(packed_embed)
		out, _ = nn.utils.rnn.padd_packed_sequence(out, batch_first=True)

		_, original_idx = sorted_idx.sort(0, descending=False)
		out = out[original_idx.long()].contiguous()
		hhid = hid[:, original_idx.long()].contiguous() 

		return out, hid[[-1]]
class Decoder(nn.Modeule):

	def __init__(self, vocab_size, hidden_size, dropout=0.2)

		self.embed = nn.Embedding(vocab_size, hidden_szie)
		self.rnn = nn.GRU(hidden_size, hidden_size, batch_first = True)
		self.fc = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(dropout)



	def forward(self, y, y_lengths, hid):
		sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        y_sorted = self.dropout(self.embed(y_sorted))

        out = F.log_softmax(self.fc(out), -1)
        return out, hid

class Seq2Seq(nn):
	def __init__(self, encoder, decoder):
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x, x_length, y, y_length):
		encoder_out, hid = self.encoder(x, x_length)
		out, hid = self.decoder(y, y_lengths, hid)
		return out, None

	def translate(self, x, x_length, y, max_len=10):
		encoder_out, hid = self.encoder(x, x_length)
		pred=[]
		batch_size = x.shape(0)
		attn = []
		for i in range(max_lenth):
			out, hid = self.decoder(y, 
				y_lenths= torch.ones(batch_size).long().to(device),
				hid)





device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100


encoder = Encoder(vocab_size=en_vocab_word, hideden_sie=100, dropout)
decoder = Decoder(vocab_size=cn_vocab_word, hidden_size=100, dropout)
model =Seq2Seq(encoder, decoder)


mdoel = model.to(device)
loss_fn = languagemodelcriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


def train(model, data):
	model.train()
	for i, (x, x_length, y, y_length) in  enumerate(data):
		x = torch.from_numpy(x).to(device).long()
		x_length = torch.from_numpy(x_length).to(device).long()

		y_in = torch.from_numpy(y[:,:-1]).to(device).long()
		y_out = torch.from_numpy(y[:,1:]).to(device).long()
		y_length = torch.from_numpy(y_length).to(device).long()

		y_length[y_length<=0] = 1

		optimizer.zero_grad()
		y_pred, attn = model(x, x_length, y_in, y_length)
		y_mask = torch.arange(y_length.max().item(), device=device)[None, :] 



		loss = loss_fn(y_pred, y_out)

		








