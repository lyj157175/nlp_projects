import torch
import torch.nn as nn



class encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, dropout=0.2):
		super(encoder, self).__init__()
		self.embed = nn.Embedding(vocab_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
		self.


