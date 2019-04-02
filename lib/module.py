import torch
import torch.nn as nn

import math

class LayerNorm(nn.Module):
	"""Layer Normalization class"""

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(MLP, self).__init__()

		self.dim_in = dim_in
		self.dim_out = dim_out

		self._init_params()

	def _init_params(self):
		self.mlp = nn.Linear(in_features = self.dim_in,
							 out_features = self.dim_out)

	def forward(self, inp):
		proj_inp = self.mlp(inp)
		return proj_inp

class BiLstm(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(BiLstm, self).__init__()

		self.dim_in = dim_in
		self.dim_out = dim_out

		self._init_params()

	def _init_params(self):
		self.bilstm = nn.LSTM(input_size = self.dim_in,
							  hidden_size = self.dim_out,
							  bidirectional = True)

	def forward(self, inp, inp_len):
		sorted_inp_len, sorted_idx = torch.sort(inp_len, dim = 0, descending=True)
		sorted_inp = torch.index_select(inp, dim = 1, index = sorted_idx)

		pack_inp = torch.nn.utils.rnn.pack_padded_sequence(sorted_inp, sorted_inp_len)
		proj_inp, _ = self.bilstm(pack_inp)
		proj_inp = torch.nn.utils.rnn.pad_packed_sequence(proj_inp)

		unsorted_idx = torch.zeros(sorted_idx.size()).long().cuda().scatter_(0, sorted_idx, torch.arange(inp.size()[1]).long().cuda())
		unsorted_proj_inp = torch.index_select(proj_inp[0], dim = 1, index = unsorted_idx)

		return unsorted_proj_inp

class Word_Emb(nn.Module):
	def __init__(self,
				 fine_tune,
				 pre_train_emb,
				 part_point,
				 size_vocab,
				 dim_emb):
		super(Word_Emb, self).__init__()

		self.fine_tune = fine_tune
		self.pre_train_emb = pre_train_emb
		self.part_point = part_point
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb

		self._init_params()

	def _init_params(self):
		self.embedding = torch.nn.ModuleList()
		if (not self.fine_tune) and self.pre_train_emb:
			self.embedding.append(nn.Embedding(self.part_point, self.dim_emb))
			self.embedding.append(nn.Embedding.from_pretrained(torch.Tensor(self.pre_train_emb), freeze = True))
		elif self.fine_tune and self.pre_train_emb:
			init_embedding = 0.01 * np.random.randn(self.size_vocab, self.dim_emb).astype(np.float32)
			init_embedding[self.part_point: ] = self.pre_train_emb
			self.embedding.append(nn.Embedding.from_pretrained(torch.Tensor(init_embedding), freeze = False))
		else:
			self.embedding.append(nn.Embedding(self.size_vocab, self.dim_emb))

	def forward(self, inp):
		if (not self.fine_tune) and self.pre_train_emb:
			def get_emb(inp):
				mask = self.inp2mask(inp)

				inp_1 = inp * mask
				emb_1 = self.embedding[0](inp_1) * mask[:, :, None].float()
				inp_2 = (inp - self.part_point) * (1 - mask)
				emb_2 = self.embedding[1](inp_2) * (1 - mask)[:, :, None].float()
				emb = emb_1 + emb_2

				return emb

			emb_inp = get_emb(inp)
		else:
			emb_inp = self.embedding[0](inp)

		return emb_inp

	def inp2mask(self, inp):
		mask = (inp < self.part_point).long()
		return mask

class Position_Emb(nn.Module):
	def __init__(self, dim_emb):
		super(Position_Emb, self).__init__()

		self.dim_emb = dim_emb

		self._init_params()

	def _init_params(self):
		pass

	def forward(self, inp):
		pass

class Wemb(nn.Module):
	"""docstring for Wemb"""
	def __init__(self,
				 fine_tune,
				 pre_train_emb,
				 part_point,
				 size_vocab,
				 dim_emb,
				 position_emb,
				 dropout_rate):
		super(Wemb, self).__init__()
		
		self.fine_tune = fine_tune
		self.pre_train_emb = pre_train_emb
		self.part_point = part_point
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb
		self.position_emb = position_emb
		self.dropout_rate = dropout_rate

		self._init_params()

	def _init_params(self):
		self.wembs = torch.nn.ModuleList()
		self.wembs.append(Word_Emb(self.fine_tune, self.pre_train_emb, self.part_point, self.size_vocab, self.dim_emb))
		if self.position_emb:
			self.wembs.append(Position_Emb(self.dim_emb))

		self.layer_norm = LayerNorm(self.dim_emb)
		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, inp):
		def add_n(inps):
			rval = inps[0] * 0
			for inp in inps:
				rval += inp
			return rval
		
		emb_inps = []
		for wemb in self.wembs:
			emb_inps.append(wemb(inp))

		emb_inp = add_n(emb_inps)
		emb_inp = self.layer_norm(emb_inp)
		emb_inp = self.dropout(emb_inp)

		return emb_inp

class Multi_Head_Attention(nn.Module):
	def __init__(self,
				 dim_proj,
				 head_count,
				 dropout_rate):
		super(Multi_Head_Attention, self).__init__()

		self.dim_proj = dim_proj
		self.head_count = head_count
		self.dim_per_head = self.dim_proj // self.head_count
		self.dropout_rate = dropout_rate

		self._init_params()

	def _init_params(self):
		self.linear_key = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)
		self.linear_value = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)
		self.linear_query = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)

		self.dropout = nn.Dropout(self.dropout_rate)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, key, value, query, mask = None):
		# key: batch X key_len X hidden
		# value: batch X value_len X hidden
		# query: batch X query_len X hidden
		# mask: batch X query_len X key_len
		batch_size = key.size()[0]
		
		key_ = self.linear_key(key)
		value_ = self.linear_value(value)
		query_ = self.linear_query(query)

		key_ = key_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
		value_ = value_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
		query_ = query_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)

		attention_scores = torch.matmul(query_, key_.transpose(2, 3))
		attention_scores = attention_scores / math.sqrt(float(self.dim_per_head))

		if mask is not None:
			mask = mask.unsqueeze(1).expand_as(attention_scores)
			attention_scores = attention_scores.masked_fill(1 - mask, -1e18)

		attention_probs = self.softmax(attention_scores)
		attention_probs = self.dropout(attention_probs)

		context = torch.matmul(attention_probs, value_)
		context = context.transpose(1, 2).reshape(batch_size, -1, self.head_count * self.dim_per_head)

		return context

class TransformerEncoderBlock(nn.Module):
	def __init__(self,
				 dim_proj,
				 head_count,
				 dim_FNN,
				 act_fn,
				 dropout_rate):
		super(TransformerEncoderBlock, self).__init__()

		self.dim_proj = dim_proj
		self.head_count = head_count
		self.dim_FNN = dim_FNN
		self.act_fn = act_fn
		self.dropout_rate = dropout_rate

		self._init_params()

	def _init_params(self):
		self.multi_head_attention = Multi_Head_Attention(self.dim_proj, self.head_count, self.dropout_rate)
		self.linear_proj_context = MLP(self.dim_proj, self.dim_proj)
		self.layer_norm_context = LayerNorm(self.dim_proj)
		self.position_wise_fnn = MLP(self.dim_proj, self.dim_FNN)
		self.linear_proj_intermediate = MLP(self.dim_FNN, self.dim_proj)
		self.layer_norm_intermediate = LayerNorm(self.dim_proj)
		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, inp, mask):
		context = self.multi_head_attention(inp, inp, inp, mask = mask)
		context = self.linear_proj_context(context)
		context = self.dropout(context)
		res_inp = self.layer_norm_context(inp + context)

		rval = self.act_fn(self.position_wise_fnn(res_inp))
		rval = self.linear_proj_intermediate(rval)
		rval = self.dropout(rval)
		res_rval = self.layer_norm_intermediate(rval + res_inp)
		
		return res_rval

def get_activation(act_str):
	if act_str == "relu":
		return torch.nn.ReLU()
	elif act_str == "tanh":
		return torch.nn.Tanh()
	elif act_str == "sigmoid":
		return torch.nn.Sigmoid()

class TransformerEncoder(nn.Module):
	def __init__(self,
				 dim_proj,
				 head_count,
				 dim_FNN,
				 act_str,
				 num_layers,
				 dropout_rate):
		super(TransformerEncoder, self).__init__()

		self.dim_proj = dim_proj
		self.head_count = head_count
		self.dim_FNN = dim_FNN
		self.act_fn = get_activation(act_str)
		self.num_layers = num_layers
		self.dropout_rate = dropout_rate

		self._init_params()

	def _init_params(self):
		self.transformer = torch.nn.ModuleList([TransformerEncoderBlock(self.dim_proj, self.head_count, self.dim_FNN, self.act_fn, self.dropout_rate) for _ in range(self.num_layers)])

	def forward(self, inp, mask = None):
		rval = []
		pre_output = inp
		for i in range(self.num_layers):
			cur_output = self.transformer[i](pre_output, mask)
			rval.append(cur_output)
			pre_output = cur_output
		return pre_output, rval

def optimizer_wrapper(optimizer, lr, parameters):
	if optimizer == "adam":
		opt = torch.optim.Adam(params = parameters, lr = lr)
	return opt
