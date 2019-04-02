import torch
import torch.nn as nn

from lib import *

class Model(nn.Module):
	def __init__(self,
				 fine_tune,
				 pre_train_emb,
				 part_point,
				 size_vocab,
				 dim_emb,                                                                                                                                                                                   
				 dim_proj,
				 head_count,  
				 dim_FNN, 
				 act_str,
				 num_layer,
				 num_class,
				 dropout_rate):
		super(Model, self).__init__()

		self.fine_tune = fine_tune
		self.pre_train_emb = pre_train_emb
		self.part_point = part_point
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb
		self.dim_proj = dim_proj
		self.head_count = head_count
		self.dim_FNN = dim_FNN
		self.act_str = act_str
		self.num_layer = num_layer
		self.num_class = num_class
		self.dropout_rate = dropout_rate

		self._init_params()

	def _init_params(self):
		self.wemb = Word_Emb(self.fine_tune,
							 self.pre_train_emb,
							 self.part_point,
							 self.size_vocab,   
							 self.dim_emb)                                       

		self.encoder = TransformerEncoder(self.dim_proj,
										  self.head_count,
										  self.dim_FNN,
										  self.act_str,
										  self.num_layer,
									       	  self.dropout_rate)

		self.dense = MLP(self.dim_proj * 3, self.dim_proj)
		self.relu = torch.nn.ReLU()
		self.classifier = MLP(self.dim_proj, self.num_class)
		self.dropout = nn.Dropout(self.dropout_rate)

	def forward(self, inp, lengths, epos):                                      
		mask, mask_l, mask_m, mask_r = self.pos2mask(epos, lengths)

		emb_inp = self.wemb(inp)
		emb_inp = self.dropout(emb_inp)

		proj_inp, _ = self.encoder(emb_inp, self.create_attention_mask(mask, mask))
		proj_inp = proj_inp * mask[:, :, None]

		pool_inp_l = torch.sum(proj_inp * mask_l[:, :, None], dim = 1) / torch.sum(mask_l, dim = 1)[:, None]
		pool_inp_m = torch.sum(proj_inp * mask_m[:, :, None], dim = 1) / torch.sum(mask_m, dim = 1)[:, None]
		pool_inp_r = torch.sum(proj_inp * mask_r[:, :, None], dim = 1) / torch.sum(mask_r, dim = 1)[:, None]

		pool_inp = torch.cat([pool_inp_l, pool_inp_m, pool_inp_r], dim = 1)

		pool_inp = self.dropout(pool_inp)

		logit = self.relu(self.dense(pool_inp))  

		logit = self.dropout(logit)

		logit = self.classifier(logit)

		return logit

	def pos2mask(self, epos, lengths):
		mask = self.len2mask(lengths)

		nsample = lengths.size()[0]
		max_len = torch.max(lengths)
		idxes = torch.arange(0, max_len).cuda() 
		mask_l = (idxes < epos[:, 0].unsqueeze(1)).float()
		mask_r = mask - (idxes < epos[:, 1].unsqueeze(1)).float()
		mask_m = torch.ones([nsample, max_len]).float().cuda() - mask_l - mask_r
		return mask, mask_l, mask_m, mask_r

	def len2mask(self, lengths):
		max_len = torch.max(lengths)
		idxes = torch.arange(0, max_len).cuda()
		mask = (idxes < lengths.unsqueeze(1)).float()
		return mask

	def create_attention_mask(self, query_mask, key_mask):
		return torch.matmul(query_mask[:, :, None], key_mask[:, None, :]).byte()