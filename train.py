import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import argparse
import logging

from lib import *
from model import *

def train(options):
	if not os.path.exists(options.folder):
		os.mkdir(options.folder)

	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s")
	hdlr = logging.FileHandler(os.path.join(options.folder, options.file_log), mode = "w")
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)

	logger.info("python %s" %(" ".join(sys.argv)))

	#################################################################################
	start_time = time.time()

	msg = "Loading dicts from %s..." %(options.file_dic)
	display(msg)
	vocab = dicfold(options.file_dic)

	word2idx, pre_train_emb, part_point = build_word2idx(vocab, options.file_emb)

	msg = "Loading data from %s..." %(options.file_train)
	display(msg)
	train = datafold(options.file_train)

	msg = "Loading data from %s..." %(options.file_test)
	display(msg)
	test = datafold(options.file_test)

	end_time = time.time()

	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg)

	options.size_vocab = len(word2idx)

	if options.devFreq == -1:
		options.devFreq = (len(train) + options.batch_size - 1) // options.batch_size

	msg = "#inst in train: %d" %(len(train))
	display(msg)
	msg = "#inst in test %d" %(len(test))
	display(msg)
	msg = "#word vocab: %d" %(options.size_vocab)
	display(msg)

	msg = "=" * 30 + "Hyperparameter:" + "=" * 30
	display(msg)
	for attr, value in sorted(vars(options).items(), key = lambda x: x[0]):
		msg = "{}={}".format(attr.upper(), value)
		display(msg)

	#################################################################################
	msg = "=" * 30 + "model:" + "=" * 30
	display(msg)
	
	os.environ["CUDA_VISIBLE_DEVICES"] = options.gpus

	if options.seed is not None:
		torch.manual_seed(options.seed)
		np.random.seed(options.seed)

	model = Model(options.fine_tune,
				  pre_train_emb,
				  part_point,
				  options.size_vocab,
				  options.dim_emb,
				  options.dim_proj,
				  options.head_count,
				  options.dim_FNN,
				  options.act_str,
				  options.num_layer,
				  options.num_class,
				  options.dropout_rate).cuda()

	if os.path.exists("{}.pt".format(options.reload_model)):
		model.load_state_dict(torch.load("{}.pt".format(options.reload_model)))

	parameters = filter(lambda param: param.requires_grad, model.parameters())
	optimizer = optimizer_wrapper(options.optimizer, options.lr, parameters)

	msg = "\n{}".format(model)
	display(msg)
	
	#################################################################################
	checkpoint_dir = os.path.join(options.folder, "checkpoints")
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)
	best_path = os.path.join(checkpoint_dir, options.saveto)

	#################################################################################
	msg = "=" * 30 + "Optimizing:" + "=" * 30
	display(msg)

	[train_rels, train_nums, train_sents, train_poss, train_eposs] = bags_decompose(train)
	[test_rels, test_nums, test_sents, test_poss, test_eposs] = bags_decompose(test)



	# batch_index = [0, 1, 2]
	# batch_rels = [train_rels[m][0] for m in batch_index]
	# batch_nums = [train_nums[m] for m in batch_index]
	# batch_sents = [train_sents[m] for m in batch_index]
	# batch_poss = [train_poss[m] for m in batch_index]
	# batch_eposs = [train_eposs[m] for m in batch_index]

	# batch_data = select_instance(batch_rels,
	# 							 batch_nums,
	# 							 batch_sents,
	# 							 batch_poss,
	# 							 batch_eposs,
	# 							 model)
	# for sent in batch_data[0]:
	# 	print(sent)
	# print(batch_data[1])
	# print(batch_data[2])
	# print(batch_data[3])

	train_idx_list = np.arange(len(train))
	steps_per_epoch = (len(train) + options.batch_size - 1) // options.batch_size
	n_updates = 0
	for e in range(options.nepochs):
		np.random.shuffle(train_idx_list)
		for step in range(steps_per_epoch):
			batch_index = train_idx_list[step * options.batch_size: (step + 1) * options.batch_size]
			batch_rels = [train_rels[m][0] for m in batch_index]
			batch_nums = [train_nums[m] for m in batch_index]
			batch_sents = [train_sents[m] for m in batch_index]
			batch_poss = [train_poss[m] for m in batch_index]
			batch_eposs = [train_eposs[m] for m in batch_index]
			batch_data = select_instance(batch_rels,
										 batch_nums,
										 batch_sents,
										 batch_poss,
										 batch_eposs,
										 model)

			disp_start = time.time()

			model.train()

			n_updates += 1

			optimizer.zero_grad()
			logit = model(batch_data[0], batch_data[1], batch_data[2])
			loss = F.cross_entropy(logit, batch_data[3])
			loss.backward()

			if options.clip_c != 0:
				total_norm = torch.nn.utils.clip_grad_norm_(parameters, options.clip_c)

			optimizer.step()

			disp_end = time.time()

			if np.mod(n_updates, options.dispFreq) == 0:
				msg = "Epoch: %d, Step: %d, Loss: %f, Time: %.2f sec" %(e, n_updates, loss.cpu().item(), disp_end - disp_start)
				display(msg)

			if np.mod(n_updates, options.devFreq) == 0:
				msg = "=" * 30 + "Evaluating" + "=" * 30
				display(msg)
				
				model.eval()

				test_predict = predict(test_rels, test_nums, test_sents, test_poss, test_eposs, model)
				test_pr = positive_evaluation(test_predict)

				msg = 'test set PR = [' + str(test_pr[0][-1]) + ' ' + str(test_pr[1][-1]) + ']'
				display(msg)

				msg = "Saving model..."
				display(msg)
				torch.save(model.state_dict(), "{}_step_{}.pt".format(best_path, n_updates))
				msg = "Model checkpoint has been saved to {}_step_{}.pt".format(best_path, n_updates)
				display(msg)

	end_time = time.time()
	msg = "Optimizing time: %f seconds" %(end_time - start_time)
	display(msg)

def predict(rels, nums, sents, poss, eposs, model):
	numBags = len(rels)
	predict_y = np.zeros((numBags), dtype=np.int32)
	predict_y_prob = np.zeros((numBags), dtype=np.float32)
	y = np.asarray(rels, dtype='int32')
	for bagIndex, insRel in enumerate(rels):
		insNum = nums[bagIndex]
		maxP = -1
		pred_rel_type = 0
		max_pos_p = -1
		positive_flag = False
		for m in range(insNum):
			insX = sents[bagIndex][m]
			epos = eposs[bagIndex][m]
			sel_x, sel_len, sel_epos = prepare_data([insX], [epos])
			results = model(sel_x, sel_len, sel_epos)
			rel_type = results.argmax()
			if positive_flag and rel_type == 0:
				continue
			else:
				# at least one instance is positive
				tmpMax = results.max()
				if rel_type > 0:
					positive_flag = True
					if tmpMax > max_pos_p:
						max_pos_p = tmpMax
						pred_rel_type = rel_type
				else:
					if tmpMax > maxP:
						maxP = tmpMax
		if positive_flag:
			predict_y_prob[bagIndex] = max_pos_p
		else:
			predict_y_prob[bagIndex] = maxP

		predict_y[bagIndex] = pred_rel_type
	return [predict_y, predict_y_prob, y]

def main(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument("--folder", help = "the dir of model", default = "workshop")
	parser.add_argument("--file_dic", help = "the file of vocabulary", default = "./data/50/dict.txt")
	parser.add_argument("--file_train", help = "the file of training data", default = "./data/gap_40_len_80/train_filtered.data")
	parser.add_argument("--file_test", help = "the file of testing data", default = "./data/gap_40_len_80/test_filtered.data")
	# parser.add_argument("--file_emb", help = "the file of embedding", default = "./data/50/dict_emb.txt")
	parser.add_argument("--file_emb", help = "the file of embedding", default = "")
	parser.add_argument("--file_log", help = "the log file", default = "train.log")
	parser.add_argument("--reload_model", help = "the pretrained model", default = "")
	parser.add_argument("--saveto", help = "the file to save the parameter", default = "model")

	parser.add_argument("--seed", help = "the random seed", default = 1234, type = int)
	parser.add_argument("--size_vocab", help = "the size of vocabulary", default = 10000, type = int)
	parser.add_argument("--dim_emb", help = "the dimension of the word embedding", default = 256, type = int)
	parser.add_argument("--dim_proj", help = "the dimension of the hidden state", default = 256, type = int)
	parser.add_argument("--head_count", help = "the num of head in multi head attention", default = 8, type = int)
	parser.add_argument("--dim_FNN", help = "the dimension of the positionwise FNN", default = 256, type = int)
	parser.add_argument("--act_str", help = "the activation function of the positionwise FNN", default = "relu")
	parser.add_argument("--num_layer", help = "the num of layers", default = 6, type = int)
	parser.add_argument("--num_class", help = "the number of labels", default = 27, type = int)
	parser.add_argument("--position_emb", help = "if true, the position embedding will be used", default = False, action = "store_true")
	parser.add_argument("--fine_tune", help = "if true, the pretrained embedding will be fine tuned", default = False, action = "store_true")

	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adam")
	parser.add_argument("--lr", help = "learning rate", default = 0.0004, type = float)
	parser.add_argument("--dropout_rate", help = "dropout rate", default = 0.5, type = float)
	parser.add_argument("--clip_c", help = "grad clip", default = 10.0, type = float)
	parser.add_argument("--nepochs", help = "the max epoch", default = 30, type = int)
	parser.add_argument("--batch_size", help = "batch size", default = 32, type = int)
	parser.add_argument("--dispFreq", help = "the frequence of display", default = 100, type = int)
	parser.add_argument("--devFreq", help = "the frequence of evaluation", default = -1, type = int)
	parser.add_argument("--wait_N", help = "use to early stop", default = 1, type = int)
	parser.add_argument("--patience", help = "use to early stop", default = 7, type = int)
	parser.add_argument("--maxlen", help = "max length of sentence", default = 100, type = int)
	parser.add_argument("--gpus", help = "specify the GPU IDs", default = "0")

	options = parser.parse_args(argv)
	train(options)

if "__main__" == __name__:
	main(sys.argv[1:])
