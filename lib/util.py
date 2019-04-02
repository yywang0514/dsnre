import sys
import re
import numpy as np
import cPickle as pkl
import codecs

import logging

from data_iterator import *

logger = logging.getLogger()
extra_token = ["<PAD>", "<UNK>"]

def display(msg):
	print(msg)
	logger.info(msg)

def datafold(filename):
	f = open(filename, 'r')
	data = []
	while 1:
		line = f.readline()
		if not line:
			break
		entities = map(int, line.split(' '))
		line = f.readline()
		bagLabel = line.split(' ')

		rel = map(int, bagLabel[0:-1])
		num = int(bagLabel[-1])
		positions = []
		sentences = []
		entitiesPos = []
		for i in range(0, num):
			sent = f.readline().split(' ')
			positions.append(map(int, sent[0:2]))
			epos = map(int, sent[0:2])
			epos.sort()
			entitiesPos.append(epos)
			sentences.append(map(int, sent[2:-1]))
		ins = InstanceBag(entities, rel, num, sentences, positions, entitiesPos)
		data += [ins]
	f.close()
	return data

def dicfold(textfile):
	vocab = []
	with codecs.open(textfile, "r", encoding = "utf8") as f:
		for line in f:
			line = line.strip()
			if line:
				vocab.append(line)
	return vocab

def build_word2idx(vocab, textFile):
	msg = "Building word2idx..."
	display(msg)

	pre_train_emb = []
	part_point = len(vocab)
	
	if textFile:
		word2emb = load_emb(vocab, textFile)

		pre_train_vocab = []
		un_pre_train_vocab = []

		for word in vocab:
			if word in word2emb:
				pre_train_vocab.append(word)
				pre_train_emb.append(word2emb[word])
			else:
				un_pre_train_vocab.append(word)
		
		part_point = len(un_pre_train_vocab)
		un_pre_train_vocab.extend(pre_train_vocab)
		vocab = un_pre_train_vocab

	word2idx = {}
	for v, k in enumerate(extra_token):
		word2idx[k] = v

	for v, k in enumerate(vocab):
		word2idx[k] = v + 2

	part_point += 2

	return word2idx, pre_train_emb, part_point

def load_emb(vocab, textFile):
	msg = 'load emb from ' + textFile
	display(msg)

	vocab_set = set(vocab)
	word2emb = {}

	emb_p = re.compile(r" |\t")
	count = 0
	with codecs.open(textFile, "r", "utf8") as filein:
		for line in filein:
			count += 1
			array = emb_p.split(line.strip())
			word = array[0]
			if word in vocab_set:
				vector = [float(array[i]) for i in range(1, len(array))]
				word2emb[word] = vector
	
	del vocab_set
	
	msg = "find %d words in %s" %(count, textFile)
	display(msg)

	msg = "Summary: %d words in the vocabulary and %d of them appear in the %s" %(len(vocab), len(word2emb), textFile)
	display(msg)

	return word2emb

def positive_evaluation(predict_results):
	predict_y = predict_results[0]
	predict_y_prob = predict_results[1]
	y_given = predict_results[2]

	positive_num = 0
	#find the number of positive examples
	for yi in range(y_given.shape[0]):
		if y_given[yi, 0] > 0:
			positive_num += 1
	# if positive_num == 0:
	#	  positive_num = 1
	# sort prob
	index = np.argsort(predict_y_prob)[::-1]

	all_pre = [0]
	all_rec = [0]
	p_n = 0
	p_p = 0
	n_p = 0
	# print y_given.shape[0]
	for i in range(y_given.shape[0]):
		labels = y_given[index[i],:] # key given labels
		py = predict_y[index[i]] # answer

		if labels[0] == 0:
			# NA bag
			if py > 0:
				n_p += 1
		else:
			# positive bag
			if py == 0:
				p_n += 1
			else:
				flag = False
				for j in range(y_given.shape[1]):
					if j == -1:
						break
					if py == labels[j]:
						flag = True # true positive
						break
					if flag:
						p_p += 1
		if (p_p+n_p) == 0:
			precision = 1
		else:
			precision = float(p_p)/(p_p+n_p)
		recall = float(p_p)/positive_num
		if precision != all_pre[-1] or recall != all_rec[-1]:
			all_pre.append(precision)
			all_rec.append(recall)
	return [all_pre[1:], all_rec[1:]]