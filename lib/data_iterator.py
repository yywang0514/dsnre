import time
import cPickle
import numpy as np
import torch

class InstanceBag(object):
	def __init__(self, entities, rel, num, sentences, positions, entitiesPos):
		self.entities = entities
		self.rel = rel
		self.num = num
		self.sentences = sentences
		self.positions = positions
		self.entitiesPos = entitiesPos

def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_pos = [data_bag.positions for data_bag in data_bags]
    bag_num = [data_bag.num for data_bag in data_bags]
    bag_rel = [data_bag.rel for data_bag in data_bags]
    bag_epos = [data_bag.entitiesPos for data_bag in data_bags]
    return [bag_rel, bag_num, bag_sent, bag_pos, bag_epos]

def select_instance(rels, nums, sents, poss, eposs, model):
    batch_x = []
    batch_len = []
    batch_epos = []
    batch_y = []
    for bagIndex, insNum in enumerate(nums):
        maxIns = 0
        maxP = -1
        if insNum > 1:
            for m in range(insNum):
            	insX = sents[bagIndex][m]
                epos = eposs[bagIndex][m]
                sel_x, sel_len, sel_epos = prepare_data([insX], [epos])
                results = model(sel_x, sel_len, sel_epos)
                tmpMax = results.max()
                if tmpMax > maxP:
                    maxIns = m
                    maxP=tmpMax

        batch_x.append(sents[bagIndex][maxIns])
        batch_epos.append(eposs[bagIndex][maxIns])
        batch_y.append(rels[bagIndex])
    
    batch_x, batch_len, batch_epos = prepare_data(batch_x, batch_epos)
    batch_y = torch.LongTensor(np.array(batch_y).astype("int32")).cuda()

    return [batch_x, batch_len, batch_epos, batch_y]

def prepare_data(sents, epos):
    lens = [len(sent) for sent in sents]

    n_samples = len(lens)
    max_len = max(lens)

    batch_x = np.zeros((n_samples, max_len)).astype("int32")
    for idx, s in enumerate(sents):
        batch_x[idx, :lens[idx]] = s

    batch_len = np.array(lens).astype("int32")
    batch_epos = np.array(epos).astype("int32")

    return torch.LongTensor(batch_x).cuda(), torch.LongTensor(batch_len).cuda(), torch.LongTensor(batch_epos).cuda()
