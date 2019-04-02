import sys
import codecs

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

def change_word_idx(data):
	new_data = []
	for inst in data:
		entities = inst.entities
		rel = inst.rel
		num = inst.num
		sentences = inst.sentences
		positions = inst.positions
		entitiesPos = inst.entitiesPos
		new_sentences = []
		for sent in sentences:
			new_sent = []
			for word in sent:
				if word == 160696:
					new_sent.append(1)
				elif word == 0:
					new_sent.append(0)
				else:
					new_sent.append(word + 1)
			new_sentences.append(new_sent)
		new_inst = InstanceBag(entities, rel, num, new_sentences, positions, entitiesPos)
		new_data.append(new_inst)
	return new_data

def save_data(data, textfile):
	with codecs.open(textfile, "w", encoding = "utf8") as f:
		for inst in data:
			f.write("%s\n" %(" ".join(map(str, inst.entities))))
			f.write("%s %s\n" %(" ".join(map(str, inst.rel)), str(inst.num)))
			for pos, sent in zip(inst.positions, inst.sentences):
				f.write("%s %s\n" %(" ".join(map(str, pos)), " ".join(map(str, sent))))

def main(argv):
	data = datafold(argv[0])
	new_data = change_word_idx(data)
	save_data(new_data, argv[1])

if "__main__" == __name__:
	main(sys.argv[1:])