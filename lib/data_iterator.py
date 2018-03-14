import numpy as np
import cPickle as pkl

from util import *

class DataIterator():
	def __init__(self, data, num_fragment = 100):
		self.data = data
		self.total = len(data[0])
		self.num_fragment = num_fragment
		self.size = self.total / num_fragment
		self.idx_list = np.arange(self.total)
		self.cursor = 0
		self.epoch = 0
		
	def shuffle(self):
		"""
		lens = [len(sent) for sent in self.data[0]]
		idx2lens = {}
		for k, v in enumerate(lens):
			idx2lens[k] = v
		idx_list = np.array([k for k, v in sorted(idx2lens.iteritems(), key = lambda x: x[1])])
		
		fragments = []
		for i in range(self.num_fragment - 1):
			fragment = np.array(idx_list[i * self.size: (i + 1) * self.size])
			np.random.shuffle(fragment)
			fragments.append(fragment)
		fragment = np.array(idx_list[(self.num_fragment - 1) * self.size:])
		np.random.shuffle(fragment)
		fragments.append(fragment)
		
		self.idx_list = np.concatenate(fragments)
		"""
		np.random.shuffle(self.idx_list)
		
	def next_batch(self, batch_size, max_len = None, max_epoch = 1, is_train = False, shuffle = False):
		if self.cursor >= self.total:
			if is_train:
				self.epoch += 1
				if self.epoch >= max_epoch:
					return None, None, None
				if shuffle:
					self.shuffle()
				batch_index = self.idx_list[: batch_size] 
				self.cursor = batch_size
			else:
				batch_index = []
		else:
			if self.cursor == 0:
				if shuffle:
					self.shuffle()
				batch_index = self.idx_list[: batch_size]
				self.cursor = batch_size
			else:
				batch_index = self.idx_list[self.cursor: self.cursor + batch_size]
				self.cursor += batch_size
		
		if len(batch_index) < 1:
			return None, None, None
		else:
			x = [self.data[0][idx] for idx in batch_index]
			y = [self.data[1][idx] for idx in batch_index]
			
			x_, y_, mask_ = self.prepare_data(x, y, max_len)
			
			return x_, y_, mask_
	
	def prepare_data(self, seqs, labs, maxlen = None):
		lengths = [len(s) for s in seqs]
				
		if maxlen:
			new_seqs = []
			new_labs = []
			new_lengths = []
			for length, seq, lab in zip(lengths, seqs, labs):
				if length < maxlen:
					new_seqs.append(seq)
					new_labs.append(lab)
					new_lengths.append(length)
				else:
					new_seqs.append(seq[: maxlen])
					new_labs.append(lab[: maxlen])
					new_lengths.append(maxlen)
			seqs = new_seqs
			labs = new_labs
			lengths = new_lengths
			
		if len(lengths) < 1:
			return None, None, None
		
		n_samples = len(seqs)
		max_len = max(lengths) + 1
		
		x_ = np.zeros((max_len, n_samples)).astype("int64")
		y_ = np.zeros((max_len, n_samples)).astype("int64")
		mask_ = np.zeros((max_len, n_samples)).astype(theano.config.floatX)
		for idx, (s, l) in enumerate(zip(seqs, labs)):
			x_[:lengths[idx], idx] = s
			y_[:lengths[idx], idx] = l
			mask_[:lengths[idx], idx] = 1.
		
		return x_, y_, mask_
	
	def reset(self):
		self.cursor = 0
		self.epoch = 0
