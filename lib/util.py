import os
import sys
import numpy
import theano
import gzip
import cPickle
import codecs
import subprocess

def conll2003fold(textFile):
	train_set, valid_set, test_set, dic = cPickle.load(open(textFile, "rb"))
	return train_set, valid_set, test_set, dic
	
def get_model_size(params):
	total_size = 0
	for k, v in params.items():
		total_size += v.size
	return total_size

def get_log_cost(f_log_cost, data_iterator, options):
	probs = []
	while True:
		x_, y_, mask_ = data_iterator.next_batch(options.batch_size)
		if x_ is not None:
			probs.append(f_log_cost(x_, y_, mask_))
		else:
			break
	return numpy.array(probs).mean()

def conlleval(f_pred, data_iterator, idx2word, idx2label, pred_file, options):
	with codecs.open(pred_file, "w", encoding = "utf8") as f:
		while True:
			x_, y_, mask_ = data_iterator.next_batch(options.batch_size)
			if x_ is not None:
				p_ = f_pred(x_, mask_)
				sentences = []
				groundtruth = []
				predictions = []
				lens = mask_.sum(axis = 0)
				n_samples = x_.shape[1]
				
				for idx in range(n_samples):
					sentences.append([idx2word[token] for token in x_[:lens[idx], idx]])
					groundtruth.append([idx2label[token] for token in y_[:lens[idx], idx]])
					predictions.append([idx2label[token] for token in p_[:lens[idx], idx]])
				
				for s, g, p in zip(sentences, groundtruth, predictions):
					f.write("BOS O O\n")
					for w, gl, pl in zip(s, g, p):
						f.write("%s %s %s\n" %(w, gl, pl))
					f.write("EOS O O\n\n")
			else:
				break
		
	return evaluate(pred_file, options)
	
def evaluate(textFile, options):
	_conlleval = options.script_eval
	proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	stdout, _ = proc.communicate(open(textFile).read())
	
	for line in stdout.split('\n'):
		if 'accuracy' in line:
			out = line.split()
			break
	
	precision = float(out[6][:-2])
	recall = float(out[8][:-2])
	f1score = float(out[10])
	
	return precision, recall, f1score