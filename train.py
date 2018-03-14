import os
import sys
import time
import platform
import cPickle as pkl
import random
import copy
import argparse
import logging

import subprocess

import numpy

from theano import config

sys.path.append(".")
from lib import *
from build import *

sys.setrecursionlimit(1500)

def display(msg, logger):
	print msg
	logger.info(msg)
	
def train(options):
	if not options.folder:
		options.folder = "workshop"
	if not os.path.exists(options.folder):
		os.mkdir(options.folder)
	
	logging.basicConfig(level = logging.DEBUG,
						format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s",
						filename = os.path.join(options.folder, options.file_log),
						filemode = "w")
	logger = logging.getLogger()

	logger.info("python %s" %(" ".join(sys.argv)))
	
	#################################################################################
	start_time = time.time()
	
	msg = "Loading data..."
	display(msg, logger)
	
	train_set, valid_set, test_set, dic = conll2003fold(options.file_data)
	
	end_time = time.time()
	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg, logger)
	
	train_x, train_y = train_set
	valid_x, valid_y = valid_set
	test_x, test_y  = test_set
	
	options.dic = dic["word2idx"]
	options.size_vocab = len(dic["word2idx"])
	options.num_class = len(dic["label2idx"])
	
	idx2word = dic["idx2word"]
	idx2label = dic["idx2label"]
	
	msg = "\n#inst in train: %d\n"	\
		  "#inst in dev %d\n"		\
		  "#inst in test %d\n"		\
		  "#vocab: %d\n"			\
		  "#label: %d" %(len(train_x), len(valid_x), len(test_x), options.size_vocab, options.num_class)
	display(msg, logger)
	
	#################################################################################
	start_time = time.time()
	msg = 'Building model...'
	display(msg, logger)
	
	numpy.random.seed(options.seed)
	random.seed(options.seed)
	
	params, tparams, use_noise, x, emb_x, y, mask, pred_prob, pred, log_cost, cost, grad_norm, grads = build_model(options)
	
	lr = tensor.scalar(dtype=config.floatX)
	
	f_grad_shared, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred = build_optimizer(lr, tparams, x, y, mask, pred_prob, pred, log_cost, cost, grad_norm, grads, options)
	
	end_time = time.time()
	msg = "#Params: %d, Building time: %f seconds" %(get_model_size(params), end_time - start_time)
	display(msg, logger)
	
	#################################################################################
	msg = 'Optimization...'
	display(msg, logger)
	
	if options.validFreq == -1:
		options.validFreq = (len(train_x) + options.batch_size - 1) / options.batch_size
	
	if options.saveFreq == -1:
		options.saveFreq = (len(train_x) + options.batch_size - 1) / options.batch_size

	train_data_iterator = DataIterator([train_x, train_y])
	train_data_iterator_beta = DataIterator([train_x, train_y])
	dev_data_iterator = DataIterator([valid_x, valid_y])
	test_data_iterator = DataIterator([test_x, test_y])
	
	estop = False
	history_f = []
	valid_record = []
	test_record = []
	best_p = None
	
	bad_counter = 0
	wait_counter = 0
	wait_N = 1
	lr_change_list = []
	
	n_updates = 0
	best_epoch_num = 0
	
	start_time = time.time()
	
	while True:
		x_, y_, mask_ = train_data_iterator.next_batch(options.batch_size, max_epoch = options.nepochs, is_train = True, shuffle = True)
		
		if x_ is None:
			break
		
		n_updates += 1
		
		use_noise.set_value(1.)
		
		disp_start = time.time()
		
		cost = f_grad_shared(x_, y_, mask_)
		f_update(options.lr)
		
		disp_end = time.time()
		
		if numpy.isnan(cost) or numpy.isinf(cost):
			msg = "NaN detected"
			display(msg, logger)
		
		if numpy.mod(n_updates, options.dispFreq) == 0:
			msg = "Epoch: %d, Update: %d, Cost: %f, Grad: %f, Time: %.2f sec" %(train_data_iterator.epoch, n_updates, cost, f_grad_norm(x_, y_, mask_), (disp_end-disp_start))
			display(msg, logger)
		
		if numpy.mod(n_updates, options.saveFreq) == 0:
			msg = "Saving..."
			display(msg, logger)
			if best_p is not None:
				params = best_p
			else:
				params = unzip(tparams)
			
			numpy.savez(os.path.join(options.folder, options.saveto), **params)
			pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
			msg = "Done"
			display(msg, logger)
		
		if numpy.mod(n_updates, options.validFreq) == 0:
			use_noise.set_value(0.)
			
			dev_cost = get_log_cost(f_log_cost, dev_data_iterator, options)
			dev_data_iterator.reset()
			dev_P, dev_R, dev_F = conlleval(f_pred, dev_data_iterator, idx2word, idx2label, os.path.join(options.folder, options.file_tmp_dev), options)
			dev_data_iterator.reset()
			history_f.append(dev_F)
			tst_cost = get_log_cost(f_log_cost, test_data_iterator, options)
			test_data_iterator.reset()
			tst_P, tst_R, tst_F = conlleval(f_pred, test_data_iterator, idx2word, idx2label, os.path.join(options.folder, options.file_tmp_test), options)
			test_data_iterator.reset()
			
			msg = "\nValid cost: %f\n"	\
				  "Valid: Precision %f, Recall %f, F1 %f\n"	\
				  "Test cost: %f\n"		\
				  "Test: Precision %f, Recall %f, F1 %f\n" 	\
				  "lrate: %f" %(dev_cost, dev_P, dev_R, dev_F, tst_cost, tst_P, tst_R, tst_F, options.lr)
			display(msg, logger)
			
			valid_record.append((dev_P, dev_R, dev_F))
			test_record.append((tst_P, tst_R, tst_F))
			
			if best_p == None or dev_F > numpy.array(history_f).max():
				best_p = unzip(tparams)
				best_epoch_num = train_data_iterator.epoch
				wait_counter = 0
				
			if dev_F <= numpy.array(history_f).max():
				wait_counter += 1
				
			if wait_counter >= wait_N:
				msg = "wait_counter max, need to half the lr"
				display(msg, logger)
				bad_counter += 1
				wait_counter = 0
				msg = "bad_counter: %d" %bad_counter
				display(msg, logger)
				options.lr *= 0.5
				lr_change_list.append(train_data_iterator.epoch)
				msg = "lrate change to: %f" %(options.lr)
				display(msg, logger)
				zipp(best_p, tparams)
			
			if bad_counter > options.patience:
				msg = "Early Stop!"
				display(msg, logger)
				estop = True
				break
		
		if estop:
			break
	
	end_time = time.time()
	msg = "Optimizing time: %f seconds" %(end_time - start_time)
	display(msg, logger)
	
	def data_split(metrics):
		P, R, F = [], [], []
		for m in metrics:
			P.append(m[0])
			R.append(m[1])
			F.append(m[2])
		
		out_val = "P: %s\nR: %s\nF: %s\n" %(",".join(map(str, P)), ",".join(map(str, R)), ",".join(map(str, F)))
		return out_val
		
	
	with open(os.path.join(options.folder, 'record.csv'), 'w') as f:
		f.write(str(best_epoch_num) + '\n')
		f.write(','.join(map(str,lr_change_list)) + '\n')
		f.write(data_split(valid_record))
		f.write(data_split(test_record))
	
	if best_p is not None:
		zipp(best_p, tparams)
		
	use_noise.set_value(0.)
	
	msg = "\n" + "=" * 80 + "\nFinal Result\n" + "=" * 80
	display(msg, logger)
		
	tra_cost = get_log_cost(f_log_cost, train_data_iterator_beta, options)
	train_data_iterator_beta.reset()
	tra_P, tra_R, tra_F = conlleval(f_pred, train_data_iterator_beta, idx2word, idx2label, os.path.join(options.folder, options.file_best_train), options)
	train_data_iterator_beta.reset()
	dev_cost = get_log_cost(f_log_cost, dev_data_iterator, options)
	dev_data_iterator.reset()
	dev_P, dev_R, dev_F = conlleval(f_pred, dev_data_iterator, idx2word, idx2label, os.path.join(options.folder, options.file_best_dev), options)
	dev_data_iterator.reset()
	tst_cost = get_log_cost(f_log_cost, test_data_iterator, options)
	test_data_iterator.reset()
	tst_P, tst_R, tst_F = conlleval(f_pred, test_data_iterator, idx2word, idx2label, os.path.join(options.folder, options.file_best_test), options)
	test_data_iterator.reset()
	msg = "\nTrain cost: %f\n"			\
		  "Train: P %f, R %f, F %f\n"	\
		  "Valid cost: %f\n"			\
		  "Valid: P %f, R %f, F %f\n"	\
		  "Test cost: %f\n"				\
		  "Test: P %f, R %f, F %f\n" 	\
		  "best epoch: %d" %(tra_cost, tra_P, tra_R, tra_F, dev_cost, dev_P, dev_R, dev_F, tst_cost, tst_P, tst_R, tst_F, best_epoch_num)
	display(msg, logger)
	if best_p is not None:
		params = best_p
	else:
		params = unzip(tparams)
	numpy.savez(os.path.join(options.folder, options.saveto), **params)
	pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
	msg = "Finished"
	display(msg, logger)
	
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", help = "the dir of model", default = "")
	parser.add_argument("--file_data", help = "the file of data", default = "../data/conll2003_eng.pkl")
	parser.add_argument("--file_tmp_dev", help = "the file of data", default = "current_dev_result")
	parser.add_argument("--file_tmp_test", help = "the file of data", default = "current_test_result")
	parser.add_argument("--file_best_train", help = "the file of data", default = "train_result")
	parser.add_argument("--file_best_dev", help = "the file of data", default = "dev_result")
	parser.add_argument("--file_best_test", help = "the file of data", default = "test_result")
	parser.add_argument("--script_eval", help = "conlleval.pl", default = "conlleval.pl")
	parser.add_argument("--file_emb", help = "the file of embedding", default = "")
	parser.add_argument("--file_log", help = "the log file", default = "train.log")
	parser.add_argument("--reload_model", help = "the pretrained model", default = "")
	parser.add_argument("--saveto", help = "the file to save the parameter", default = "model")
	parser.add_argument("--dic", help = "word2idx", default = None, type = object)
	
	parser.add_argument("--size_vocab", help = "the size of vocabulary", default = 10000, type = int)
	
	parser.add_argument("--dim_emb", help = "the dimension of the word embedding", default = 200, type = int)
	parser.add_argument("--dim_proj", help = "the dimension of the lstm layers", default = 200, type = int)
	parser.add_argument("--num_class", help = "the dimension of the MLP layer", default = 3, type = int)
	
	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adadelta")
	parser.add_argument("--batch_size", help = "batch size", default = 16, type = int)
	parser.add_argument("--maxlen", help = "max length of sentence", default = 100, type = int)
	parser.add_argument("--seed", help = "the seed of random", default = 345, type = int)
	parser.add_argument("--dispFreq", help = "the frequence of display", default = 10, type = int)
	parser.add_argument("--validFreq", help = "the frequence of valid", default = -1, type = int)
	parser.add_argument("--saveFreq", help = "the frequence of saving", default = -1, type = int)
	parser.add_argument("--nepochs", help = "the max epoch", default = 5000, type = int)
	parser.add_argument("--lr", help = "the initial learning rate", default = 0.0970806646812754, type = float)
	parser.add_argument("--dropout_rate", help = "keep rate", default = 0.8, type = float)
	parser.add_argument("--patience", help = "used to early stop", default = 10, type = int)
	parser.add_argument("--decay", help = "the flag to indicate whether to decay the learning rate", action = "store_false", default = True)
	parser.add_argument("--decay_c", help = "decay rate", default = 0, type = float)
	parser.add_argument("--clip_c", help = "grad clip", default = 10, type = float)
	parser.add_argument("--debug", help = "mode flag", action = "store_false", default = True)
	
	options = parser.parse_args(argv)
	train(options)
	
if "__main__" == __name__:
	main(sys.argv[1:])
