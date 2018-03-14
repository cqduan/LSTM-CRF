from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.path.append(".")
from lib import *

optimizer = {"adam": adam, "adadelta": adadelta}

def zipp(params, tparams):
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)

def unzip(zipped):
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params

def load_params(path, params):
	pp = numpy.load(path)
	for kk, vv in params.iteritems():
		if kk not in pp:
			raise Warning("%s is not in the archive" % kk)
		params[kk] = pp[kk]

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams
	
def build_model(options):
	trng = RandomStreams(options.seed)
	
	# Used for dropout.
	use_noise = theano.shared(numpy_floatX(0.))

	params = OrderedDict()
	
	emb_layer = EMB_layer()
	emb_layer.init_params(options, params)
	
	encoder_lstm_fw = LSTM_layer(prefix = "encoder_lstm_fw", dim_in = options.dim_emb, dim_out = options.dim_proj)
	encoder_lstm_fw.init_params(options, params)
	
	encoder_lstm_bw = LSTM_layer(prefix = "encoder_lstm_bw", dim_in = options.dim_emb, dim_out = options.dim_proj)
	encoder_lstm_bw.init_params(options, params)
	
	encoder_mlp = MLP_layer(prefix = "encoder_mlp", dim_in = options.dim_proj * 2, dim_out = options.num_class)
	encoder_mlp.init_params(options, params)
	
	decoder_crf = CRF_layer(prefix = "decoder_crf", dim = options.num_class)
	decoder_crf.init_params(options, params)
	
	if options.reload_model:
		load_params(os.path.join(options.folder, options.reload_model), params)
	
	tparams = init_tparams(params)
	
	x, emb_x, y, mask = emb_layer.build(tparams, options)
	
	emb_x_dp = dropout_layer(emb_x, use_noise, trng, options)
	
	proj_fw_x = encoder_lstm_fw.build(tparams, emb_x_dp, options, mask)
	proj_bw_x = reverse(encoder_lstm_bw.build(tparams, reverse(emb_x_dp), options, reverse(mask)))
	
	proj_x = tensor.concatenate([proj_fw_x, proj_bw_x], axis = -1)
	
	fusion_x = encoder_mlp.build(tparams, proj_x, options)
	
	pred_prob, pred, log_cost = decoder_crf.build(tparams, fusion_x, y, options, mask)
	
	cost = log_cost
	if options.decay_c > 0.:
		decay_c = theano.shared(numpy.float32(options.decay_c), name='decay_c')
		weight_decay = 0.
		for kk, vv in tparams.iteritems():
			weight_decay += (vv ** 2).sum()
		weight_decay *= decay_c
		cost += weight_decay
	
	grads = tensor.grad(cost, wrt = tparams.values())
	g2 = 0.
	for g in grads:
		g2 += (g ** 2).sum()
	grad_norm = tensor.sqrt(g2)
	
	if options.clip_c > 0.:
		new_grads = []
		for g in grads:
			new_grads.append(tensor.switch(g2 > options.clip_c ** 2, g * options.clip_c / tensor.sqrt(g2), g))
		grads = new_grads
		
	return params, tparams, use_noise, x, emb_x, y, mask, pred_prob, pred, log_cost, cost, grad_norm, grads

def build_optimizer(lr, tparams, x, y, mask, pred_prob, pred, log_cost, cost, grad_norm, grads, options):
	f_grad_shared, f_update = optimizer[options.optimizer](lr, tparams, [x, y, mask], cost, grads)
	f_log_cost = theano.function(inputs = [x, y, mask], outputs = log_cost, name = "f_log_cost")
	f_grad_norm = theano.function(inputs = [x, y, mask], outputs = grad_norm, name = "f_grad_norm")
	f_pred_prob = theano.function(inputs = [x, mask], outputs = pred_prob, name = "f_pred_prob")
	f_pred = theano.function(inputs = [x, mask], outputs = pred, name = "f_pred")
	return f_grad_shared, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred
