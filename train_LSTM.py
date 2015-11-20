import numpy as np

from layers import *
from utils import data
from climin import RmsProp, Adam

def extract_weights(model):
	weights = []
	for layer in model:
		w = layer.get_weights()
		if w is None:
			continue
		weights.append(w)
	W = np.concatenate(weights)
	return W


def load_weights(model, W):
	for layer in model:
		w = layer.get_weights()
		if w is None:
			continue
		w_shape = w.shape
		w, W = np.split(W, [np.prod(w_shape)])
		layer.set_weights(w.reshape(w_shape))
	return model


def extract_grads(model):
	grads = []
	for layer in model:
		g = layer.get_grads()
		if g is None:
			continue
		grads.append(g)
	dW = np.concatenate(grads)
	return dW


def forward(model, input):
	for layer in model:
		input = layer.forward(input)
	return input


def backward(model, gradient):
	for layer in reversed(model):
		gradient = layer.backward(gradient)


def fgrad(W):
	load_weights(model, W)
	input, target = gen()
	output = forward(model, input)
	loss = np.sum(target * np.log(output)) / (-1.0 * seq_len * batch_size)
	backward(model, target)
	dW = extract_grads(model)
	dW = np.clip(dW, -5.0, 5.0)
	print 'loss: ',  loss, '\t'
	return dW

seq_len = 50
batch_size = 60
gen = data('input.txt', seq_len, batch_size)
vocab_size = gen.vocab_size
states1, states2 = 128, 128
nb_inputs = vocab_size
nb_outputs = vocab_size
nb_iter = 2000

model = [LSTM(vocab_size, states1), LSTM(states1, states2),  Linear(states2, nb_outputs), Softmax()]

W = extract_weights(model)
print 'number of parameters: ', W.size

opt = RmsProp(W, fgrad, 2e-3, 0.8, 0.9)
for i in opt:
	print "iter: ", i["n_iter"]
	
	if i["n_iter"] > nb_iter:
		break

	model[0].forget()
	model[1].forget()

	if i["n_iter"] % 1 == 0:
		model[0].forget()
		model[1].forget()
		x = np.zeros((1, vocab_size, 1))
		x[0, 13, 0] = 1
		ixes = []
		for t in xrange(1000):
			p = forward(model, np.array(x).reshape(1, vocab_size, 1))
			ix = np.random.choice(range(vocab_size), p=p.ravel())
			x = np.zeros((vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		sample = ''.join(gen.decoder.to_c[ix] for ix in ixes)
		print '\n\n', sample, '\n\n'
		model[0].forget()
		model[1].forget()
