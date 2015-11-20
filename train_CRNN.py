import numpy as np

from layers import *
from utils import data
from climin import RmsProp, Adam

import time
import matplotlib.pyplot as plt
plt.ion()

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
	losses.append(loss * 0.01 + losses[-1] * 0.99)
	backward(model, target)
	dW = extract_grads(model)
	dW = np.clip(dW, -10.0, 10.0)
	return dW

seq_len = 10
batch_size = 10
gen = data('input.txt', seq_len, batch_size)
vocab_size = gen.vocab_size


states1 = 16 
states2 = 64
nb_inputs = vocab_size
nb_outputs = vocab_size

lr = 1e-3
nb_iter = 10

# dont forget reverse
all_clocks = [[1, 2, 4, 16, 64, 128]]

for clocks in all_clocks:
	print clocks, '\t',

	losses = [5.32]

	model = [CRNN(nb_inputs, states1, states2, clocks), Linear(states2, nb_outputs), Softmax()]

	W = extract_weights(model)
	print 'number of parameters: ', W.size - model[0].Wh.size / 2

	opt = RmsProp(W, fgrad, lr)

	tick = time.time()
	for i in opt:
		if i["n_iter"] > nb_iter:
			break

		if i["n_iter"] % 100 == 0:
			print "iter: ", i["n_iter"]
			model[0].forget()

		'''
		if i["n_iter"] % 1200000 == 0:
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
		'''
	
	tock = time.time()
	
	string = str(clocks)
	plt.plot(losses, label= string + ' time: ' + str((tock - tick) / nb_iter))
	plt.legend()
	plt.draw()

raw_input()
raw_input()
raw_input()
			