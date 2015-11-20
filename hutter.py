import numpy as np

from layers import *
from network import *
from climin import RmsProp, Adam
from text import Corpus

import time
import pickle

import matplotlib.pyplot as plt
plt.ion()


clocks = [1, 4, 16, 64, 16, 4, 1]
vocabulary_size = 205
states = 2048 / 4			# per clock
output = 8192 / 4			# for all clocks
ninputs = vocabulary_size
noutputs = vocabulary_size

sequence_length = 128

batch_size = 16
learning_rate = 3e-4
nepochs = 10
niterations = 100000 * nepochs
momentum = 0.7

forget_every = 5
gradient_clip = (-10.0, 10.0)

sample_every = 1000
save_every = 500
plot_every = 100


logs = {}
logs['loss'] = []
logs['smooth_loss'] = [5.32]
logs['gradient_norm'] = []
logs['clipped_gradient_norm'] = []


data = Corpus('enwik8', sequence_length, batch_size)
model = [CRNN(ninputs, states, output, clocks), Linear(output, noutputs), Softmax()]


def dW(W):
	load_weights(model, W)
	input, target = data.fetch()
	output = forward(model, input)
	backward(model, target)
	gradients = extract_grads(model)
	clipped_gradients = np.clip(gradients, gradient_clip[0], gradient_clip[1])

	loss = -1.0 * np.sum(target * np.log(output + 2e-23)) / (sequence_length * batch_size)
	gradient_norm = (gradients ** 2).sum() / gradients.size
	clipped_gradient_norm = (clipped_gradients ** 2).sum() / gradients.size
	
	logs['loss'].append(loss)
	logs['smooth_loss'].append(loss * 0.01 + logs['smooth_loss'][-1] * 0.99)
	logs['gradient_norm'].append(gradient_norm) 
	logs['clipped_gradient_norm'].append(clipped_gradient_norm) 
	
	return clipped_gradients

W = extract_weights(model)

optimizer = Adam(W, dW, learning_rate, momentum=momentum)

tick = time.time()
for i in optimizer:
	print i['n_iter'], '\t',
	print logs['loss'][-1], '\t',
	print logs['gradient_norm'][-1]
	print time.time() - tick
	tick = time.time()
	
	if i['n_iter'] % forget_every == 0:
		forget(model)
		if i['n_iter'] > 100:
			forget_every = 10
		if i['n_iter'] > 1000:
			forget_every = 100
		if i['n_iter'] > 10000:
			forget_every = 1000

	if i['n_iter'] % plot_every == 0:
		plt.clf()
		plt.plot(logs['smooth_loss'])
		plt.draw()

	if i['n_iter'] % sample_every == 0:
		forget(model)
		x = np.zeros((20, vocabulary_size, 1))
		input, _ = data.fetch()
		x[:20, :, :] = input[:20, :, 0:1]
		ixes = []
		for t in xrange(1000):
			p = forward(model, np.array(x))
			p = p[-1]
			ix = np.random.choice(range(vocabulary_size), p=p.ravel())
			x = np.zeros((1, vocabulary_size, 1))
			x[0, ix, 0] = 1
			ixes.append(ix)
		sample = ''.join(data.decoder.to_c[ix] for ix in ixes)
		print '----' * 20
		print sample
		print '----' * 20
		forget(model)

	if i['n_iter'] % save_every == 0:
		print 'serializing... '
		f = open('model' + str(i['n_iter']), 'w')
		pickle.dump(model, f)
		f.close()

		print 'serializing logs... '
		f = open('logs_' + str(i['n_iter']), 'w')
		pickle.dump(logs, f)
		f.close()

	if i['n_iter'] > 2000:
		optimizer.step_rate = learning_rate / 2
	elif i['n_iter'] > 4000:
		optimizer.step_rate = learning_rate / 4
	elif i['n_iter'] > niterations:
		break


print 'serializing final model... '
f = open('model_' + str(i['n_iter']), 'w')
pickle.dump(model, f)
f.close()

print 'serializing logs... '
f = open('logs_' + str(i['n_iter']), 'w')
pickle.dump(logs, f)
f.close()
