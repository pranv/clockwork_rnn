import numpy as np
import pickle

from network_tools import *

from mnist import loader


batch_size = 100

data = loader(batch_size=batch_size)

f = open('results/hsn_seq_mnist_elu/final.model', 'r')
model = pickle.load(f)
f.close()

# remove dropout
model = model[1:]

inputs, labels = data.fetch_test()
nsamples = inputs.shape[2]
inputs = np.split(inputs, nsamples / batch_size, axis=2)
labels = np.split(labels, nsamples / batch_size, axis=2)

correct = 0
for j in range(len(inputs)):
	forget(model)
	input = inputs[j]
	label = labels[j]
	pred = forward(model, input)
	good = np.sum(label.argmax(axis=1) == pred.argmax(axis=1))
	correct += good

correct /= float(nsamples)

print 'final accuracy: ', correct