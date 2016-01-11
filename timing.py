import numpy as np

from network_tools import *
from layers import *

from climin import GradientDescent, Adam, RmsProp


T = 100
c = 20

X = np.zeros((T, 1, 1))
Y = np.zeros((T, 1, 1))

for t in range(T):
	if t % c == 0:
		Y[t, : , :] = 1.0

model = [
			LSTM(1, 8),
			LSTM(8, 8),
			Linear(8, 1)
		]


def dW(W):
	load_weights(model, W)
	forget(model)	
	preds = forward(model, X)
	backward(model, preds - Y)
	gradients = extract_grads(model)
	loss = 0.5 * np.sum((Y - preds) ** 2) / T
	print loss
	return gradients


W = extract_weights(model)
optimizer = RmsProp(W, dW, 1e-2)


try:
	for i in optimizer:
		if i['n_iter'] > 3000:
			break
except KeyboardInterrupt:
	pass

forget(model)
a = forward(model, X)

import matplotlib.pyplot as plt
plt.ion()
plt.plot(a[:, 0, 0])
raw_input()

