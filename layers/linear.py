import numpy as np

from base import Layer
from utils import random, glorotize

class Linear(Layer):
	def __init__(self, dinput, doutput):
		W = random(doutput, dinput + 1)
		W = glorotize(W)
		self.W = W

		self.dinput = dinput
		self.doutput = doutput

	def forward(self, X):
		T, n, B = X.shape
		flat_X = X.swapaxes(0, 1).reshape(n, -1)
		flat_X = np.concatenate([flat_X, np.ones((1, B * T))], axis=0)
		Y = np.dot(self.W, flat_X)
		Y = Y.reshape((-1, T, B)).swapaxes(0,1)
		self.flat_X = flat_X
		return Y

	def backward(self, dY):
		T, n, B = dY.shape
		dY = dY.swapaxes(0,1).reshape(n, -1)
		self.dW = np.dot(dY, self.flat_X.T)
		dX = np.dot(self.W.T, dY)
		dX = dX[:-1]
		dX = dX.reshape((-1, T, B)).swapaxes(0,1)
		return dX

	def get_weights(self):
		return self.W.flatten()

	def set_weights(self, W):
		self.W = W.reshape(self.W.shape)

	def get_grads(self):
		return self.dW.flatten()

	def clear_grads(self):
		self.dW *= 0