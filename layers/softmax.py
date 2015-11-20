import numpy as np

from base import Layer

class Softmax(Layer):
	def forward(self, X):
		max = np.amax(X, axis=1, keepdims=True)
		exp = np.exp(X - max)
		probs = exp / np.sum(exp, axis=1, keepdims=True)

		self.probs = probs
		
		return probs
	
	def backward(self, Y):
		dX = self.probs - Y

		return dX / (Y.shape[0] * Y.shape[2])
