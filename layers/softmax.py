import numpy as np

from base import Layer

class Softmax(Layer):
	def forward(self, X):
		exp = np.exp(X)
		probs = exp / np.sum(exp, axis=1, keepdims=True)
		self.probs = probs
		return probs
	
	def backward(self, dY):
		dX = self.probs - dY
		return dX / (dY.shape[0] * dY.shape[2])
