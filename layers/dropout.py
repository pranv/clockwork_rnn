import numpy as np

from base import Layer

class Dropout(Layer):
	def __init__(self, p):
		self.p = p

	def forward(self, X):
		p = self.p
		mask = np.random.binomial([np.ones_like(X)], 1 - p)[0]
		Y =  (mask * X) * (1.0 / (1 - p))
		return Y

	def backward(self, dY):
		dX = dY
		return dX