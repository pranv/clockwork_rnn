import numpy as np


def random(*shape):
	'''
	'''
	return np.random.randn(*shape) * 0.01


def glorotize(W):
	W *= np.sqrt(6)
	W /= np.sqrt(np.sum(W.shape))
	return W 


def orthogonalize(W):
	W, _, _ = np.linalg.svd(W)
	return W
