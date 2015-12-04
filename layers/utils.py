import numpy as np


def random(*shape):
	'''
		Gaussian with sigma = 0.01 and mu = 0.0
	'''
	np.random.seed(np.random.randint(1213))
	return np.random.randn(*shape)


def glorotize(W):
	W *= np.sqrt(6)
	W /= np.sqrt(np.sum(W.shape))
	return W 


def orthogonalize(W):
	W, _, _ = np.linalg.svd(W)
	return W
