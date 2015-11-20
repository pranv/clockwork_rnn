import numpy as np


def random(*shape):
	return np.random.randn(*shape)


def glorotize(W):
	W *= np.sqrt(6)
	W /= np.sqrt(np.sum(W.shape))
	return W 


def orthogonalize(W):
	W[:, :-1], _, _ = np.linalg.svd(W[:, :-1])
	return W
