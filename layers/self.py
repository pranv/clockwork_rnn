import numpy as np

from base import Layer
from utils import random, glorotize, orthogonalize


def pact(X, k=2):
	return np.exp(k * np.cos(X)) / np.exp(k)


def dpact(X, active, k=2):
	X = active * k * -np.sin(X) * k
	return X


class SCRNN(Layer):
	def __init__(self, dinput, nstates, ngroups, doutput, sigma=0.1):
		Wi = random(nstates, dinput + 1) * sigma
		Wh = random(nstates, nstates + 1) * sigma
		Wo = random(doutput, nstates + 1) * sigma
		
		H_0 = np.zeros((nstates, 1))
	
		# column vector to selectively activate rows based on time
		base = np.random.random((ngroups, 1))
		tick = np.random.random((ngroups, 1))
		
		# store it all
		self.dinput = dinput
		self.nstates = nstates
		self.ngroups = ngroups
		self.doutput = doutput

		self.Wi = Wi
		self.Wh = Wh
		self.Wo = Wo
		self.H_0 = H_0

		self.base = base
		self.tick = tick

		self.H_last = None

	def forward(self, X):
		T, n, B = X.shape
		ngroups = self.ngroups
		nstates = self.nstates

		Wi = self.Wi
		Wh = self.Wh
		Wo = self.Wo

		# caches
		inputs, _H_prevs, H_news, Hs, _Hs = {}, {}, {}, {}, {}
		Ys = np.zeros((T, self.doutput, B))

		# H_prev is previous hidden state
		if self.H_last is not None:
			# if we didn't explicitly forget, continue with previous states
			H_prev = self.H_last 
		else:
			H_prev = np.concatenate([self.H_0] * B, axis=1)

		base = self.base
		tick = self.tick

		for t in xrange(T):
			active = pact(base + t * tick).repeat(nstates / ngroups).reshape(-1, 1)
			
			input = np.concatenate([X[t], np.ones((1, B))], axis=0)
			i_h = np.dot(Wi, input)		# input to hidden

			_H_prev = np.concatenate([H_prev, np.ones((1, B))], axis=0) 
			h_h = np.dot(Wh, _H_prev)	# hidden to hidden

			h_new = i_h + h_h
			H_new = np.tanh(h_new)
			
			H = active * H_new + (1 - active) * H_prev

			_H = np.concatenate([H, np.ones((1, B))], axis=0)
			y = np.dot(Wo, _H)
			
			Y = np.tanh(y)
			
			# update
			H_prev = H

			# gotta cache em all
			inputs[t] = input; 
			_H_prevs[t] = _H_prev;  
			H_news[t] = H_new;
			Hs[t] = H; 
			_Hs[t] = _H; 
			Ys[t] = Y
		
		self.inputs = inputs
		self._H_prevs = _H_prevs
		self.H_news = H_news
		self.Hs = Hs
		self._Hs = _Hs
		self.Ys = Ys	
		self.H_last = H
		self.T = T
		self.n = n
		self.B = T

		return Ys[-1:]
	
	def backward(self, dY):
		last_step_error = dY.copy()
		dY = np.zeros_like(self.Ys)
		dY[-1:] = last_step_error[:]

		T, _, B = dY.shape
		n = self.n
		ngroups = self.ngroups
		nstates = self.nstates
		
		Wi = self.Wi
		Wh = self.Wh
		Wo = self.Wo
		
		dH_prev = np.zeros((nstates, B))
		dWi = np.zeros_like(Wi)
		dWh = np.zeros_like(Wh)
		dWo = np.zeros_like(Wo)
		db = np.zeros((self.nstates, B))
		dt = np.zeros((self.nstates, B))
		dX = np.zeros((T, n, B))

		base = self.base
		tick = self.tick

		for t in reversed(xrange(T)):
			active = pact(base + t * tick).repeat(nstates / ngroups).reshape(-1, 1)
			
			input = self.inputs[t]
			_H_prev = self._H_prevs[t]
			H_prev = _H_prev[:-1]
			H_new = self.H_news[t]
			H = self.Hs[t]
			_H = self._Hs[t]
			Y = self.Ys[t]

			dy = (1.0 - Y ** 2) * dY[t]

			dWo += np.dot(dy, _H.T)
			d_H = np.dot(Wo.T, dy)
			
			dH = d_H[:-1] + dH_prev

			dH_prev = (1 - active) * dH

			da = (H_new - H_prev) * dH

			dH_new = active * dH

			dA = dpact(da, active)

			db += dA
			dt += dA * t

			dH_new = (1.0 - H_new ** 2) * dH_new

			dWh += np.dot(dH_new, _H_prev.T)
			dH_prev += np.dot(Wh.T, dH_new)[:-1]

			dWi += np.dot(dH_new, input.T)
			dX[t] = np.dot(Wi.T, dH_new)[:-1]

		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo
		self.db = db.sum(axis=1, keepdims=True)
		self.dt = dt.sum(axis=1, keepdims=True)

		return dX

	def forget(self):
		self.H_last = None

	def remember(self, state):
		self.H_last = state

	def get_weights(self):
		Wi = self.Wi.flatten()
		Wh = self.Wh.flatten()
		Wo = self.Wo.flatten()
		b = self.base.flatten()
		t = self.tick.flatten()
		return np.concatenate([Wi, Wh, Wo, b, t])

	def set_weights(self, W):
		i, h, o, b = self.Wi.size, self.Wh.size, self.Wo.size, self.base.size
		Wi, Wh, Wo, b, t = np.split(W, [i, i + h, i + h + o, i + h + o + b])
		self.Wi = Wi.reshape(self.Wi.shape)
		self.Wh = Wh.reshape(self.Wh.shape)
		self.Wo = Wo.reshape(self.Wo.shape)
		self.base = b.reshape(self.base.shape)
		self.tick = t.reshape(self.tick.shape)
		
	def get_grads(self):
		dWi = self.dWi.flatten()
		dWh = self.dWh.flatten()
		dWo = self.dWo.flatten()
		db = self.db.flatten()
		dt = self.dt.flatten()
		return np.concatenate([dWi, dWh, dWo, db, dt])

	def clear_grads(self):
		self.dWi = None
		self.dWh = None
		self.dWo = None
		self.db = None
		self.dt = None
