import numpy as np

from base import Layer
from utils import random


def make_schedule(clock_periods, nstates):
	sch = []
	for c in clock_periods:
		for i in range(nstates):
			sch.append(c)
	return sch


def sigmoid(X, B=1):
	return 1 / (1 + np.exp(-B * X))


def sigmoid_grad(dY, Y, B=1):
	return Y * (1.0 - Y) * B * dY


class SCWRNN2(Layer):
	def __init__(self, dinput, nstates, clocks, doutput, sigma=0.1):
		ngroups = len(clocks)
		Wi = random(ngroups * nstates, dinput + 1) * sigma
		Wh = random(ngroups * nstates, ngroups * nstates + 1) * sigma
		Wo = random(doutput, ngroups * nstates + 1) * sigma

		connection_matrix = np.random.random((ngroups, ngroups)) - 0.5

		schedules = make_schedule(clocks, nstates)
		schedules = np.array(schedules).reshape(-1, 1)
		
		# store it all
		self.dinput = dinput
		self.nstates = nstates
		self.ngroups = ngroups
		self.doutput = doutput

		self.Wi = Wi
		self.Wh = Wh
		self.Wo = Wo
		self.connection_matrix = connection_matrix
		self.schedules = schedules

		self.H_last = None


	def forward(self, X):
		T, n, B = X.shape
		ngroups = self.ngroups
		nstates = self.nstates

		Wi = self.Wi
		Wh = self.Wh.copy()
		Wo = self.Wo
		connection_matrix = self.connection_matrix

		mask = connection_matrix.repeat(nstates, axis=0).repeat(nstates, axis=1)
		mask = sigmoid(mask)
		Wh[:, :-1] *= mask

		self.mask = mask

		# caches
		inputs = np.concatenate([X, np.ones((T, 1, B))], axis=1)
		_H_prevs = {}
		H_news = {}
		Hs = {}
		_Hs = {}
		Ys = np.zeros((T, self.doutput, B))

		if self.H_last is not None:
			H_prev = self.H_last 
		else:
			H_prev = np.zeros((ngroups * nstates, B))


		for t in xrange(T):
			active = (((t) % self.schedules) == 0)

			i_h = np.dot(Wi, inputs[t])		# input to hidden

			_H_prev = np.concatenate([H_prev, np.ones((1, B))], axis=0) 
			h_h = np.dot(Wh, _H_prev)	# hidden to hidden

			h_new = h_h + i_h
			H_new = np.tanh(h_new)
			
			H = active * H_new + (1 - active) * H_prev	

			_H = np.concatenate([H, np.ones((1, B))], axis=0)
			y = np.dot(Wo, _H)
			Y = np.tanh(y)
			
			# update
			H_prev = H
			
			# store for backprop
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
		Wh = self.Wh.copy()
		Wo = self.Wo

		Wh[:, :-1] *= self.mask
		
		dH_prev = np.zeros((ngroups * nstates, B))
		dWi = np.zeros_like(Wi)
		dWh = np.zeros_like(Wh)
		dWo = np.zeros_like(Wo)
		dX = np.zeros((T, n, B))

		for t in reversed(xrange(T)):
			input = self.inputs[t]
			_H_prev = self._H_prevs[t]
			H_prev = _H_prev[:-1]
			H_new = self.H_news[t]
			H = self.Hs[t]
			_H = self._Hs[t]
			Y = self.Ys[t]

			active = (((t) % self.schedules) == 0)

			dy = (1.0 - Y ** 2) * dY[t]

			dWo += np.dot(dy, _H.T)
			d_H = np.dot(Wo.T, dy)[:-1]
			
			dH = d_H + dH_prev

			dH_new = active * dH
			dH_prev = (1 - active) * dH

			dH_new = (1.0 - H_new ** 2) * dH_new

			dWh += np.dot(dH_new, _H_prev.T)
			dH_prev += np.dot(Wh.T, dH_new)[:-1]

			dWi += np.dot(dH_new, input.T)
			dX[t] = np.dot(Wi.T, dH_new)[:-1]

		dmask = dWh[:, :-1] * self.Wh[:, :-1]		
		dWh[:, :-1] *= self.mask
		dmask = sigmoid_grad(dmask, self.mask)
		dconnection_matrix = dmask.reshape(ngroups, nstates, ngroups, nstates).sum(axis=3).sum(axis=1)
		
		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo
		self.dconnection_matrix = dconnection_matrix

		return dX

	def forget(self):
		self.H_last = None

	def remember(self, state):
		self.H_last = state

	def get_weights(self):
		Wi = self.Wi.flatten()
		Wh = self.Wh.flatten()
		Wo = self.Wo.flatten()
		connection_matrix = self.connection_matrix.flatten()
		return np.concatenate([Wi, Wh, Wo, connection_matrix])

	def set_weights(self, W):
		i, h, o = self.Wi.size, self.Wh.size, self.Wo.size
		Wi, Wh, Wo, connection_matrix = np.split(W, [i, i + h, i + h + o])
		self.Wi = Wi.reshape(self.Wi.shape)
		self.Wh = Wh.reshape(self.Wh.shape)
		self.Wo = Wo.reshape(self.Wo.shape)
		self.connection_matrix = connection_matrix.reshape(self.connection_matrix.shape)
		
	def get_grads(self):
		dWi = self.dWi.flatten()
		dWh = self.dWh.flatten()
		dWo = self.dWo.flatten()
		dconnection_matrix = self.dconnection_matrix.flatten()
		return np.concatenate([dWi, dWh, dWo, dconnection_matrix])

	def clear_grads(self):
		self.dWi = None
		self.dWh = None
		self.dWo = None
		self.dconnection_matrix = None
