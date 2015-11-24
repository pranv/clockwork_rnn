import numpy as np

from base import Layer
from utils import random, glorotize, orthogonalize


def recurrent_mask(nclocks, nstates):
	matrix = []
	for c in range(nclocks, 0, -1):
		blocks = [np.zeros((nstates, nstates))] * (nclocks - c)
		blocks.extend([np.ones((nstates, nstates))] * c)
		matrix.append(np.concatenate(blocks, axis=1))
	mask = np.concatenate(matrix, axis=0)
	return mask


def make_schedule(clocks, n):
	sch = []
	for c in clocks:
		for i in range(n):
			sch.append(c)
	return sch


class CRNN(Layer):
	def __init__(self, dinput, nstates, doutput, clock_periods, full_recurrence=False, learn_state=True):
		nclocks = len(clock_periods)
		
		Wi = random(nclocks * nstates, dinput + 1)
		Wh = random(nclocks * nstates, nclocks * nstates + 1)
		Wo = random(doutput, nclocks * nstates + 1)
		
		if learn_state:
			H_0 = random(nclocks * nstates, 1)
		else:
			H_0 = np.zeros((nclocks * nstates, 1))

		# some fancy inits
		Wi = glorotize(Wi)
		Wh = orthogonalize(Wh)
		Wo = glorotize(Wo)
	
		# mask to make Wh a block upper triangle matrix
		utri_mask = recurrent_mask(nclocks, nstates)
		if not full_recurrence:
			Wh[:,:-1] *= utri_mask

		# column vector to selectively activate block rows based on time
		schedules = make_schedule(clock_periods, nstates)
		schedules = np.array(schedules).reshape(-1, 1)

		# store it all
		self.dinput = dinput
		self.nstates = nstates
		self.doutput = doutput
		self.clock_periods = clock_periods
		self.nclocks = nclocks
		self.Wi = Wi
		self.Wh = Wh
		self.Wo = Wo
		self.H_0 = H_0
		self.utri_mask = utri_mask
		self.schedules = schedules
		self.full_recurrence = full_recurrence
		self.learn_state = learn_state

		self.forget()

	def forward(self, X):
		T, n, B = X.shape

		self.inputs, self._H_ps, \
		 self.H1s, self.Hs, self._Hs = {}, {}, {}, {}, {}
		Ys = np.zeros((T, self.doutput, B))

		if self.H_T is not None:
			H_start = self.H_T 
		else:
			H_start = np.concatenate([self.H_0] * B, axis=1)

		H_p = H_start

		for t in xrange(T):
			activate = (((t + 1) % self.schedules) == 0)
			
			input = np.concatenate([X[t], np.ones((1, B))], axis=0)
			i_h = np.dot(self.Wi, input)

			_H_p = np.concatenate([H_p, np.ones((1, B))], axis=0) 
			h_h = np.dot(self.Wh, _H_p)

			H1 = np.tanh(i_h + h_h)
			H = activate * H1 + (1 - activate) * H_p

			_H = np.concatenate([H, np.ones((1, B))], axis=0)
			y = np.dot(self.Wo, _H)
			Y = np.tanh(y)
			
			# update
			H_p = H

			# gotta cache em all
			self.inputs[t] = input; 
			self._H_ps[t] = _H_p;  
			self.H1s[t] = H1;
			self.Hs[t] = H; 
			self._Hs[t] = _H; 
			
			Ys[t] += Y
			
		self.Ys = Ys
		self.H_T = H
		self.T = T
		self.n = n
		self.B = T

		return Ys
	
	def backward(self, dY):
		T, _, B = dY.shape
		
		dH_p = np.zeros((self.nclocks * self.nstates, B))
		dWi = np.zeros_like(self.Wi)
		dWh = np.zeros_like(self.Wh)
		dWo = np.zeros_like(self.Wo)
		dX = np.zeros((T, self.n, B))

		for t in reversed(xrange(T)):
			activate = (((t + 1) % self.schedules) == 0)

			input = self.inputs[t]; _H_p = self._H_ps[t]; H1= self.H1s[t]; H = self.Hs[t]; _H = self._Hs[t]; Y = self.Ys[t];

			dy = (1.0 - Y ** 2) * dY[t]

			dWo += np.dot(dy, _H.T)
			d_H = np.dot(self.Wo.T, dy)
			
			dH = d_H[:-1] + dH_p

			dH_p = (1 - activate) * dH
			
			dH1 = activate * dH

			dH1 = (1.0 - H1 ** 2) * dH1

			dWh += np.dot(dH1, _H_p.T)
			dH_p += np.dot(self.Wh.T, dH1)[:-1]

			dWi += np.dot(dH1, input.T)
			dX[t] = np.dot(self.Wi.T, dH1)[:-1]

		# mask grads, so zeros grads for lower triangle
		if not self.full_recurrence:
			dWh[:, :-1] *= self.utri_mask

		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo
		
		if self.learn_state:
			self.dH_0 = dH_p
		else:
			self.dH_0 = np.zeros_like(dH_p)

		return dX

	def forget(self):
		self.H_T = None

	def remember(self, state):
		self.H_T = state

	def get_weights(self):
		Wi = self.Wi.flatten()
		Wh = self.Wh.flatten()
		Wo = self.Wo.flatten()
		H_0 = self.H_0.flatten()
		return np.concatenate([Wi, Wh, Wo, H_0])

	def set_weights(self, W):
		i, h, o = self.Wi.size, self.Wh.size, self.Wo.size
		Wi, Wh, Wo, H_0 = np.split(W, [i, i + h, i + h + o])
		self.Wi = Wi.reshape(self.Wi.shape)
		self.Wh = Wh.reshape(self.Wh.shape)
		self.Wo = Wo.reshape(self.Wo.shape)
		self.H_0 = H_0.reshape(self.H_0.shape)
		
	def get_grads(self):
		dWi = self.dWi.flatten()
		dWh = self.dWh.flatten()
		dWo = self.dWo.flatten()
		dH_0 = self.dH_0.sum(axis=1, keepdims=True).flatten()
		return np.concatenate([dWi, dWh, dWo, dH_0])

	def clear_grads(self):
		self.dWi *= 0
		self.dWh *= 0
		self.dWo *= 0
		self.dH_0 *= 0
