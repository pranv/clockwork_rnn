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
	def __init__(self, dinput, nstates, doutput, clock_rates, learn_state=True):
		nclocks = len(clock_rates)
		
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
		Wh[:,:-1] = utri_mask * Wh[:,:-1]

		# column vector to selectively activate block rows based on time
		schedules = make_schedule(clock_rates, nstates)
		schedules = np.array(schedules).reshape(-1, 1)

		# store it all
		self.dinput = dinput; self.nstates = nstates; self.doutput = doutput; self.clock_rates = clock_rates; self.nclocks = nclocks; self.Wi = Wi; self.Wh = Wh; self.Wo = Wo; self.H_0 = H_0; self.utri_mask = utri_mask; self.schedules = schedules; self.learn_state = learn_state;

		self.forget()

	def forward(self, X):
		T, n, B = X.shape
		nclocks = self.nclocks
		nstates = self.nstates
		doutput = self.doutput

		activates = np.zeros((T, self.schedules.shape[0]))
		inputs = np.ones((T, n + 1, B))
		_H_ps = np.ones((T, nclocks * nstates + 1, B))
		H1s = np.zeros((T, nclocks * nstates, B))
		Hs = np.zeros((T, nclocks * nstates, B))
		_Hs = np.ones((T, nclocks * nstates + 1, B))
		Ys = np.zeros((T, doutput, B))

		inputs[:, :-1, :] = X


		if self.H_T is not None:
			H_p = self.H_T 
		else:
			H_p = np.concatenate([self.H_0] * B, axis=1)

		schedules = self.schedules
		Wi = self.Wi
		Wo = self.Wo
		Wh = self.Wh

		for t in xrange(T):
			activate = ((t % schedules) == 0)
			
			input = inputs[t]
			_H_ps[t, :-1, :] = H_p 

			H1 = np.tanh(np.dot(Wi, input) + np.dot(Wh, _H_ps[t]))
			H = H_p * (1 - activate) + H1 * activate

			_Hs[t, :-1, :] = H
			Y = np.tanh(np.dot(Wo, _Hs[t]))

			H_p = H

 			activates[t] = activate.flatten()
			H1s[t] = H1;
			Hs[t] = H;  
			Ys[t] = Y
		
		self.activates = activates
		self.inputs = inputs
		self._H_ps = _H_ps
		self.H1s = H1s
		self.Hs = Hs
		self._Hs = _Hs
		self.Ys = Ys
		
		self.H_T = Hs[-1]
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
			activate = self.activates[t].reshape(-1, 1)

			input = self.inputs[t]
			_H_p = self._H_ps[t]
			H1= self.H1s[t]
			H = self.Hs[t]
			_H = self._Hs[t]
			Y = self.Ys[t];

			dy = (1.0 - Y ** 2) * dY[t]

			dWo += np.dot(dy, _H.T)
			d_H = np.dot(self.Wo.T, dy)
			
			dH = d_H[:-1] + dH_p

			dH_p = (1 - activate) * dH
			
			dH1 = activate * dH

			dH1 = (1.0 - H1 ** 2) * dH1

			dh_h = dH1
			dWh += np.dot(dh_h, _H_p.T)
			dH_p += np.dot(self.Wh.T, dh_h)[:-1]

			di_h = dH1
			dWi += np.dot(di_h, input.T)
			#dX[t] = np.dot(self.Wi.T, di_h)[:-1]

		# mask grads, so zeros grads for lower triangle
		dWh[:, :-1] = self.utri_mask * dWh[:, :-1]

		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo
		
		if self.learn_state:
			self.dH_0 = dH_p
		else:
			self.dH_0 = np.zeros_like(dH_p)

		Ys = np.zeros(0)

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
		dWi *= 0
		dWh *= 0
		dWo *= 0
		dH_0 *= 0
