import numpy as np

from base import Layer
from utils import random

#import matplotlib.pyplot as plt


#plt.ion()
#plt.style.use('kosh')
#plt.figure(num=2)

k = 2

def p_act(X):
	return np.exp(k * np.cos(X)) / np.exp(k)


def dp_act(dY, X, Y):
	dX = Y * k * -np.sin(X) * dY
	return dX


class SCWRNN(Layer):
	def __init__(self, dinput, nstates, ngroups, doutput, sigma=0.1):
		Wi = random(nstates, dinput + 1) * sigma
		Wh = random(nstates, nstates + 1) * sigma
		Wo = random(doutput, nstates + 1) * sigma

		base = np.zeros((ngroups, 1))
		tick = np.random.random((ngroups, 1)) - 1.5
		
		#for n in range(ngroups):
		#	tick[n] = 137.0 / (2.0 ** n)

		# store it all
		self.dinput = dinput
		self.nstates = nstates
		self.ngroups = ngroups
		self.doutput = doutput

		self.Wi = Wi
		self.Wh = Wh
		self.Wo = Wo

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
		times = {}
		_actives = np.zeros((T, ngroups, 1)) 
		actives = np.zeros((T, nstates, 1)) 
		inputs = np.concatenate([X, np.ones((T, 1, B))], axis=1)
		_H_prevs = {}
		H_news = {}
		Hs = {}
		_Hs = {}
		Ys = np.zeros((T, self.doutput, B))

		if self.H_last is not None:
			H_prev = self.H_last 
		else:
			H_prev = np.zeros((nstates, B))

		base = self.base.copy()
		tick = self.tick.copy()

		# 
		timers = np.zeros((nstates, T)) 

		time = base
		for t in xrange(T):
			time = base + t * tick
			_active = p_act(time)
			active = _active.repeat(nstates / ngroups).reshape(-1, 1)

			i_h = np.dot(Wi, inputs[t])		# input to hidden

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
			
			# store for backprop
			times[t] = time
			_actives[t] = _active
			actives[t] = active
			_H_prevs[t] = _H_prev;  
			H_news[t] = H_new;
			Hs[t] = H; 
			_Hs[t] = _H; 
			Ys[t] = Y

			timers[:, t] = active[:, 0]

		#plt.matshow(timers, fignum=2, interpolation='nearest')
		#plt.draw()

		self.times = times
		self._actives = _actives
		self.actives = actives
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
		dbase = np.zeros((ngroups, 1))
		dtick = np.zeros((ngroups, 1))
		dtime_prev = np.zeros((ngroups, 1))
		dX = np.zeros((T, n, B))

		for t in reversed(xrange(T)):
			time = self.times[t]
			_active = self._actives[t]
			active = self.actives[t]
			input = self.inputs[t]
			_H_prev = self._H_prevs[t]
			H_prev = _H_prev[:-1]
			H_new = self.H_news[t]
			H = self.Hs[t]
			_H = self._Hs[t]
			Y = self.Ys[t]

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

			dactive = ((H_new - H_prev) * dH).sum(axis=1, keepdims=True).reshape(ngroups, -1).sum(axis=1, keepdims=True)
			dtime = dp_act(dactive, time, _active)
			dbase += dtime
			dtick += t * dtime

		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo
		self.dbase = dbase * 0
		self.dtick = dtick

		return dX

	def forget(self):
		self.H_last = None

	def remember(self, state):
		self.H_last = state

	def get_weights(self):
		Wi = self.Wi.flatten()
		Wh = self.Wh.flatten()
		Wo = self.Wo.flatten()
		base = self.base.flatten()
		tick = self.tick.flatten()
		return np.concatenate([Wi, Wh, Wo, base, tick])

	def set_weights(self, W):
		i, h, o, b = self.Wi.size, self.Wh.size, self.Wo.size, self.base.size
		Wi, Wh, Wo, base, tick = np.split(W, [i, i + h, i + h + o, i + h + o + b])
		self.Wi = Wi.reshape(self.Wi.shape)
		self.Wh = Wh.reshape(self.Wh.shape)
		self.Wo = Wo.reshape(self.Wo.shape)
		self.base = base.reshape(self.base.shape)
		self.tick = tick.reshape(self.tick.shape)
		
	def get_grads(self):
		dWi = self.dWi.flatten()
		dWh = self.dWh.flatten()
		dWo = self.dWo.flatten()
		dbase = self.dbase.flatten()
		dtick = self.dtick.flatten()
		return np.concatenate([dWi, dWh, dWo, dbase, dtick])

	def clear_grads(self):
		self.dWi = None
		self.dWh = None
		self.dWo = None
		self.dbase = None
		self.dtick = None
