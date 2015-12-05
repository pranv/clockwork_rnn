import numpy as np

class RRNN(object):
	def __init__(self, dinput, nstates, doutput, nclocks, sigma=0.01, orthogonal=False, identity=False):
		np.random.seed(np.random.randint(1213))
		
		Wi = np.random.randn(nclocks * nstates, dinput + 1) * sigma
		Wh = np.random.randn(nclocks * nstates, nclocks * nstates + 1) * sigma
		Wo = np.random.randn(doutput, nclocks * nstates + 1) * sigma
		
		if orthogonal:
			Wh[:, :-1], _, _ = np.linalg.svd(Wh[:, :-1])

		if identity:
			Wh[:, :-1] = np.eye(nclocks * nstates, nclocks * nstates)

		H_0 = np.zeros((nclocks * nstates, 1))

		# store it all
		self.dinput = dinput
		self.nstates = nstates
		self.doutput = doutput
		self.nclocks = nclocks
		self.Wi = Wi
		self.Wh = Wh
		self.Wo = Wo
		self.H_0 = H_0

		self.forget()

	def forward(self, X):
		T, n, B = X.shape
		nclocks = self.nclocks
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

		for t in xrange(T):
			i = t % nclocks
			active = np.zeros_like(H_prev)
			active[i * nstates: (i + 1) * nstates, :] = 1
		
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

		return Ys
	
	def backward(self, dY):
		T, _, B = dY.shape
		n = self.n
		nclocks = self.nclocks
		nstates = self.nstates
		
		Wi = self.Wi
		Wh = self.Wh
		Wo = self.Wo
		
		dH_prev = np.zeros((nclocks * nstates, B))
		dWi = np.zeros_like(Wi)
		dWh = np.zeros_like(Wh)
		dWo = np.zeros_like(Wo)
		
		dX = np.zeros((T, n, B))
		
		for t in reversed(xrange(T)):
			i = t % nclocks
			active = np.zeros_like(dH_prev)
			active[i * nstates: (i + 1) * nstates, :] = 1

			input = self.inputs[t]
			_H_prev = self._H_prevs[t]
			H_new= self.H_news[t]
			H = self.Hs[t]
			_H = self._Hs[t]
			Y = self.Ys[t]

			dy = (1.0 - Y ** 2) * dY[t]

			dWo_t = np.dot(dy, _H.T)
			dWo += dWo_t
			d_H = np.dot(Wo.T, dy)
			
			dH = d_H[:-1] + dH_prev

			dH_prev = (1 - active) * dH
			
			dH_new = active * dH

			dH_new = (1.0 - H_new ** 2) * dH_new

			dWh_t = np.dot(dH_new, _H_prev.T)
			dWh += dWh_t
			dH_prev += np.dot(Wh.T, dH_new)[:-1]

			dWi_t = np.dot(dH_new, input.T)
			dWi += dWi_t

			dX[t] = np.dot(Wi.T, dH_new)[:-1]

		self.dWi = dWi
		self.dWh = dWh
		self.dWo = dWo

		return dX

	def forget(self):
		self.H_last = None

	def remember(self, state):
		self.H_last = state

	def get_weights(self):
		Wi = self.Wi.flatten()
		Wh = self.Wh.flatten()
		Wo = self.Wo.flatten()
		return np.concatenate([Wi, Wh, Wo])

	def set_weights(self, W):
		i, h = self.Wi.size, self.Wh.size
		Wi, Wh, Wo = np.split(W, [i, i + h])
		self.Wi = Wi.reshape(self.Wi.shape)
		self.Wh = Wh.reshape(self.Wh.shape)
		self.Wo = Wo.reshape(self.Wo.shape)
		
	def get_grads(self):
		dWi = self.dWi.flatten()
		dWh = self.dWh.flatten()
		dWo = self.dWo.flatten()
		return np.concatenate([dWi, dWh, dWo])

	def clear_grads(self):
		self.dWi *= 0
		self.dWh *= 0
		self.dWo *= 0
