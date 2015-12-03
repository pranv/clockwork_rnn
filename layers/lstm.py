import numpy as np

from base import Layer
from utils import random, glorotize, orthogonalize

class LSTM(Layer):
	def __init__(self, dinput, nstates, fbias=5.0):
		W = random(nstates * 4, dinput + nstates + 1)
		W = glorotize(W)
		W[0 * nstates: 1 * nstates, dinput:-1] = orthogonalize(W[0 * nstates: 1 * nstates, dinput:-1])
		W[1 * nstates: 2 * nstates, dinput:-1] = orthogonalize(W[1 * nstates: 2 * nstates, dinput:-1])
		W[2 * nstates: 3 * nstates, dinput:-1] = orthogonalize(W[2 * nstates: 3 * nstates, dinput:-1])
		W[3 * nstates: 4 * nstates, dinput:-1] = orthogonalize(W[3 * nstates: 4 * nstates, dinput:])
		W[:, -1] = 0 								# initialize all biases to zero
		W[2 * nstates : 3 * nstates] = fbias		# fancy forget bias
		self.W = W

		self.c_0 = np.zeros((nstates, 1))
		self.Y_0 = np.zeros((nstates, 1))

		self.dinput = dinput
		self.nstates = nstates

		self.forget()
	
	def forward(self, X):
		T, _, B = X.shape
		nstates = self.nstates

		self.acc_V, self.acc_S, self.acc_z, self.acc_i, self.acc_f, self.acc_o, \
			self.acc_Z, self.acc_I, self.acc_F, self.acc_O, self.acc_c, self.acc_C, self.acc_Y \
				= {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
		
		if self.c_T is not None:
			c_start = self.c_T 
		else:
			c_start = np.concatenate([self.c_0] * B, axis=1)

		if self.Y_T is not None:
			Y_start = self.Y_T
		else:
			Y_start = np.concatenate([self.Y_0] * B, axis=1)

		c_p, Y_p = c_start , Y_start 

		Ys = np.zeros((T, nstates, B))

		for t in range(T):
			V = np.concatenate([X[t], Y_p, np.ones((1, B))], axis=0)
			
			# giant matrix multiply that gives us all transforms
			S = np.dot(self.W, V)
			
			# split it up for easier calculations
			z, i, f, o = np.split(S, 4, axis=0)
			
			# apply activations
			Z, F, O = 1.0 / (1.0 + np.exp(-z)), 1.0 / (1.0 + np.exp(-f)), 1.0 / (1.0 + np.exp(-o)) # sigmoid
			I = np.tanh(i)

			# memory update
			c = Z * I + F * c_p
			C = np.tanh(c)

			# outputs
			Y = O * C

			# store all signals to accumulators for use in backprop
			self.acc_V[t] = V; self.acc_S[t] = S; self.acc_z[t] = z; self.acc_i[t] = i; self.acc_f[t] = f; self.acc_o[t] = o; self.acc_Z[t] = Z; self.acc_I[t] = I; self.acc_F[t] = F; self.acc_O[t] = O; self.acc_c[t] = c; self.acc_C[t] = C

			Ys[t] += Y

			# updates
			c_p = c
			Y_p = Y


		self.c_start = c_start
		self.Y_start = Y_start
		self.c_T = c
		self.Y_T = Y

		return Ys

	def backward(self, dY):
		T, _, B = dY.shape
		h = self.nstates
		dinput = self.dinput

		dW = np.zeros_like(self.W)
		dX = np.zeros((T, dinput, B))
		dc_p = np.zeros((h, B))
		dY_p = np.zeros((h, B))

		for t in reversed(range(T)):
			V, S, z, i, f, o, Z, I, F, O, c, C = self.acc_V[t], self.acc_S[t], \
				self.acc_z[t], self.acc_i[t], self.acc_f[t], self.acc_o[t], self.acc_Z[t], \
					self.acc_I[t], self.acc_F[t], self.acc_O[t], self.acc_c[t], self.acc_C[t]

			dY_t = dY[t] + dY_p

			if t == 0:
				c_p = self.c_start
			else:
				c_p = self.acc_c[t - 1]

			# emelent-wise multiplication of squshed cell states and output through output gate
			dO = C * dY_t
			dC = O * dY_t

			# through sigmoid squash of output gate
			do = O * (1.0 - O) * dO

			# through tanh squash of cell states
			dc = (1.0 - C ** 2) * dC

			# gradient of cell states from previous step (actually next step)
			dc = dc + dc_p

			# element-wies multiplication of previous cell states and forget gate
			dF = c_p * dc
			dc_p = F * dc

			# through sigmoid squash of forget gate
			df = F * (1.0 - F) * dF

			# through the emelent-wise multiplication of input gate and input
			dI = Z * dc
			dZ = I * dc

			# through the tanh of the input gate
			di = (1.0 - I ** 2) * dI

			# through the sigmoid of input activation
			dz = Z * (1.0 - Z) * dZ

			dS = np.concatenate([dz, di, df, do], axis=0)

			# grad for this time step
			dW += np.dot(dS, V.T)
			dV = np.dot(self.W.T, dS)

			# error in the input
			dX[t] += dV[:dinput]
			
			# pass the error to the previous iteration
			dY_p = dV[dinput:-1]

		self.dX = dX
		self.dW = dW
		self.dc_0 = dc_p
		self.dY_0 = dY_p

		# clear all 
		self.acc_V, self.acc_S, self.acc_z, self.acc_i, self.acc_f, self.acc_o, \
			self.acc_Z, self.acc_I, self.acc_F, self.acc_O, self.acc_c, self.acc_C, self.acc_Y \
				= {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

		return dX

	def forget(self):
		self.c_T = None
		self.Y_T = None

	def remember(self, state):
		self.c_T, self.Y_T = np.split(state, 2, axis=1)

	def get_weights(self):
		W = self.W.flatten()
		c_0 = self.c_0.flatten()
		Y_0 = self.Y_0.flatten()
		return np.concatenate([W, c_0, Y_0])

	def set_weights(self, W):
		w, c = self.W.size, self.c_0.size
		W, c, Y = np.split(W, [w, w + c])
		self.W = W.reshape(self.W.shape)
		self.c_0 = c.reshape(self.c_0.shape)
		self.Y_0 = Y.reshape(self.Y_0.shape)
		self.forget()

	def get_grads(self):
		dW = self.dW.flatten()
		dc_0 = self.dc_0.sum(axis=1, keepdims=True).flatten()
		dY_0 = self.dY_0.sum(axis=1, keepdims=True).flatten()
		return np.concatenate([dW, dc_0, dY_0])

	def clear_grads(self):
		self.dW *= 0
		self.dc_0 *= 0
		self.dY_0 *= 0
