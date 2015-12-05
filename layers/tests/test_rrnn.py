import numpy as np

from rrnn import RRNN

def test_CRNN():
	
	T = 50
	batch_size = 2
	nstates = 3
	input_size = 4
	output_size = 6
	nclocks = 3

	unit = RRNN(input_size, nstates, output_size, nclocks)

	W = unit.get_weights()

	X = np.random.randn(T, input_size, batch_size)

	unit.forget()
	acc_Y = unit.forward(X)
	wrand = np.random.randn(*acc_Y.shape)
	loss = np.sum(acc_Y * wrand)
	dY = wrand
	dX = unit.backward(dY)
	dW = unit.get_grads()
	unit.forget()

	def fwd():
		unit.set_weights(W)
		h = unit.forward(X)
		unit.forget()
		return np.sum(h * wrand)

	delta = 1e-4
	error_threshold = 1e-3
	all_values = [X, W]
	backpropagated_gradients = [dX, dW]
	names = ['X', 'W']

	error_count = 0
	for v in range(len(names)):
		values = all_values[v]
		dvalues = backpropagated_gradients[v]
		name = names[v]
		
		for i in range(values.size):
			actual = values.flat[i]
			values.flat[i] = actual + delta
			loss_minus = fwd()
			values.flat[i] = actual - delta
			loss_plus = fwd()
			values.flat[i] = actual
			backpropagated_gradient = dvalues.flat[i]
			numerical_gradient = (loss_minus - loss_plus) / (2 * delta)
			

			if numerical_gradient == 0 and backpropagated_gradient == 0:
				error = 0 
			elif abs(numerical_gradient) < 1e-7 and abs(backpropagated_gradient) < 1e-7:
				error = 0 
			else:
				error = abs(backpropagated_gradient - numerical_gradient) / abs(numerical_gradient + backpropagated_gradient)
			
			if error > error_threshold:
				print 'FAILURE!!!\n'
				print '\tparameter: ', name, '\tindex: ', np.unravel_index(i, values.shape)
				print '\tvalues: ', actual
				print '\tbackpropagated_gradient: ', backpropagated_gradient 
				print '\tnumerical_gradient', numerical_gradient 
				print '\terror: ', error
				print '\n\n'

				error_count += 1

	if error_count == 0:
		print 'RRNN Gradient Check Passed'
	else:
		print 'Failed for {} parameters'.format(error_count)


test_CRNN()
