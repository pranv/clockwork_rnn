import numpy as np

from layers import *

from network_tools import *


time_steps = 2
batch_size = 2
nstates = 2
dinput = 4
doutput = 4
ep = 2e-23

model = [
			Softmax(),
 		]

X = np.random.randn(time_steps, dinput, batch_size)
#W = extract_weights(model)

forget(model)
acc_Y = forward(model, X)
#target = np.random.binomial([np.ones_like(acc_Y)], 1.0 / dinput)[0]
target = np.zeros_like(acc_Y)
target[0][2][0] = 1
target[1][3][0] = 1
target[0][1][1] = 1
target[1][2][1] = 1
loss = -1.0 * np.sum(target * np.log(acc_Y)) / (batch_size * time_steps)

dX = backward(model, target)
#dW = extract_grads(model)

forget(model)

def fwd():
	y = forward(model, X)
	forget(model)
	return -1.0 * np.sum(target * np.log(y)) / (batch_size * time_steps)


delta = 1e-5
error_threshold = 1e-3
all_values = [X]
backpropagated_gradients = [dX]
names = ['X']


error_count = 0
error_sum = 0
for v in range(len(names)):
	values = all_values[v]
	dvalues = backpropagated_gradients[v]
	name = names[v]
	
	for i in range(values.size):
		actual = values.flat[i]

		values.flat[i] = actual + delta
		loss_plus = fwd()

		values.flat[i] = actual - delta
		loss_minus = fwd()

		values.flat[i] = actual
		
		numerical_gradient = (loss_plus - loss_minus) / (2 * delta)
		backpropagated_gradient = dvalues.flat[i]
		
		if numerical_gradient == 0 and backpropagated_gradient == 0:
			error = 0 
		elif abs(numerical_gradient) < 1e-7 and abs(backpropagated_gradient) < 1e-7:
			error = 0 
		else:
			error = abs(backpropagated_gradient - numerical_gradient) / abs(numerical_gradient + backpropagated_gradient)
		
		error_sum += error
		
		if error > error_threshold:
			print 'FAILURE!!!\n'
			print '\tparameter: ', name, '\tindex: ', np.unravel_index(i, values.shape)
			print '\tvalues: ', actual
			print '\tbackpropagated_gradient: ', backpropagated_gradient 
			print '\tnumerical_gradient', numerical_gradient 
			print '\terror: ', error
			print '\n\n'

			error_count += 1

print 'total error: ', error_sum

if error_count == 0:
	print 'Gradient Check Passed'
else:
	print 'Failed for {}/{} parameters'.format(error_count, dX.size)# + dW.size)
