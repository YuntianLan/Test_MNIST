import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def toFloat(lst):
	for i in range(len(lst)):
		lst[i] = float(lst[i])
	return lst


def get_mnist_data():
	with open('mnist_train.csv','rb') as train_file:
		train_reader = csv.reader(train_file,delimiter=',',quotechar='|')
		i = 0
		train_X = np.zeros((50000,784))
		train_Y = np.zeros(50000,dtype=int)
		for row in train_reader:
			i += 1
			if i>=50000: break
			floatRow = toFloat(row) # len = 785
			train_X[i] = np.array(floatRow[1:])
			train_Y[i] = np.array(int(floatRow[0]))
		train_Y = train_Y.T
	return train_X, train_Y

def two_layer():
	N, D1, D2, D3 = 200, 50, 60, 3
	X = np.random.randn(N, D1)
	W1 = np.random.randn(D1, D2)
	W2 = np.random.randn(D2, D3)
	a = np.maximum(0, X.dot(W1)).dot(W2)

	print 'Before batch normalization:'
	print '  means: ', a.mean(axis=0)
	print '  stds: ', a.std(axis=0)

	# Means should be close to zero and stds close to one
	print 'After batch normalization (gamma=1, beta=0)'
	a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
	print '  mean: ', a_norm.mean(axis=0)
	print '  std: ', a_norm.std(axis=0)

	# Now means should be close to beta and stds close to gamma
	gamma = np.asarray([1.0, 2.0, 3.0])
	beta = np.asarray([11.0, 12.0, 13.0])
	a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
	print 'After batch normalization (nontrivial gamma, beta)'
	print '  means: ', a_norm.mean(axis=0)
	print '  stds: ', a_norm.std(axis=0)

def test_time_forward():
	N, D1, D2, D3 = 200, 50, 60, 3
	W1 = np.random.randn(D1, D2)
	W2 = np.random.randn(D2, D3)

	bn_param = {'mode': 'train'}
	gamma = np.ones(D3)
	beta = np.zeros(D3)
	for t in xrange(50):
	  X = np.random.randn(N, D1)
	  a = np.maximum(0, X.dot(W1)).dot(W2)
	  batchnorm_forward(a, gamma, beta, bn_param)
	bn_param['mode'] = 'test'
	X = np.random.randn(N, D1)
	a = np.maximum(0, X.dot(W1)).dot(W2)
	a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)

	# Means should be close to zero and stds close to one, but will be
	# noisier than training-time forward passes.
	print 'After batch normalization (test-time):'
	print '  means: ', a_norm.mean(axis=0)
	print '  stds: ', a_norm.std(axis=0)

def deep_net():
	hidden_dims = [100, 100, 100, 100, 100]

	num_train = 1000
	small_data = {
	  'X_train': data['X_train'][:num_train],
	  'y_train': data['y_train'][:num_train],
	  'X_val': data['X_val'],
	  'y_val': data['y_val'],
	}

	weight_scale = 2e-2
	bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
	model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

	bn_solver = Solver(bn_model, small_data,
	                num_epochs=10, batch_size=50,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 1e-3,
	                },
	                verbose=True, print_every=200)
	bn_solver.train()

	solver = Solver(model, small_data,
	                num_epochs=10, batch_size=50,
	                update_rule='adam',
	                optim_config={
	                  'learning_rate': 1e-3,
	                },
	                verbose=True, print_every=200)
	solver.train()

def test_mnist():
	X, Y = get_mnist_data()
	hidden_dims= [100,100,100]
	num_train = 48000
	test_data = {
		'X_train': X[:num_train],
		'y_train': Y[:num_train],
		'X_val': X[len(X)-1500:],
		'y_val': Y[len(Y)-1500:]
	}
	weight_scale = 2e-2
	bn_model = FullyConnectedNet(hidden_dims,input_dim=1*784,weight_scale=weight_scale,use_batchnorm=True)
	bn_solver = Solver(bn_model, test_data,
		num_epochs=20, batch_size=60,
		update_rule='sgd',
		optim_config={
			'learning_rate': 3e-3,
		},
		verbose=True, print_every=200)
	step, train_accuracies, val_accuracies, loss = bn_solver.train()
	return bn_model, step, train_accuracies, val_accuracies, loss

def plot_train(step, train_accuracies, val_accuracies, loss):
	plt.subplot(3,1,1)
	plt.plot(step, train_accuracies)
	plt.title("Training accuracies")

	plt.subplot(3,1,2)
	plt.plot(step, val_accuracies)
	plt.title("Validation accuracies")

	plt.subplot(3,1,3)
	plt.plot(step, loss)
	plt.title("Losses")

	plt.show()


if __name__ == "__main__":
	_, step, train_accuracies, val_accuracies, loss = test_mnist()
	plot_train(step, train_accuracies, val_accuracies, loss)
