import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

#%matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def toFloat(lst):
	for i in range(len(lst)):
		lst[i] = float(lst[i])
	return lst


def get_mnist_data(file_name,num):
	with open(file_name,'rb') as train_file:
		train_reader = csv.reader(train_file,delimiter=',',quotechar='|')
		i = 0
		train_X = np.zeros((num,784))
		train_Y = np.zeros(num,dtype=int)
		for row in train_reader:
			i += 1
			if i>=num: break
			floatRow = toFloat(row) # len = 785
			train_X[i] = np.array(floatRow[1:])
			train_Y[i] = np.array(int(floatRow[0]))
		train_Y = train_Y.T
	return train_X, train_Y


def test_mnist(num_epochs=60,batch_size=60,learning_rate=3e-3):
	X_train, y_train = get_mnist_data('mnist_train.csv',50000)
	X_val, y_val = get_mnist_data('mnist_test.csv',10000)
	hidden_dims= [100,100,100]
	# num_train = 48000
	test_data = {
		'X_train': X_train,
		'y_train': y_train,
		'X_val': X_val,
		'y_val': y_val
	}
	weight_scale = 2e-2
	bn_model = FullyConnectedNet(hidden_dims,input_dim=1*784,weight_scale=weight_scale,use_batchnorm=True)
	bn_solver = Solver(bn_model, test_data,
		num_epochs=num_epochs, batch_size=batch_size,
		update_rule='sgd',
		optim_config={
			'learning_rate': learning_rate,
		},
		verbose=True, print_every=400)
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

def write_train_result(step, train_accuracies, val_accuracies, loss):
	file = open("Output.txt",'w')
	file.write("Step:\n")
	file.write(str(step)+'\n')
	file.write("Train Accuracies:\n")
	file.write(str(train_accuracies)+'\n')
	file.write("Validation Accuracies:\n")
	file.write(str(val_accuracies)+'\n')
	file.write("Loss:\n")
	file.write(str(loss)+'\n')
	file.close()

def process_output(output):
	file = open(output,'r')
	l = []
	for row in file:
		l.append(row)
	l[1], l[3], l[5], l[7] = l[1][1:-2], l[3][1:-2], l[5][1:-2], l[7][1:-2]
	# print type(l[1]), type(l[3]), type(l[5]), type(l[7])

	l[1], l[3], l[5], l[7] = l[1].split(','), l[3].split(','), l[5].split(','), l[7].split(',')
	f = lambda lst: map(float,lst)
	step, train_accuracies, val_accuracies, loss = f(l[1]), f(l[3]), f(l[5]), f(l[7])
	file.close()
	plot_train(step, train_accuracies, val_accuracies, loss)



if __name__ == "__main__":
	# _, step, train_accuracies, val_accuracies, loss = test_mnist(num_epochs=60)
	# write_train_result(step, train_accuracies, val_accuracies, loss)
	process_output('Output.txt')


