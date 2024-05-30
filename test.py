from Neuron_network import Neuron_network
from utilities import load_data
import numpy as np

if __name__ == "__main__":
	X_train, y_train, X_test, y_test = load_data()
	# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

	X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
	X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
	X_train_reshape = X_train_reshape.T
	X_test_reshape = X_test_reshape.T
	y_train = y_train.T
	y_test = y_test.T
	# print(X_train_reshape.shape, y_train.shape, X_test_reshape.shape, y_test.shape)

	neuron = Neuron_network(X_train_reshape.shape[0], 32, 1)
	neuron.train(X_train_reshape, y_train, 0.01, 10000, X_test_reshape, y_test, display=True)
