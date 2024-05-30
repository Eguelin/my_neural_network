import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import accuracy_score

class Neuron_network:

	def __init__(self, n0: int = None, n1: int = None, n2: int = None, W1: np.ndarray = None, b1: float = None, W2: np.ndarray = None, b2: float = None):
		"""
		Constructor method
		Args:
			n0: input dimension
			n1: hidden dimension
			n2: output dimension
			W1: weights 1
			b1: bias 1
			W2: weights 2
			b2: bias 2
		"""
		if (n0 is not None and n0 <= 0 and W1 is None) or (n1 is not None and n1 <= 0 and W2 is None) or (n2 is not None and n2 <= 0 and W2 is None):
			raise ValueError("Error: Input dimension must be greater than 0")

		if (n0 is None and W1 is None) or (n1 is None and W2 is None) or (n2 is None and W2 is None):
			raise ValueError("Error: Input dimension or weights must be provided")

		self.W: list[np.ndarray] = list(range(2))
		self.dW: list[np.ndarray] = list(range(2))
		self.b: list[float] = list(range(2))
		self.db: list[float] = list(range(2))
		self.A: list[np.ndarray] = list(range(2))

		if W1 is None:
			self.W[0] = np.random.randn(n1, n0)
		else:
			self.W[0] = W1

		if b1 is None:
			self.b[0] =  np.zeros((n1, 1))
		else:
			self.b[0] = b1

		if W2 is None:
			self.W[1] = np.random.randn(n2, n1)
		else:
			self.W[1] = W2

		if b2 is None:
			self.b[1] = np.zeros((n2, 1))
		else:
			self.b[1] = b2


	def forward_propagation(self, X: np.ndarray) -> np.ndarray:
		"""
		Forward propagation function
		Args:
			X: input data
		Returns:
			np.ndarray: output data
		"""
		for i in range(len(self.W)):
			X = self.model(X, i)
		return X

	def model(self, X: np.ndarray, layers: int) -> np.ndarray:
		"""
		Model function
		Args:
			X: input data
		Returns:
			float: output data
		"""
		if layers < 0:
			raise ValueError("Error: Number of layers must be greater than 0")
		if self.W[layers].shape[1] != X.shape[0]:
			raise ValueError("Error: Input data and weights must have the same dimension")

		Z: np.ndarray = self.W[layers].dot(X) + self.b[layers]
		self.A[layers] = self.__sigmoid(Z)
		return self.A[layers]


	def __sigmoid(self, Z: np.ndarray) -> np.ndarray:
		"""
		Sigmoid function
		Args:
			Z: input data
		Returns:
			np.ndarray: output data
		"""
		return 1 / (1 + np.exp(-Z))


	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict function for dataset
		Args:
			X: input data
		Returns:
			np.ndarray: output data
		"""
		return (self.forward_propagation(X) > 0.5).astype(int)


	def loss(self, A: np.ndarray, y: np.ndarray) -> float:
		"""
		Loss function
		Args:
			A: output data
			y: reference data
		Returns:
			float: loss value
		"""
		if A.shape != y.shape:
			raise ValueError("Error: Output data and reference data must have the same shape")
		epsilon: float = 1e-15
		return float(1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))


	def __backward_propagation(self, X: np.ndarray, y: np.ndarray) -> None:
		"""
		Backward propagation function
		Args:
			X: input data
			y: reference data
		Returns:
			None
		"""
		if X.shape[1] != y.shape[1]:
			raise ValueError("Error: Input data and reference data must have the same number of samples")

		m = y.shape[1]

		dZ2: np.ndarray = self.A[1] - y
		self.dW[1] = 1 / m * dZ2.dot(self.A[0].T)
		self.db[1] = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

		dZ1: np.ndarray = self.W[1].T.dot(dZ2) * self.A[0] * (1 - self.A[0])
		self.dW[0] = 1 / m * dZ1.dot(X.T)
		self.db[0] = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

	def __update_weights(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
		"""
		Update weights function
		Args:
			X: input data
			y: reference data
			lr: learning rate
		Returns:
			None
		"""
		self.__backward_propagation(X, y)
		for i in range(len(self.W)):
			self.W[i] = self.W[i] - lr * self.dW[i]
			self.b[i] = self.b[i] - lr * self.db[i]

	def train(self, X: np.ndarray, y: np.ndarray, lr: float, epochs: int, X_test: np.ndarray = None, y_test: np.ndarray = None, display: bool = False) -> None:
		"""
		Train function
		Args:
			X: input data
			y: reference data
			lr: learning rate
			epochs: number of epochs
			X_test: test input data
			y_test: test reference data
			display: display flag
		Returns:
			None
		"""
		if display:
			loss: List[float] = []
			acc: List[float] = []
			loss_test: List[float] = []
			acc_test: List[float] = []

		for i in tqdm(range(epochs)):
			A: np.ndarray = self.forward_propagation(X)
			self.__update_weights(X, y, lr)
			if display and i % 10 == 0:
				loss.append(self.loss(A, y))
				acc.append(accuracy_score(y.flatten(), self.predict(X).flatten()))
				if X_test is not None and y_test is not None:
					loss_test.append(self.loss(self.forward_propagation(X_test), y_test))
					acc_test.append(accuracy_score(y_test.flatten(), self.predict(X_test).flatten()))

		if display:
			self.__display(loss, acc, loss_test, acc_test)


	def __display(self, loss: List[float], acc: List[float], loss_test: List[float], acc_test: List[float]) -> None:
		"""
		Display function
		Args:
			loss: loss list
			acc: accuracy list
			loss_test: test loss list
			acc_test: test accuracy list
		"""
		plt.figure(figsize=(12, 4))
		plt.subplot(1, 2, 1)
		plt.plot(loss)
		plt.plot(loss_test)
		plt.title("Loss")
		plt.subplot(1, 2, 2)
		plt.plot(acc)
		plt.plot(acc_test)
		plt.title("Accuracy")
		plt.show()


if __name__ == "__main__":
	from sklearn.datasets import make_circles

	X, y = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=0)
	X = X.T
	y = y.reshape((1, y.shape[0]))

	plt.scatter(X[:, 0], X[:, 1], c=y)
	plt.show()

	neuron = Neuron_network(X.shape[0], 32, 1)
	neuron.train(X, y, 0.1, 1000, display=True)
