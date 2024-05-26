import numpy as np
from typing import List, Tuple
from sklearn.metrics import accuracy_score

class Neuron:
	"""
	Neuron class
	Args:
		dim: input dimension
	Attributes:
		W: weights
		b: bias
	Attributes (private):
		__dW: weights gradient
		__db: bias gradient
	Methods:
		model_dataset: model function for dataset
		model: model function
		predict_dataset: predict function for dataset
		predict: predict function
		loss: loss function
		train: train function
	Methods (private):
		__gradient: gradient function
		__update_parameters: update parameters function
	"""
	def __init__(self, dim: int, W: np.ndarray = None, b: float = None):
		if dim < 1:
			raise ValueError("Error: Input dimension must be greater than 0")
		if W is None:
			self.W: np.ndarray = np.random.randn(dim, 1) * np.sqrt(2 / dim) # He initialization
		elif W.shape != (dim, 1):
			raise ValueError("Error: Weights must have the same dimension as input data")
		else:
			self.W: np.ndarray = W
		if b is None:
			self.b: float = 0.0
		else:
			self.b: float = b

	def model_dataset(self, X: np.ndarray) -> np.ndarray:
		"""
		Model function for dataset
		Args:
			X: input data
		Returns:
			np.ndarray: output data
		"""
		if X.shape[1] != self.W.shape[0]:
			raise ValueError("Error: Input data and weights must have the same dimension")
		Z: np.ndarray = X.dot(self.W) + self.b
		return 1 / (1 + np.exp(-Z))

	def model(self, X: np.ndarray) -> float:
		"""
		Model function
		Args:
			X: input data
		Returns:
			float: output data
		"""
		X = X.reshape((1, X.shape[0]))
		return float(self.model_dataset(X))

	def predict_dataset(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict function for dataset
		Args:
			X: input data
		Returns:
			np.ndarray: output data
		"""
		return (self.model_dataset(X) > 0.5).astype(int)

	def predict(self, x: np.ndarray) -> int:
		"""
		Predict function
		Args:
			x: input data
		Returns:
			int: output data
		"""
		return int(self.model(x) > 0.5)


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
		return float(1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A)))

	def __gradient(self, X: np.ndarray, A: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
		"""
		Gradient function
		Args:
			X: input data
			A: output data
			y: reference data
		Returns:
			None
		"""
		if X.shape[0] != y.shape[0]:
			raise ValueError("Error: Input data and reference data must have the same number of samples")
		dW: np.ndarray = 1 / len(y) * X.T.dot(A - y)
		db: float = float(1 / len(y) * np.sum(A - y))
		return (dW, db)

	def __update_parameters(self, X: np.ndarray, A: np.ndarray, y: np.ndarray, lr: float) -> None:
		"""
		Update parameters function
		Args:
			lr: learning rate
		Returns:
			None
		"""
		dW, db = self.__gradient(X, A, y)
		self.W = self.W - lr * dW
		self.b = self.b - lr * db

	def train(self, X: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> Tuple[List[float], float]:
		"""
		Train function
		Args:
			X: input data
			y: reference data
			lr: learning rate
			epochs: number of epochs
		Returns:
			Tuple[List[float], float]: loss and accuracy
		"""
		if X.shape[0] != y.shape[0]:
			raise ValueError("Error: Input data and reference data must have the same number of samples")
		loss: List[float] = []
		for i in range(epochs):
			A: np.ndarray = self.model_dataset(X)
			self.__update_parameters(X, A, y,lr)
			loss.append(self.loss(A, y))
		y_pred = self.predict_dataset(X)
		accuracy = accuracy_score(y, y_pred)
		return (loss, accuracy)

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from sklearn.datasets import make_blobs

	X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
	y = y.reshape((y.shape[0], 1))

	# plt.scatter(X[:, 0], X[:, 1], c=y)
	# plt.show()

	neuron = Neuron(X.shape[1])
	loss, accuracy = neuron.train(X, y, 0.1, 1000)

	print("Accuracy:", accuracy)
	plt.plot(loss)
	plt.show()

	test = np.array([1, 4])
	prob = neuron.model(test)
	pre = neuron.predict(test)
	print("Probability:", round(prob, 3))
	print("Prediction:", neuron.predict(test))

	x0 = np.linspace(-1, 5, 100)
	x1 = -(neuron.W[0] * x0 + neuron.b) / neuron.W[1]

	plt.scatter(X[:, 0], X[:, 1], c=y)
	test = test.reshape((1, test.shape[0]))
	plt.scatter(test[:, 0], test[:, 1], c='red')
	plt.plot(x0, x1, c='green')
	plt.show()
