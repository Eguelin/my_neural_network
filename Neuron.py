import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
	Methods:
		model_dataset: model function for dataset
		model: model function
		predict_dataset: predict function for dataset
		predict: predict function
		loss: loss function
		train: train function
	"""


	def __init__(self, dim: int = None, W: np.ndarray = None, b: float = None):
		"""
		Constructor method
		Args:
			dim: input dimension
			W: weights
			b: bias
		"""
		if dim is not None and dim <= 0 and W is None:
			raise ValueError("Error: Input dimension must be greater than 0")

		if dim is None and W is None:
			raise ValueError("Error: Input dimension or weights must be provided")

		if W is None:
			self.W: np.ndarray = np.random.randn(dim, 1)
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
		epsilon: float = 1e-15
		return float(1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))


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


	def __update_weights(self, X: np.ndarray, A: np.ndarray, y: np.ndarray, lr: float) -> None:
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
		if X.shape[0] != y.shape[0] or (X_test is not None and y_test is not None and X_test.shape[0] != y_test.shape[0] != X.shape[0]):
			raise ValueError("Error: Input data and reference data must have the same number of samples")

		if display:
			loss: List[float] = []
			acc: List[float] = []
			loss_test: List[float] = []
			acc_test: List[float] = []

		for i in tqdm(range(epochs)):
			A: np.ndarray = self.model_dataset(X)
			self.__update_weights(X, A, y, lr)
			if display and i % 10 == 0:
				loss.append(self.loss(A, y))
				acc.append(accuracy_score(y, self.predict_dataset(X)))
				if X_test is not None and y_test is not None:
					loss_test.append(self.loss(self.model_dataset(X_test), y_test))
					acc_test.append(accuracy_score(y_test, self.predict_dataset(X_test)))

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
	from sklearn.datasets import make_blobs

	X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
	y = y.reshape((y.shape[0], 1))

	# plt.scatter(X[:, 0], X[:, 1], c=y)
	# plt.show()

	neuron = Neuron(X.shape[1])
	neuron.train(X, y, 0.1, 1000, display=True)

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
