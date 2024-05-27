import numpy as np
import matplotlib.pyplot as plt
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
			self.W: np.ndarray = np.random.randn(dim, 1) * np.sqrt(2 / dim) # He initialization
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
		s = 1e-8
		return float(1 / len(y) * np.sum(-y * np.log(A + s) - (1 - y) * np.log(1 - A + s)))


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
			display: display flag
			X_test: test input data
			y_test: test reference data
		Returns:
			None
		"""
		if X.shape[0] != y.shape[0] or (X_test is not None and y_test is not None and X_test.shape[0] != y_test.shape[0] != X.shape[0]):
			raise ValueError("Error: Input data and reference data must have the same number of samples")

		loss: List[float] = []
		loss_test: List[float] = []

		for i in range(epochs):
			A: np.ndarray = self.model_dataset(X)
			self.__update_weights(X, A, y, lr)
			loss.append(self.loss(A, y))
			if display and X_test is not None and y_test is not None:
				A_test = self.model_dataset(X_test)
				loss_test.append(self.loss(A_test, y_test))

		if display:
			self.__display(X, y, X_test, y_test, loss, loss_test)


	def __display(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, loss: List[float], loss_test: List[float]) -> None:
		"""
		Display function
		Args:
			X: input data
			y: reference data
			X_test: test input data
			y_test: test reference data
			loss: loss list
			loss_test: test loss list
		Returns:
			None
		"""
		y_pred = self.predict_dataset(X)
		accuracy = accuracy_score(y, y_pred)
		print("Accuracy:", accuracy)

		if X_test is not None and y_test is not None:
			y_pred_test = self.predict_dataset(X_test)
			accuracy_test = accuracy_score(y_test, y_pred_test)
			print("Accuracy test:", accuracy_test)
			plt.plot(loss_test, c='orange')

		plt.plot(loss)
		plt.title("Loss")
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
