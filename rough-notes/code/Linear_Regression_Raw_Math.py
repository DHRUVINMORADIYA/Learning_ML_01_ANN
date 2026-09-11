import math

import random


class DataModule:
	"""Generate synthetic data and prepare train/test batches."""

	def __init__(
		self,
		sample_count=100,
		test_ratio=0.2,
		batch_size=10,
		seed=42,
		true_w1=3.0,
		true_w2=-2.0,
		true_b=5.0,
		noise_std=0.5,
	):
		self.batch_size = batch_size
		self.true_parameters = (true_w1, true_w2, true_b)
		self.random = random.Random(seed)

		data = []
		for _ in range(sample_count):
			x1 = self.random.gauss(0, 1)
			x2 = self.random.gauss(0, 1)
			noise = self.random.gauss(0, noise_std)
			y = x1 * true_w1 + x2 * true_w2 + true_b + noise
			data.append((x1, x2, y))

		self.random.shuffle(data)
		test_count = int(sample_count * test_ratio)
		self.test_data = data[:test_count]
		self.train_data = data[test_count:]

	def train_batches(self):
		"""Yield shuffled training data in mini-batches."""
		shuffled_data = self.train_data[:]
		self.random.shuffle(shuffled_data)

		for start in range(0, len(shuffled_data), self.batch_size):
			yield shuffled_data[start:start + self.batch_size]

	def get_test_data(self):
		return self.test_data


class Model:
	"""Store parameters and perform raw linear-regression math."""

	def __init__(self, w1=0.0, w2=0.0, b=0.0):
		self.w1 = w1
		self.w2 = w2
		self.b = b

	def predict(self, x1, x2):
		return x1 * self.w1 + x2 * self.w2 + self.b

	def loss(self, data):
		total_loss = 0.0

		for x1, x2, y in data:
			prediction = self.predict(x1, x2)
			total_loss += (prediction - y) ** 2

		return total_loss / len(data)

	def train_batch(self, batch, learning_rate):
		d_loss_w1 = 0.0
		d_loss_w2 = 0.0
		d_loss_b = 0.0

		for x1, x2, y in batch:
			prediction = self.predict(x1, x2)
			prediction_error = prediction - y

			# MSE gradients for y_hat = x1*w1 + x2*w2 + b.
			d_loss_w1 += 2 * prediction_error * x1
			d_loss_w2 += 2 * prediction_error * x2
			d_loss_b += 2 * prediction_error

		batch_size = len(batch)
		d_loss_w1 /= batch_size
		d_loss_w2 /= batch_size
		d_loss_b /= batch_size

		# Apply one update using gradients from the same parameter state.
		self.w1 -= learning_rate * d_loss_w1
		self.w2 -= learning_rate * d_loss_w2
		self.b -= learning_rate * d_loss_b

		return self.loss(batch)


class Trainer:
	"""Coordinate training and read-only test evaluation."""

	def __init__(self, model, data_module, learning_rate=0.05, epochs=20):
		self.model = model
		self.data_module = data_module
		self.learning_rate = learning_rate
		self.epochs = epochs

	def evaluate(self, data):
		return self.model.loss(data)

	def fit(self):
		history = []

		for epoch in range(1, self.epochs + 1):
			total_training_loss = 0.0
			batch_count = 0

			for batch in self.data_module.train_batches():
				total_training_loss += self.model.train_batch(
					batch,
					self.learning_rate,
				)
				batch_count += 1

			training_loss = total_training_loss / batch_count
			test_loss = self.evaluate(self.data_module.get_test_data())
			history.append((epoch, training_loss, test_loss))

			print(
				f"Epoch {epoch:02d} | "
				f"training loss: {training_loss:.4f} | "
				f"test loss: {test_loss:.4f}"
			)

		return history


def main():
	data_module = DataModule()
	model = Model(w1=0.0, w2=0.0, b=0.0)
	trainer = Trainer(model, data_module)

	trainer.fit()

	true_w1, true_w2, true_b = data_module.true_parameters
	print("\nLearned parameters:")
	print(f"w1: {model.w1:.4f} (true: {true_w1:.4f})")
	print(f"w2: {model.w2:.4f} (true: {true_w2:.4f})")
	print(f"b:  {model.b:.4f} (true: {true_b:.4f})")


if __name__ == "__main__":
	main()

