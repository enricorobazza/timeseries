import random
from collections import deque
import numpy as np

class Dataset:
	def __init__(self, x, y, labels):
		self.x = x
		self.y = y
		self.labels = labels

	def split_sequence(self, n_steps, n_features, shuffle=False, random_seed = False):
		x_out, y_out, labels_out = list(), list(), list()
		out = list()
		buffer = deque(maxlen=n_steps)
		for i in range(len(self.x)):
			buffer.append(self.x[i])
			if len(buffer) == n_steps:
				out.append([[x for x in buffer], self.y[i], self.labels[i]])

		if shuffle:
			if random_seed:
				random.shuffle(out)
			else:
				random.Random(42).shuffle(out)

		for i in range(len(out)):
			x_out.append(out[i][0])
			y_out.append(out[i][1])
			labels_out.append(out[i][2])

		x_out, y_out, labels_out = np.array(x_out), np.array(y_out), np.array(labels_out)

		x_out.reshape((x_out.shape[0], x_out.shape[1], n_features))

		return Dataset(x_out, y_out, labels_out)

	def __add__(self, b):
		if len(self.x) == 0:
			return b

		if len(b.x) == 0:
			return self

		x = np.concatenate((self.x, b.x))
		y = np.concatenate((self.y, b.y))
		labels = np.concatenate((self.labels, b.labels))

		return Dataset(x, y, labels)


	def shuffle(self):
		observations = []
		for i in range(len(self.x)):
			observation = [self.x[i], self.y[i], self.labels[i]]
			observations.append(observation)
		
		random.Random(42).shuffle(observations)

		x = []
		y = []
		labels = []

		for observation in observations:
			x.append(observation[0])
			y.append(observation[1])
			labels.append(observation[2])

		return Dataset(np.array(x), np.array(y), np.array(labels))