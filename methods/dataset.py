import random
from collections import deque
import numpy as np

class Dataset:
	def __init__(self, x, y, labels):
		self.x = x
		self.y = y
		self.labels = labels

	def split_sequence(self, n_steps, n_features, random_seed = False):
		x_out, y_out, labels_out = list(), list(), list()
		out = list()
		buffer = deque(maxlen=n_steps)
		for i in range(len(self.x)):
			buffer.append(self.x[i])
			if len(buffer) == n_steps:
				out.append([[x for x in buffer], self.y[i], self.labels[i]])

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