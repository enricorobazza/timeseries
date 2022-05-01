
import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf
	
def cnn(train, validation, preprocess, classification = True, n_steps = 4, n_features = 1, filters = 64, kernel_size = 2, epochs = 100, verbose=0):
	cnn_dataset = train.split_sequence(n_steps, n_features)
	val_cnn_dataset = validation.split_sequence(n_steps, n_features)

	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_steps, n_features)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam', loss='mse')

	model.fit(cnn_dataset.x, cnn_dataset.y, epochs=epochs, verbose=verbose)

	cnn_prob = model.predict(val_cnn_dataset.x, verbose=verbose)
	cnn_pred = cnn_prob.argmax(axis=-1)

	return cnn_pred, val_cnn_dataset.y, val_cnn_dataset.labels