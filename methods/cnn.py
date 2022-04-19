
import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf
from .lstm import split_sequence
	
def cnn(train_x, train_y, validation_x, validation_y, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0):
	cnn_x, cnn_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_cnn_x, val_cnn_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam', loss='mse')

	model.fit(cnn_x, cnn_y, epochs=epochs, verbose=verbose)

	cnn_prob = model.predict(val_cnn_x, verbose=verbose)
	cnn_pred = cnn_prob.argmax(axis=-1)

	return cnn_pred, val_cnn_y