from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from .lstm import split_sequence
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras

def mlp(train_x, train_y, validation_x, validation_y, preprocess, n_steps = 4, n_features = 1):
	mlp_x, mlp_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_mlp_x, val_mlp_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	max_iter = 3000
	# max_iter = 300

	model = MLPClassifier(hidden_layer_sizes=(50, 50, 2),
                        max_iter = max_iter, activation = 'relu',
                        solver = 'adam')

	model.fit(mlp_x, mlp_y)

	mlp_pred = model.predict(val_mlp_x)

	return mlp_pred, val_mlp_y

def mlp_keras(train_x, train_y, validation_x, validation_y, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0):
	mlp_x, mlp_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_mlp_x, val_mlp_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	model = Sequential()
	model.add(Dense(200, input_shape=(n_steps,), kernel_initializer='glorot_uniform', activation = 'relu'))
	model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(100, init='glorot_uniform', activation='relu'))
	# model.add(Dense(100, init='glorot_uniform', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
	model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
	model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))

	opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	model.fit(mlp_x, mlp_y, epochs=epochs, verbose=verbose)

	mlp_prob = model.predict(val_mlp_x, verbose=verbose)
	mlp_pred = mlp_prob.argmax(axis=-1)

	return mlp_pred, val_mlp_y
