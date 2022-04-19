
import random
from collections import deque
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf

def split_sequence(x, y, n_steps, n_features, random_seed = False):
	x_out, y_out = list(), list()
	out = list()
	buffer = deque(maxlen=n_steps)
	for i in range(len(x)):
		buffer.append(x[i])
		if len(buffer) == n_steps:
			out.append([[x for x in buffer], y[i]])

	if random_seed:
		random.shuffle(out)
	else:
		random.Random(42).shuffle(out)

	for i in range(len(out)):
		x_out.append(out[i][0])
		y_out.append(out[i][1])

	x_out, y_out = np.array(x_out), np.array(y_out)

	x_out.reshape((x_out.shape[0], x_out.shape[1], n_features))

	return x_out, y_out
	
def lstm(train_x, train_y, validation_x, validation_y, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0):
	lstm_x, lstm_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_lstm_x, val_lstm_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	model = Sequential()
	model.add(LSTM(50, input_shape=(n_steps, n_features)))

	if classification:
		opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
		model.add(Dense(2, activation='softmax'))
		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
		)
	else:
		model.compile(optimizer='adam', loss='mse')

	model.fit(lstm_x, lstm_y, epochs=epochs, verbose=verbose)

	lstm_prob = model.predict(val_lstm_x, verbose=verbose)
	lstm_pred = lstm_prob.argmax(axis=-1)

	return lstm_pred, val_lstm_y

def stacked_lstm(train_x, train_y, validation_x, validation_y, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=1):
	lstm_x, lstm_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_lstm_x, val_lstm_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	model = Sequential()
	model.add(LSTM(200, input_shape=(n_steps, n_features), kernel_initializer='glorot_uniform', 
	activation = 'relu',
	return_sequences=True))

	# model.add(Dense(200, kernel_initializer='glorot_uniform', activation = 'relu'))

	model.add(LSTM(100, kernel_initializer='glorot_uniform', 
	activation = 'relu',
	return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

	model.add(LSTM(100, kernel_initializer='glorot_uniform', 
	activation = 'relu',
	return_sequences=True))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(LSTM(100, kernel_initializer='glorot_uniform', 
		activation = 'relu'
	))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())


	if classification:
		opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
		model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
		)
	else:
		model.compile(optimizer='adam', loss='mse')

	model.fit(lstm_x, lstm_y, epochs=epochs, verbose=verbose)

	lstm_prob = model.predict(val_lstm_x, verbose=verbose)
	lstm_pred = lstm_prob.argmax(axis=-1)

	return lstm_pred, val_lstm_y


def train_lstm(train_x, train_y, model = None, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0):
	lstm_x, lstm_y = split_sequence(train_x, train_y, n_steps, n_features)

	if model is None:
		model = Sequential()
		model.add(LSTM(50, input_shape=(n_steps, n_features)))

		if classification:
			opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
			model.add(Dense(2, activation='softmax'))
			model.compile(
				loss='sparse_categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy']
			)
		else:
			model.compile(optimizer='adam', loss='mse')

	model.fit(lstm_x, lstm_y, epochs=epochs, verbose=verbose)

	return model

def predict_lstm(validation_x, validation_y, model, n_steps = 4, n_features = 1, verbose=0):
	val_lstm_x, val_lstm_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	lstm_prob = model.predict(val_lstm_x, verbose=verbose)
	lstm_pred = lstm_prob.argmax(axis=-1)

	return lstm_pred, val_lstm_y