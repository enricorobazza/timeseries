
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf

def lstm(train, validation, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0, cells = 50, layers = 1):
	lstm_dataset = train.split_sequence(n_steps, n_features)
	val_lstm_dataset = validation.split_sequence(n_steps, n_features)

	model = Sequential()

	if layers == 1:
		model.add(LSTM(cells, input_shape=(n_steps, n_features)))
	else:
		model.add(LSTM(cells, input_shape=(n_steps, n_features), return_sequences = True))

	if layers > 2:
		model.add(LSTM(cells, kernel_initializer='glorot_uniform', 
			activation = 'relu',
			return_sequences=True))
		model.add(Dropout(0.2))
		model.add(BatchNormalization())

	if layers > 1:
		model.add(LSTM(cells, kernel_initializer='glorot_uniform', 
			activation = 'relu'
		))
		model.add(Dropout(0.1))
		model.add(BatchNormalization())

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

	model.fit(lstm_dataset.x, lstm_dataset.y, epochs=epochs, verbose=verbose)

	lstm_prob = model.predict(val_lstm_dataset.x, verbose=verbose)
	lstm_pred = lstm_prob.argmax(axis=-1)

	return lstm_pred, val_lstm_dataset.y, val_lstm_dataset.labels

def stacked_lstm(train, validation, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=1):
	lstm_dataset = train.split_sequence(n_steps, n_features)
	val_lstm_dataset = validation.split_sequence(n_steps, n_features)

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

	model.fit(lstm_dataset.x, lstm_dataset.y, epochs=epochs, verbose=verbose)

	lstm_prob = model.predict(val_lstm_dataset.x, verbose=verbose)
	lstm_pred = lstm_prob.argmax(axis=-1)

	return lstm_pred, val_lstm_dataset.y, val_lstm_dataset.labels


# def train_lstm(train_x, train_y, model = None, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0):
# 	lstm_x, lstm_y = split_sequence(train_x, train_y, n_steps, n_features)

# 	if model is None:
# 		model = Sequential()
# 		model.add(LSTM(50, input_shape=(n_steps, n_features)))

# 		if classification:
# 			opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
# 			model.add(Dense(2, activation='softmax'))
# 			model.compile(
# 				loss='sparse_categorical_crossentropy',
# 				optimizer=opt,
# 				metrics=['accuracy']
# 			)
# 		else:
# 			model.compile(optimizer='adam', loss='mse')

# 	model.fit(lstm_x, lstm_y, epochs=epochs, verbose=verbose)

# 	return model

# def predict_lstm(validation_x, validation_y, model, n_steps = 4, n_features = 1, verbose=0):
# 	val_lstm_x, val_lstm_y = split_sequence(validation_x, validation_y, n_steps, n_features)

# 	lstm_prob = model.predict(val_lstm_x, verbose=verbose)
# 	lstm_pred = lstm_prob.argmax(axis=-1)

# 	return lstm_pred, val_lstm_y