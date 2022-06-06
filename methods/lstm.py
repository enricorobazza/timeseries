
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import tensorflow as tf

def lstm(train, validation, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0, cells = 50, layers = 1, to_train = True, to_validate = True, model = None, weights_file = None, n_classes = 2):
	# lstm_dataset = train.split_sequence(n_steps, n_features)
	# val_lstm_dataset = validation.split_sequence(n_steps, n_features)
	lstm_dataset = train
	val_lstm_dataset = validation
	mapping = None

	if n_classes > 2:
		lstm_dataset, val_lstm_dataset, mapping = preprocess.to_categorical(lstm_dataset, val_lstm_dataset, n_classes)

	if model is None:
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
			model.add(Dense(n_classes, activation='softmax'))

			if weights_file is not None:
				model.load_weights(weights_file)

			loss = 'sparse_categorical_crossentropy'
			if n_classes != 2:
				loss = 'categorical_crossentropy'

			model.compile(
				loss=loss,
				optimizer=opt,
				metrics=['accuracy']
			)
		else:
			if weights_file is not None:
				model.load_weights(weights_file)
				
			model.compile(optimizer='adam', loss='mse')

	if to_train:
		model.fit(lstm_dataset.x, lstm_dataset.y, epochs=epochs, verbose=verbose)

	lstm_pred = None
	if to_validate:
		lstm_prob = model.predict(val_lstm_dataset.x, verbose=verbose)
		lstm_pred = lstm_prob.argmax(axis=-1)

	if not to_validate:
		return model

	if n_classes > 2:
		val_lstm_dataset = preprocess.from_categorical(val_lstm_dataset, mapping)

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