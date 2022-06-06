from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score

def mlp(train, validation, preprocess, n_steps = 4, n_features = 1):
	# mlp_dataset = train.split_sequence(n_steps, n_features)
	# val_mlp_dataset = validation.split_sequence(n_steps, n_features)
	mlp_dataset = train
	val_mlp_dataset = validation

	max_iter = 3000
	# max_iter = 300

	model = MLPClassifier(hidden_layer_sizes=(50, 50, 2),
                        max_iter = max_iter, activation = 'relu',
                        solver = 'adam')

	model.fit(mlp_dataset.x, mlp_dataset.y)

	mlp_pred = model.predict(val_mlp_dataset.x)

	return mlp_pred, val_mlp_dataset.y, val_mlp_dataset.labels

def mlp_keras(train, validation, preprocess, classification = True, n_steps = 4, n_features = 1, epochs = 100, verbose=0, layers = 3, cells = 100):
	# mlp_dataset = train.split_sequence(n_steps, n_features)
	# val_mlp_dataset = validation.split_sequence(n_steps, n_features)
	mlp_dataset = train
	val_mlp_dataset = validation

	model = Sequential()
	model.add(Dense(200, input_shape=(n_steps,), kernel_initializer='glorot_uniform', activation = 'relu'))

	if layers > 0:
		model.add(Dense(cells, kernel_initializer='glorot_uniform', activation='relu'))

	model.add(Dropout(0.2))

	if layers > 1:
		model.add(Dense(cells, kernel_initializer='glorot_uniform', activation='relu'))
	if layers > 2:
		model.add(Dense(cells, kernel_initializer='glorot_uniform', activation='relu'))

	model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))

	opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	model.fit(mlp_dataset.x, mlp_dataset.y, epochs=epochs, verbose=verbose)

	mlp_prob = model.predict(val_mlp_dataset.x, verbose=verbose)
	mlp_pred = mlp_prob.argmax(axis=-1)

	return mlp_pred, val_mlp_dataset.y, val_mlp_dataset.labels
