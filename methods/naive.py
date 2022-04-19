import numpy as np

def average_forecast(train_x, train_y, validation_x, validation_y, preprocess, n_steps = 4):
	x = validation_x

	avg_pred = []
	for i in range(len(x)):
		if i < n_steps:
			avg_pred.append(np.nan)
		else:
			y = sum(x[i-n_steps:i+1])/n_steps
			avg_pred.append(preprocess.classify(x[i], y))

	avg_pred = avg_pred[n_steps:]
	_validation_y = validation_y[n_steps:]

	return avg_pred, _validation_y

def all_equal(train_x, train_y, validation_x, validation_y, preprocess, value=1):
	return [value for x in range(len(validation_y))], validation_y

def all_true(train_x, train_y, validation_x, validation_y, preprocess):
	return all_equal(train_x, train_y, validation_x, validation_y, preprocess, value=1)

def all_false(train_x, train_y, validation_x, validation_y, preprocess):
	return all_equal(train_x, train_y, validation_x, validation_y, preprocess, value=0)
