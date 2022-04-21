import numpy as np

def average_forecast(train, validation, preprocess, n_steps = 4):
	x = validation.x

	avg_pred = []
	for i in range(len(x)):
		if i < n_steps:
			avg_pred.append(np.nan)
		else:
			y = sum(x[i-n_steps:i+1])/n_steps
			avg_pred.append(preprocess.classify(x[i], y))

	avg_pred = avg_pred[n_steps:]
	_validation_y = validation.y[n_steps:]
	_validation_labels = validation.labels[n_steps:]

	return avg_pred, _validation_y, _validation_labels

def all_equal(train, validation, preprocess, value=1):
	return [value for x in range(len(validation.y))], validation.y, validation.labels

def all_true(train, validation, preprocess):
	return all_equal(train, validation, preprocess, value=1)

def all_false(train, validation, preprocess):
	return all_equal(train, validation, preprocess, value=0)
