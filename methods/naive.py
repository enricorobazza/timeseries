import numpy as np

def average_forecast(train, validation, preprocess, n_steps = 4):
	x = validation.x

	avg_pred = []
	for i in range(len(x)):
		avg = np.average(x[i])
		avg_pred.append(preprocess.classify(x[i][-1], avg))

	return avg_pred, validation.y, validation.labels

def all_equal(train, validation, preprocess, value=1):
	return [value for x in range(len(validation.y))], validation.y, validation.labels

def all_true(train, validation, preprocess):
	return all_equal(train, validation, preprocess, value=1)

def all_false(train, validation, preprocess):
	return all_equal(train, validation, preprocess, value=0)
