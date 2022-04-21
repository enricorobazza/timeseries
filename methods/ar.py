from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def ar(train, validation, preprocess, n_steps = 4):
	start = len(train.x)-len(validation.x)
	end = start + len(validation.x)

	model = AutoReg(train.x[:start], n_steps)
	model_fit = model.fit()
	prediction = model_fit.predict(start, end)
	classified_pred = []

	for i in range(1, len(prediction)):
		classified_pred.append(preprocess.classify(prediction[i], prediction[i-1]))

	return classified_pred, validation.y, validation.labels

def arima(train, validation, preprocess, order = 4, n_steps = 4, differencing = 1):
	start = len(train.x)-len(validation.x)
	end = start + len(validation.x)

	model = ARIMA(train.x, order=(order, differencing, n_steps))
	model_fit = model.fit()
	prediction = model_fit.predict(start, end)
	classified_pred = []
	for i in range(1, len(prediction)):
		classified_pred.append(preprocess.classify(prediction[i], prediction[i-1]))

	return classified_pred, validation.y, validation.labels