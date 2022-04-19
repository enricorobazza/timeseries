from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def ar(train_x, train_y, validation_x, validation_y, preprocess):
	start = len(train_x)-len(validation_x)
	end = start + len(validation_x)

	model = AutoReg(train_x[:start], 4)
	model_fit = model.fit()
	prediction = model_fit.predict(start, end)
	classified_pred = []

	for i in range(1, len(prediction)):
		classified_pred.append(preprocess.classify(prediction[i], prediction[i-1]))

	return classified_pred, validation_y

def arima(train_x, train_y, validation_x, validation_y, preprocess):
	start = len(train_x)-len(validation_x)
	end = start + len(validation_x)

	model = ARIMA(train_x, order=(4, 1, 0))
	model_fit = model.fit()
	prediction = model_fit.predict(start, end)
	classified_pred = []
	for i in range(1, len(prediction)):
		classified_pred.append(preprocess.classify(prediction[i], prediction[i-1]))

	return classified_pred, validation_y