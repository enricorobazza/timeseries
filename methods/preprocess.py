from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score

class Preprocess():
	def __init__(self, future_period_predict = 1, classification = True):
		self.future_period_predict = future_period_predict
		self.classification = classification

	def classify(self, current, future):
		if not self.classification:
			return future
		if float(future) > float(current):
			return 1
		return 0

	def preprocess_df(self, df, derivs = 1):
		for i in range(derivs):
			df = df.pct_change()
		for col in df.columns:
			df[col] = preprocessing.scale(df[col].values)
		df.dropna(inplace=True)
		return df

	def create_future_column(self, df):
		df["future"] = df["value"].shift(-self.future_period_predict)
		df["label"] = df.index + self.future_period_predict
		df["target"] = list(map(self.classify, df["value"], df["future"]))
		df.drop(columns=['future'], inplace=True)

		return df

	def pick_validation(self, df):
		return df.iloc[-int(df.shape[0]*0.05):]

	def pick_train(self, df):
		return df.iloc[:-int(df.shape[0]*0.05)]

	def separate_xy(self, df):
		x = df["value"].tolist()
		y = df["target"].tolist()
		labels = df["label"].tolist()

		return x, y, labels

	def evaluate(self, y, y_pred):
		mse = 0
		accuracy = 0
		fmeasure = 0
		roc = 0
		periods = len(y)

		if periods == 0:
			return {"mse": mse, "accuracy": accuracy, "roc": roc, "fmeasure": fmeasure, "periods": periods}
		
		if not self.classification:
			mse = mean_squared_error(y, y_pred)
		else:
			try:
				accuracy = accuracy_score(y, y_pred)
			except:
				pass
			try:
				roc = roc_auc_score(y, y_pred)
			except:
				pass
			try:
				fmeasure = f1_score(y, y_pred)
			except:
				pass
			
		return {"mse": mse, "accuracy": accuracy, "roc": roc, "fmeasure": fmeasure, "periods": periods}
		

		