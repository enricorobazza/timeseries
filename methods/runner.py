import pandas as pd
import numpy as np
import os 
from .preprocess import Preprocess
from .ar import ar, arima
from .naive import average_forecast, all_true, all_false
from .lstm import lstm, stacked_lstm
from .mlp import mlp
from .mlp import mlp_keras
from .cnn import cnn
import time

from IPython.display import clear_output

pd.options.mode.chained_assignment = None  # default='warn'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Runner:
	def __init__(self, models = None):
		if models is None:
			models = {
				"Naive": average_forecast,
				"All True": all_true,
				"All False": all_false,
				"AR": ar,
				"ARIMA": arima,
				"MLP": mlp,
				"MLP Keras": mlp_keras,
				"CNN": cnn,
				"LSTM": lstm,
				"Stacked LSTM": stacked_lstm
			}
		self.models = models

	def get_df(self, df):
		df = df[["Trade Flow", "Trade Value (US$)"]]
		df.rename(columns={"Trade Value (US$)": "value"}, inplace=True)

		exports = df[df["Trade Flow"] == "Exports"]
		imports = df[df["Trade Flow"] == "Imports"]
		exports.drop(columns=["Trade Flow"], inplace=True)
		imports.drop(columns=["Trade Flow"], inplace=True)

		return exports

	def evaluate(self, models, evaluations, metric = "accuracy"):
		def get_average_evaluation(evaluations, name, metric="accuracy"):
			metric_sum = 0
			metric_count = 0

			right_sum = 0
			total_sum = 0

			for key in evaluations:
				metric_count += 1
				country = evaluations[key]
				acc = country[metric]
				total = country['periods']

				right = acc * total

				total_sum += total
				right_sum += right
				metric_sum += acc


			avg = metric_sum/metric_count if metric_count != 0 else 0
			avg_sample = right_sum/total_sum if total_sum != 0 else 0

			print(f"Average country {metric} for {name}: {avg} for {metric_count} countries")
			print(f"Average sample {metric} for {name}: {avg_sample} for {total_sum} samples\n")

		for model in models:
			get_average_evaluation(evaluations[model], model, metric)

	def run(self, metric = "accuracy"):
		folder = "data/"
		evaluations = {}

		for model in self.models:
			evaluations[model] = {}

		min_validation_size = 4
		num_steps = 4
		validation_split = 0.05
		min_size = min_validation_size * (num_steps - 1) / validation_split

		print(f"Running only for datasets with more then {str(min_size)} rows")

		files = sorted(os.listdir(folder))

		for i, file in enumerate(files):
			df = pd.read_csv(os.path.join(folder, file), index_col='Period').sort_index()

			if df.shape[0] < min_size:
				continue

			clear_output(wait=True)

			self.evaluate(self.models, evaluations, metric)
			print(f"Running {file} ({str(i)}/{str(len(files))}) with shape {str(df.shape)}")

			preprocess = Preprocess()

			df = self.get_df(df)
			df = preprocess.preprocess_df(df)
			df = preprocess.create_future_column(df)

			validation_df = preprocess.pick_validation(df)

			df.dropna(inplace=True)
			validation_df.dropna(inplace=True)

			train_x, train_y = preprocess.separate_xy(df)
			validation_x, validation_y = preprocess.separate_xy(validation_df)

			for model in self.models:
				func = self.models[model]
				pred_y, model_validation_y = func(train_x, train_y, validation_x, validation_y, preprocess)
				if len(pred_y) == 0:
					continue
				evaluations[model][file] = preprocess.evaluate(model_validation_y, pred_y)
				print(model, file, evaluations[model][file])

		clear_output(wait=True)
		time.sleep(1)
		self.evaluate(self.models, evaluations, metric)
