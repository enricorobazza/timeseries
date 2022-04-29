import pandas as pd
import numpy as np
import os 
from .preprocess import Preprocess
from .ar import ar, arima
from .naive import average_forecast, all_true, all_false
from .lstm import lstm
from .mlp import mlp
from .mlp import mlp_keras
from .cnn import cnn
from .dataset import Dataset
import time
import datetime

from IPython.display import clear_output

pd.options.mode.chained_assignment = None  # default='warn'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Runner:
	def __init__(self, models, callback = None):
		self.models = models
		self.callback = callback

	def get_df(self, df):
		df = df[["Trade Flow", "Trade Value (US$)"]]
		df.rename(columns={"Trade Value (US$)": "value"}, inplace=True)

		exports = df[df["Trade Flow"] == "Exports"]
		imports = df[df["Trade Flow"] == "Imports"]
		exports.drop(columns=["Trade Flow"], inplace=True)
		imports.drop(columns=["Trade Flow"], inplace=True)

		return exports

	def save_result(self, file, model, result, run_ts):
		results_folder = self.get_folder("results")
		csv_file = os.path.join(results_folder, f"{run_ts}.csv")
		file_exists = os.path.exists(csv_file)
		csv = open(csv_file, "a")

		if not file_exists:
			csv.write("y,y_pred,y_label,time,file,model\n")

		y = result["true"]
		y_pred = result["pred"]
		y_labels = result["labels"]
		time = result["time"]
		avg_time = time / len(y)
		for i in range(len(y)):
			csv.write(f"{y[i]},{y_pred[i]},{y_labels[i]},{avg_time},\"{file}\",\"{model}\"\n")

		csv.close()

	def evaluate(self, preprocess, results, metric = "accuracy"):
		for model in results:
			y = results[model]["true"]
			y_pred = results[model]["pred"]
			metrics = preprocess.evaluate(y, y_pred)
			print(f"Average sample {metric} for {model}: {metrics[metric]} for {len(y_pred)} samples\n")

	def get_folder(self, folder):
		script_path = os.path.realpath(__file__)
		parent = os.path.abspath(os.path.join(script_path, os.pardir))
		parent = os.path.abspath(os.path.join(parent, os.pardir))
		return os.path.join(parent, folder)

	def continue_run(self, run_ts, metric = "accuracy"):
		results_folder = self.get_folder("results")
		csv_file = os.path.join(results_folder, f"{run_ts}.csv")

		if not os.path.exists(csv_file):
			print("Run doesnt exist")
			return

		df = pd.read_csv(csv_file)

		if df.shape[0] == 0:
			print("Run is empty")
			return

		last = df.tail(1).iloc[0]

		if last["model"] not in self.models:
			print("Inconsistent models")
			return

		self.run(metric, last, run_ts)

	def run_all(self, metric = "accuracy", last = None, run_ts = None):
		folder = self.get_folder("data")
		evaluations = {}
		results = {}
		self.errors = []

		min_validation_size = 4
		num_steps = 4
		validation_split = 0.05
		min_size = min_validation_size * (num_steps - 1) / validation_split

		for model in self.models:
			evaluations[model] = {}
			results[model] = {
				"true": [],
				"pred": [],
				"labels": [],
				"files": {}
			}

		files = sorted(os.listdir(folder))

		preprocess = Preprocess()

		join_train_x, join_train_y, join_train_labels = [], [], []
		join_validation_x, join_validation_y, join_validation_labels = [], [], []

		for i, file in enumerate(files):
			df = pd.read_csv(os.path.join(folder, file), index_col='Period').sort_index()

			if df.shape[0] < min_size:
				continue

			df = self.get_df(df)
			df = preprocess.preprocess_df(df)
			df = preprocess.create_future_column(df)

			validation_df = preprocess.pick_validation(df)
			df = preprocess.pick_train(df)

			df.dropna(inplace=True)
			validation_df.dropna(inplace=True)

			train_x, train_y, train_labels = preprocess.separate_xy(df)
			validation_x, validation_y, validation_labels = preprocess.separate_xy(validation_df)

			join_train_x += train_x
			join_train_y += train_y
			join_train_labels += train_labels

			join_validation_x += validation_x
			join_validation_y += validation_y
			join_validation_labels += validation_labels

		train = Dataset(join_train_x, join_train_y, join_train_labels)
		validation = Dataset(join_validation_x, join_validation_y, join_validation_labels)

		for model in self.models:
			func = self.models[model]

			pred_y, model_validation_y, model_validation_labels = None, None, None

			start = time.time()

			try:
				pred_y, model_validation_y, model_validation_labels = func(train, validation, preprocess)
			except Exception as E:
				self.errors += [[model, "All.csv", E]]
				continue

			end = time.time()

			if len(pred_y) == 0:
				continue

			evaluations = preprocess.evaluate(model_validation_y, pred_y)

			results[model]["true"] += list(model_validation_y)
			results[model]["pred"] += list(pred_y)

			result = {
				"true": list(model_validation_y),
				"pred": list(pred_y),
				"labels": list(model_validation_labels),
				"time": end - start
			}

			self.save_result("All.csv", model, result, run_ts)

			if self.callback is not None:
				self.callback()

			print(model, "All.csv", evaluations)

	def run(self, metric = "accuracy", last = None, run_ts = None):
		
		folder = self.get_folder("data")
		evaluations = {}
		results = {}
		self.errors = []

		for model in self.models:
			evaluations[model] = {}
			results[model] = {
				"true": [],
				"pred": [],
				"labels": [],
				"files": {}
			}

		min_validation_size = 4
		num_steps = 4
		validation_split = 0.05
		min_size = min_validation_size * (num_steps - 1) / validation_split

		print(f"Running only for datasets with more then {str(min_size)} rows")

		files = sorted(os.listdir(folder))

		self.total_countries = len(files)
		self.current_country = 0

		if run_ts is None:
			run_ts = datetime.datetime.now().strftime("%Y%m%d%H%M")

		self.run_ts = run_ts

		for i, file in enumerate(files):

			self.current_country += 1

			if last is not None:
				if file != last["file"]:
					continue

			df = pd.read_csv(os.path.join(folder, file), index_col='Period').sort_index()

			if df.shape[0] < min_size:
				continue

			# if file != "Brazil.csv":
			# 	continue

			clear_output(wait=True)

			preprocess = Preprocess()

			self.evaluate(preprocess, results, metric)
			print(f"Running {file} ({str(i)}/{str(len(files))}) with shape {str(df.shape)}")

			df = self.get_df(df)
			df = preprocess.preprocess_df(df)
			df = preprocess.create_future_column(df)

			validation_df = preprocess.pick_validation(df)
			df = preprocess.pick_train(df)

			df.dropna(inplace=True)
			validation_df.dropna(inplace=True)

			train_x, train_y, train_labels = preprocess.separate_xy(df)
			validation_x, validation_y, validation_labels = preprocess.separate_xy(validation_df)

			train = Dataset(train_x, train_y, train_labels)
			validation = Dataset(validation_x, validation_y, validation_labels)

			for model in self.models:
				if last is not None:
					if model != last["model"]:
						continue

				# starting from where stopped
				last = None

				func = self.models[model]

				pred_y, model_validation_y, model_validation_labels = None, None, None

				start = time.time()

				try:
					pred_y, model_validation_y, model_validation_labels = func(train, validation, preprocess)
				except Exception as E:
					self.errors += [[model, file, E]]
					continue

				end = time.time()

				if len(pred_y) == 0:
					continue

				evaluations = preprocess.evaluate(model_validation_y, pred_y)

				results[model]["true"] += list(model_validation_y)
				results[model]["pred"] += list(pred_y)

				result = {
					"true": list(model_validation_y),
					"pred": list(pred_y),
					"labels": list(model_validation_labels),
					"time": end - start
				}

				self.save_result(file, model, result, run_ts)

				print(model, file, evaluations)
			
			if self.callback is not None:
				self.callback()

		clear_output(wait=True)
		time.sleep(1)
		self.evaluate(preprocess, results, metric)
		print("Errors: %d"%(len(self.errors)))

	def run_transfer(self, metric = "accuracy"):
		folder = self.get_folder("data")
		evaluations = {}
		results = {}
		self.saved_models = {}
		self.errors = []
		validations = {}

		for model in self.models:
			evaluations[model] = {}
			results[model] = {
				"true": [],
				"pred": [],
				"labels": [],
				"files": {}
			}
			self.saved_models[model] = None

		min_validation_size = 4
		num_steps = 4
		validation_split = 0.05
		min_size = min_validation_size * (num_steps - 1) / validation_split

		print(f"Running only for datasets with more then {str(min_size)} rows")

		files = sorted(os.listdir(folder))

		run_ts = datetime.datetime.now().strftime("%Y%m%d%H%M")

		preprocess = Preprocess()

		self.run_ts = run_ts

		for i, file in enumerate(files):

			df = pd.read_csv(os.path.join(folder, file), index_col='Period').sort_index()

			if df.shape[0] < min_size:
				continue

			if file != "Brazil.csv":
				continue

			clear_output(wait=True)

			print(f"Trainning {file} ({str(i)}/{str(len(files))}) with shape {str(df.shape)}")

			df = self.get_df(df)
			df = preprocess.preprocess_df(df)
			df = preprocess.create_future_column(df)

			validation_df = preprocess.pick_validation(df)
			df = preprocess.pick_train(df)

			df.dropna(inplace=True)
			validation_df.dropna(inplace=True)

			train_x, train_y, train_labels = preprocess.separate_xy(df)
			validation_x, validation_y, validation_labels = preprocess.separate_xy(validation_df)

			train = Dataset(train_x, train_y, train_labels)
			validation = Dataset(validation_x, validation_y, validation_labels)
			validations[file] = validation

			for model in self.models:
				# starting from where stopped

				func = self.models[model]

				start = time.time()

				try:
					self.saved_models[model] = func(train, validation, preprocess, to_validate = False, model = self.saved_models[model])
				except Exception as E:
					self.errors += [[model, file, E]]
					continue

				end = time.time()
			
			if self.callback is not None:
				self.callback()

		for file in validations:
			validation = validations[file]

			clear_output(wait=True)
			self.evaluate(preprocess, results, metric)

			for model in self.models:
				func = self.models[model]
				saved_model = self.saved_models[model]

				pred_y, model_validation_y, model_validation_labels = func(train, validation, preprocess, model = saved_model, to_train = False)

				result = {
					"true": list(model_validation_y),
					"pred": list(pred_y),
					"labels": list(model_validation_labels),
					"time": end - start
				}

				results[model]["true"] += list(model_validation_y)
				results[model]["pred"] += list(pred_y)

				self.save_result(file, model, result, run_ts)

		clear_output(wait=True)
		self.evaluate(preprocess, results, metric)