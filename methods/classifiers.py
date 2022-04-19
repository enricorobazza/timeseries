from methods.lstm import split_sequence
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def generic_classifier(model_to_use, train_x, train_y, validation_x, validation_y, n_steps = 4, n_features = 1):
	model_x, model_y = split_sequence(train_x, train_y, n_steps, n_features)
	val_model_x, val_model_y = split_sequence(validation_x, validation_y, n_steps, n_features)

	max_iter = 3000
	model = None

	if model_to_use == "KNN":
		model = KNeighborsClassifier(n_neighbors=2)
	elif model_to_use == "Gaussian":
		model = GaussianNB()
	elif model_to_use == "RandomForest":
		model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

	model.fit(model_x, model_y)

	model_pred = model.predict(val_model_x)

	return model_pred, val_model_y

def knn(train_x, train_y, validation_x, validation_y, preprocess, n_steps = 4, n_features = 1):
	return generic_classifier("KNN", train_x, train_y, validation_x, validation_y, n_steps, n_features)

def gaussian(train_x, train_y, validation_x, validation_y, preprocess, n_steps = 4, n_features = 1):
	return generic_classifier("Gaussian", train_x, train_y, validation_x, validation_y, n_steps, n_features)

def random_forest(train_x, train_y, validation_x, validation_y, preprocess, n_steps = 4, n_features = 1):
	return generic_classifier("RandomForest", train_x, train_y, validation_x, validation_y, n_steps, n_features)