from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def generic_classifier(model_to_use, train, validation, n_steps = 4, n_features = 1):
	model_dataset = train.split_sequence(n_steps, n_features)
	val_model_dataset = validation.split_sequence(n_steps, n_features)

	max_iter = 3000
	model = None

	if model_to_use == "KNN":
		model = KNeighborsClassifier(n_neighbors=2)
	elif model_to_use == "Gaussian":
		model = GaussianNB()
	elif model_to_use == "RandomForest":
		model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

	model.fit(model_dataset.x, model_dataset.y)

	model_pred = model.predict(val_model_dataset.x)

	return model_pred, val_model_dataset.y, val_model_dataset.labels

def knn(train, validation, preprocess, n_steps = 4, n_features = 1, n_neighbors = 2, distance = 2):
	model_dataset = train.split_sequence(n_steps, n_features)
	val_model_dataset = validation.split_sequence(n_steps, n_features)

	model = KNeighborsClassifier(n_neighbors=n_neighbors, p=distance)
	model.fit(model_dataset.x, model_dataset.y)
	model_pred = model.predict(val_model_dataset.x)

	return model_pred, val_model_dataset.y, val_model_dataset.labels

def gaussian(train, validation, preprocess, n_steps = 4, n_features = 1):
	return generic_classifier("Gaussian", train, validation, n_steps, n_features)

def random_forest(train, validation, preprocess, n_steps = 4, n_features = 1):
	return generic_classifier("RandomForest", train, validation, n_steps, n_features)