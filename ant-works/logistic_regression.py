import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mglearn
import matplotlib.pyplot as plt
import pickle
from util import Helper


helper = Helper()
(train_data, target) = helper.get_train_data()
print(repr(train_data))
test_data = helper.get_test_data()
print(repr(test_data))
feature_names = helper.get_feature_names()
print("Length features :", len(feature_names))
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)
print(target[:10])
print(label_encoder.classes_)
X_train, X_val, y_train, y_val = train_test_split(train_data, target,
                               test_size = 0.3, random_state = 42,
                               stratify = target)

lr = LogisticRegression()
params = {"C" : [0.0001, 0.001, 0.01, 0.1, 1], "solver" : ["liblinear"]}
model = GridSearchCV(lr, params, cv = 5, verbose = 10)
print("Training the model ...")
model.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(model.best_score_))
print("Best parameters: ", model.best_params_)
print("Best estimator: ", model.best_estimator_)
print("Predicting the trained model ...")
model = model.best_estimator_
pred = model.predict(X_val)
print("Classification Report : ", classification_report(y_val, pred, target_names = label_encoder.classes_))
print("Accuracy Score : ", accuracy_score(y_val, pred))
mglearn.tools.visualize_coefficients(model.coef_, feature_names, n_top_features = 50)
plt.show()
with open("models/lr.pickle", "wb") as f:
    f.write(pickle.dumps(model))
# Generate the submit file
pred = model.predict(test_data)
pred = label_encoder.inverse_transform(pred)
helper.generate_result(pred, suffix = "lr")

