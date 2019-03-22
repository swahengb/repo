from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from util import Helper
import mglearn
import matplotlib.pyplot as plt


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
                               test_size = 0.2, random_state = 42,
                               stratify = target)

log_reg = LogisticRegression(solver = "liblinear")
rnd_clf = RandomForestClassifier(n_estimators = 100) 
svm_clf = SVC(kernel = "linear", C = 0.1, probability = True)
#svm_clf = SVC(kernel = "linear", C = 0.1)
model = VotingClassifier(estimators = [('lr', log_reg), 
                                         ('rf', rnd_clf), 
                                         ('svc', svm_clf)
                                        ], voting = 'soft') 
print("Training voting classifier ...")
model.fit(X_train, y_train)
pred = model.predict(X_val)
print("Classification Report : ", classification_report(y_val, pred, target_names = label_encoder.classes_))
print("Accuracy Score : ", accuracy_score(y_val, pred))
# Generate the submit file
pred = model.predict(test_data)
pred = label_encoder.inverse_transform(pred)
helper.generate_result(pred, suffix = "ens")
#mglearn.tools.visualize_coefficients(model.coef_, feature_names, n_top_features = 50)
plt.show()
