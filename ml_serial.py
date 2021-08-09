# import the library
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score # 正解率(accuracy)
from sklearn.metrics import precision_score # 適合率(precision)
from sklearn.metrics import recall_score # 検出率(recall)
from sklearn.metrics import f1_score # F値
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# load the dataset
dataset_org = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Convert to the matrix
dataset = dataset_org.to_numpy()
np.set_printoptions(suppress=True, precision=2)

# Create the new matrix without test-data
X = dataset[:,0:12]

# standardize X
sc = StandardScaler()
X_std = sc.fit_transform(X)

# Create the test-data
y = dataset[:,12]

# Split the dataset into training and testing set(sklearn)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.5)


# Create linearSVC object
lin_clf = svm.LinearSVC(tol=1e-5, max_iter=10000)

# Train the model using the training set
lin_clf.fit(X_train, y_train)

# Make predictions using the testing set
predicted = lin_clf.predict(X_test)


# 正解率(accuracy), Accuracy = (TP + TN) / (TP + TN + FP + FN)
print("正解率 = " + str(accuracy_score(y_test, predicted)))

# 適合率(precision), Precision = TP / (TP + FP)
print("適合率 = " + str(precision_score(y_test, predicted)))

# 検出率(recall), Recall = TP / (TP + FN)
print("検出率 = " + str(recall_score(y_test, predicted)))

# F値, F1 = 2 * (precision * recall) / (precision + recall)
print("F値 = " + str(f1_score(y_test, predicted)))

