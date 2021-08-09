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
import time
import threading

# load the dataset
natural_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

def conv_matrix(natural_data):
    dataset = natural_data.to_numpy()
    np.set_printoptions(suppress=True, precision=2)
    return dataset

def create_features_matrix(dataset):
    X = dataset[:, 0:12]
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    return X_std

def create_test_matrix(dataset):
    y = dataset[:, 12]
    return y

"""
thread1 = threading.Thread(target=create_features_matrix(conv_matrix(natural_data)))
thread2 = threading.Thread(target=create_test_matrix(conv_matrix(natural_data)))
thread1.start()
thread2.start()
thread1.join()
thread2.join()
"""

# Split the dataset into training and testing set(sklearn)
X_train, X_test, y_train, y_test = train_test_split(create_features_matrix(conv_matrix(natural_data)), create_test_matrix(conv_matrix(natural_data)), test_size = 0.5)


def ml(X_train, y_train, X_test):
    lin_clf = svm.LinearSVC(tol=1e-5, max_iter=10000)
    lin_clf.fit(X_train, y_train)
    predicted = lin_clf.predict(X_test)
    return predicted

predicted = ml(X_train, y_train,X_test)


def index(y_test, predicted):
    print("正解率 = " + str(accuracy_score(y_test, predicted)))
    print("適合率 = " + str(precision_score(y_test, predicted)))
    print("検出率 = " + str(recall_score(y_test, predicted)))
    print("F値 = " + str(f1_score(y_test, predicted)))


index(y_test, predicted)




