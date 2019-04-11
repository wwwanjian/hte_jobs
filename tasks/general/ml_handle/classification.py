from sklearn import svm
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from settings.default import MEDIA_DIR
import random
import os


def get_random_filename(n=20):
    charter = '01233456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
    filename = ''
    for i in range(n):
        filename += charter[random.randint(0, len(charter) - 1)]
    filename += '.xlsx'
    return filename


def split_dataset(X, Y, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=618)
    return x_train, x_test, y_train, y_test


def get_X_Y_data(dataset, label, features):
    X = dataset[features]
    Y = None
    if label:
        Y = dataset[label]
    return X, Y


def svm_classifier(params):
    clf = svm.SVC()
    clf.set_params(**params)
    return clf


def svm_result(label, features, clf, dataset):
    print(1)
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


def gpc_classifier(params):
    clf = GaussianProcessClassifier()
    clf.set_params(**params)
    return clf


def gpc_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


def mlp_classifier(params):
    clf = MLPClassifier()
    clf.set_params(**params)
    return clf


def mlp_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


if __name__ == '__main__':
    params = {'gamma': 'scale', 'decision_function_shape': 'ovr', 'degree': 4}
    svm_classifier(params)
    filename = get_random_filename(20)
