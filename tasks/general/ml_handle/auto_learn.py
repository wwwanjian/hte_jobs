from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import make_scorer
from .classification import get_X_Y_data, split_dataset, get_random_filename
from sklearn import metrics
import pandas as pd
import os
from settings.default import MEDIA_DIR


def auto_classfication(params):
    clf = AutoSklearnClassifier()
    clf.set_params(**params)
    return clf


def auto_regession(params):
    clf = AutoSklearnRegressor()
    clf.set_params(**params)
    return clf


def auto_class_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    #socer = make_scorer('f1', metrics.f1_score(y_test, y_prediciton))
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    #result['f1'] = socer
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


def auto_reg_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    y_prediciton = clf.predict(x_test)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


if __name__ == "__main__":
    params = {'time_left_for_this_task': 60, 'per_run_time_limit': 6,
              'include_estimators': ['svm']}
    clf = auto_classfication(params)
    print(clf)
