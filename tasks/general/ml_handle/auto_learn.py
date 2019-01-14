from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import make_scorer
from .classification import get_X_Y_data, split_dataset
from sklearn import metrics


def auto_classfication(params):
    clf = AutoSklearnClassifier()
    clf.set_params(**params)
    return clf


def auto_regession(params):
    clf = AutoSklearnRegressor()
    clf.set_params(**params)
    return clf


def auto_class_result(label,features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    socer = make_scorer('f1', metrics.f1_score(y_test, y_prediciton))
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    return result


def auto_reg_result(label,features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    return result


if __name__ == "__main__":
    params = {'time_left_for_this_task': 60, 'per_run_time_limit': 6,
              'include_estimators': ['svm']}
    clf = auto_classfication(params)
    print(clf)

