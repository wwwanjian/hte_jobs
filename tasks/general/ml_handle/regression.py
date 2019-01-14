from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import pandas as pd
from .classification import split_dataset, get_X_Y_data


def linear_regression(params):
    clf = LinearRegression()
    clf.set_params(**params)
    return clf


def linear_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    result['score'] = clf.score(x_test, y_test)
    result['coef'] = list(clf.coef_)
    result['intercept'] = clf.intercept_
    return result


def ridg_regression(params):
    clf = Ridge()
    clf.set_params(**params)
    return clf


def ridge_result(label,features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    result['score'] = clf.score(x_test, y_test)
    result['coef'] = list(clf.coef_)
    result['intercept'] = clf.intercept_
    return result


def bayes_regression(params):
    clf = BayesianRidge()
    clf.set_params(**params)
    return clf


def bayes_result(label, features,clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    result['score'] = clf.score(x_test, y_test)
    result['coef'] = list(clf.coef_)
    result['intercept'] = clf.intercept_
    return result


if __name__ == "__main__":
    df = pd.read_excel('linear.xlsx')
    clf = linear_regression({})
    X, Y = get_X_Y_data(df, ['Y'])
    x_train, x_test, y_train, y_test = split_dataset(X, Y, 0.3)
    clf.fit(x_train, y_train)
    result = {}
    result['score'] = clf.score(x_test, y_test)
    result['coef'] = clf.coef_
    result['intercept'] = clf.intercept_
    imgs = []
    imgs.append(get_heatmap_path(X))
    imgs.append(get_feature_imp_path(X, Y))
    print(result)
    print(imgs)

