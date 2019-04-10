from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from werkzeug.utils import secure_filename


def get_clf():
    clf = svm.SVC()
    return clf


def get_data(path):
    df = pd.read_excel(path)
    Y = df['Y']
    print(df.columns)
    X = df[['X1', 'X2', 'X3', 'X4']]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=618)
    return x_train, x_test, y_train, y_test


def get_result(path):
    x_train, x_test, y_train, y_test = get_data(path)
    clf = get_clf()
    clf.fit(x_train, y_train)
    y_prediciton = clf.predict(x_test)
    result = {}
    result['acc'] = metrics.accuracy_score(y_test, y_prediciton)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    result_file = pd.concat([x_test, y_test,
                             pd.DataFrame(y_prediciton, columns=['Y_pred'])], axis=1)
    filename = secure_filename('result_file.xlsx')
    result_file.to_excel(filename)
    print(result)
    return result


if __name__ == '__main__':
    path = 'test.xlsx'
    result = get_result(path)
