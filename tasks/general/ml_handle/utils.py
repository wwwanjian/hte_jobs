from .classification import svm_classifier, svm_result, gpc_classifier, \
    gpc_result, mlp_classifier, mlp_result
from .regression import linear_regression, linear_result, ridg_regression, \
    ridge_result, bayes_regression, bayes_result
import pandas as pd
from .cluster import kmeans_cluster, kmeans_result
from .features import pca_feature, pca_result
from .auto_learn import auto_classfication, auto_regession, auto_class_result, auto_reg_result

SUPPORT_FILE_TYPE = ['xlsx', 'csv', 'xls', 'txt']
Y_NAMES = ['Y', 'y', '标签', 'label']
ALG_SELECTION = {
    'SVM': svm_classifier,
    'GPC': gpc_classifier,
    'MLPC': mlp_classifier,
    'Linear': linear_regression,
    'Ridge': ridg_regression,
    'Bayes': bayes_regression,
    'KMeans': kmeans_cluster,
    'PCA': pca_feature,
    'AUTO_CLASS': auto_classfication,
    'AUTO_REG': auto_regession,
}
RESULT_SELECTION = {
    'SVM': svm_result,
    'GPC': gpc_result,
    'MLPC': mlp_result,
    'Linear': linear_result,
    'Ridge': ridge_result,
    'Bayes': bayes_result,
    'KMeans': kmeans_result,
    'PCA': pca_result,
    'AUTO_CLASS': auto_class_result,
    'AUTO_REG': auto_reg_result,
}


def check_file_type(file_path: str):
    file_type = file_path.split('.')[-1]
    if file_type in SUPPORT_FILE_TYPE:
        return file_type
    else:
        return False


def read_dataset(file_path: str) -> pd.DataFrame:
    file_type = check_file_type(file_path)
    if not file_type:
        return 'error'
    elif file_type in ['xlsx', 'xls']:
        dataset = pd.read_excel(file_path)
    elif file_type in ['csv', 'txt']:
        dataset = pd.read_csv(file_path)
    return dataset


def get_x_y_data(dataset: pd.DataFrame):
    names = dataset.columns
    y_name = [i for i in names if i in Y_NAMES][0]
    x_name = names.drop([y_name])
    y_data = dataset[x_name]
    x_data = dataset[y_name]
    return y_data, x_data


def get_train_model(alg, params):
    if alg in ALG_SELECTION.keys():
        clf = ALG_SELECTION[alg](params)
    else:
        return 'error'
    return clf


def get_prediction_model(alg):
    clf = 'load model'
    return clf


def get_result_and_imgs(alg, clf, label, features, file_path):
    dataset = read_dataset(file_path)
    result = RESULT_SELECTION[alg](label,features, clf, dataset)
    return result

