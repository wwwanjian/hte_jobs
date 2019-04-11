from sklearn.decomposition import PCA
from .classification import get_X_Y_data, get_random_filename
import pandas as pd
import os
from settings.default import MEDIA_DIR


def pca_feature(params):
    clf = PCA(n_components=2)
    clf.set_params(**params)
    return clf


def pca_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    clf.fit(X)
    X_after_pca = clf.transform(X)
    result = {}
    result['components'] = [list(component) for component in clf.components_]
    result['explained_variance_ratio'] = list(clf.explained_variance_ratio_)
    result['singular_values'] = list(clf.singular_values_)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file = pd.DataFrame(X_after_pca)
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result
