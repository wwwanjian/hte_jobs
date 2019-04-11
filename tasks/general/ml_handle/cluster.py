from sklearn.cluster import KMeans
from .classification import get_X_Y_data,get_random_filename
from settings.default import MEDIA_DIR
import pandas as pd
import os


def kmeans_cluster(params):
    clf = KMeans(n_clusters=3)
    clf.set_params(**params)
    return clf


def kmeans_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    clf.fit(X)
    result = {}
    result['centers'] = [list(center) for center in clf.cluster_centers_]
    y_pre = clf.predict(X)
    result_file = pd.concat([X, pd.DataFrame(y_pre, columns=['Y_pred'])], axis=1)
    filename = get_random_filename(20)
    result['filename'] = filename
    result_file.to_excel(os.path.join(MEDIA_DIR, 'result', result['filename']), index=False)
    return result


if __name__ == "__main__":
    pass

