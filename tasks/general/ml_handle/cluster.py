from sklearn.cluster import KMeans
from .classification import get_X_Y_data


def kmeans_cluster(params):
    clf = KMeans(n_clusters=3)
    clf.set_params(**params)
    return clf


def kmeans_result(label, features, clf, dataset):
    X, Y = get_X_Y_data(dataset, label, features)
    clf.fit(X)
    result = {}
    result['centers'] = [list(center) for center in clf.cluster_centers_]
    imgs = None
    return result, imgs


if __name__ == "__main__":
    pass

