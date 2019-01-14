import time

import sklearn.datasets
import sklearn.ensemble

X, y = sklearn.datasets.make_classification(600000, 66)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)

print('Starting to fit the RF.')
start_time = time.time()
rf.fit(X, y)
end_time = time.time()
print(end_time - start_time)