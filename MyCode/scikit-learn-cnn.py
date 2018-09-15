import numpy as np
import pandas as pd
from sklearn import datasets

# Get IRIS dataset.
scikit_iris = datasets.load_iris()

# print(scikit_iris)

# Translate it to pandas' DataFrame format.
pd_iris = pd.DataFrame(
    data = np.c_[scikit_iris['data'], scikit_iris['target']],
    columns = np.append(scikit_iris.feature_names, ['y'])
    )

# print(pd_iris.head(30))

# Use all the data to train the model.
x = pd_iris[scikit_iris.feature_names]
y = pd_iris['y']

from sklearn.cross_validation import train_test_split
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.5, random_state = 0)

# (1) Select the model.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

# (2) Train the model.
knn.fit(x, y)

# (3) Predict with new data.
# res = knn.predict([[4, 3, 5, 3]])
# print(res)
y_predict_on_train = knn.predict(x_train)
y_predict_on_test = knn.predict(x_test)

print('Prediction rate on train data is: {}'.format(metrics.accuracy_score(y_train, y_predict_on_train)))
print('Prediction rate on test data is: {}'.format(metrics.accuracy_score(y_test, y_predict_on_test)))

