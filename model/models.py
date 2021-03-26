from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import numpy as np


def random_forest():
    # Number of trees in random forest
    n_estimators = np.arange(200, 2000, 200)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    rf_random_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}

    rf_cls = RandomForestClassifier()

    return rf_cls, rf_random_grid


def svc():
    # Regularization parameter
    c_value = [0.5, 0.75, 1.0]
    # Specifies the kernel type to be used in the algorithm.
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    # Degree of the polynomial kernel function (‘poly’)
    degree = [2, 3]
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    gamma = ['scale', 'auto']

    svc_random_grid = {'C': c_value,
                       'kernel': kernel,
                       'degree': degree,
                       'gamma': gamma
                       }

    svc_cls = SVC()

    return svc_cls, svc_random_grid
