from sklearn.base import BaseEstimator
from sklearn import ensemble
import sklearn
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.model = ensemble.RandomForestRegressor(max_depth=40, n_estimators=100, random_state=57, n_jobs=-1)

    def fit(self, X, Y):
        self.model.fit(np.log(X), np.log(Y))

    def predict(self, X):
        res = self.model.predict(np.log(X))
        return np.exp(res)

