import os
import string
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

problem_title = "In-Vino-Veritas"

Predictions = rw.prediction_types.make_regression(label_names="prix_m")
workflow = rw.workflows.Regressor()

class MAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MAPE", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mape = (np.abs(y_true - y_pred) / y_true).mean()
        return mape


score_types = [
    MAPE(name="MAPE"),
]

def get_train_data(path="."):
    train_data = pd.read_csv("data/train.csv", sep=";", encoding='utf-8')
    train_data = train_data[train_data['millesime'].notna()].reset_index()
    train_data = train_data[train_data['prix_m'].notna()].reset_index()

    Y = train_data[['prix_m']].copy()

    X = train_data.drop(columns=[ 'prix', 'prix_min', 'prix_max', 'prix_m'])
    print("finish get_train_data")
    return X,Y

def get_test_data(path="."):
    test_data = pd.read_csv("data/test.csv", sep=";", encoding='utf-8')
    test_data = test_data[test_data['millesime'].notna()].reset_index()
    test_data = test_data[test_data['prix_m'].notna()].reset_index()

    Y = test_data[['prix_m']].copy()

    X = test_data.drop(columns=[ 'prix', 'prix_min', 'prix_max', 'prix_m'])
    print("finish get_test_data")
    return X,Y

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
