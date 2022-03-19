import os
import string
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

problem_title = "Isotopic inventory of a nuclear reactor core in operation"

_target_names = [
    j + str(i + 1) for j in list(string.ascii_uppercase) for i in range(80)
]

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
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


def get_file_list_from_dir(*, path, datadir):
    data_files = sorted(glob(os.path.join(path, "data", datadir, "*.csv.gz")))
    return data_files


def _get_data(path=".", split="train"):
    # load and concatenate data in one dataset
    # ( train data are composed of 690 different
    # simulations of an operating reactor
    # and test data of 230 simulations)
    # returns X (input) and Y (output) arrays
    data_files = get_file_list_from_dir(path=path, datadir=split)
    dataset = pd.concat([pd.read_csv(f) for f in data_files])

    # Isotopes are named from A to Z
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0.
    # Those are the input composition
    # The input parameter space is composed of those initial
    # compositions + operating parameters p1 to p5
    input_params = alphabet[:8] + ["p1", "p2", "p3", "p4", "p5"]

    data = dataset[alphabet].add_prefix("Y_")
    data["times"] = dataset["times"]
    data = data[data["times"] > 0.0]

    temp = pd.DataFrame(
        np.repeat(dataset.loc[0][input_params].values, 80, axis=0),
        columns=input_params
    ).reset_index(drop=True)
    data = pd.concat([temp, data.reset_index(drop=True)], axis=1)

    # data = shuffle(data, random_state=57)

    X_df = (
        data.groupby(input_params)["A"]
        .apply(list)
        .apply(pd.Series)
        .rename(columns=lambda x: "A" + str(x + 1))
        .reset_index()[input_params]
    )
    Y_df = []
    for i in alphabet:
        Y_df.append(
            data.groupby(input_params)["Y_" + i]
            .apply(list)
            .apply(pd.Series)
            .rename(columns=lambda x: i + str(x + 1))
            .reset_index()
            .iloc[:, len(input_params):]
        )
    Y_df = pd.concat(Y_df, axis=1)

    X = X_df.to_numpy()
    Y = Y_df.to_numpy()
    return X, Y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
