from sklearn.base import BaseEstimator
from sklearn import ensemble
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce

class Regressor(BaseEstimator):
    def __init__(self):
        print("init Regressor")
        self.ohe_cols = ['classement', 'region', 'commentaire', 'type', 'consommation', 'elevage']
        self.tgt_cols = ['vin', 'producteur', 'appellation']
        ohe_pipeline = Pipeline([
        ("one-hot-encoder", OneHotEncoder(handle_unknown='ignore'))
        ])

        tgt_pipeline = Pipeline([
            ("target-encoder", ce.TargetEncoder())
        ])

        preprocessing = ColumnTransformer([
            ("ohe_preproc", ohe_pipeline, self.ohe_cols),
            ("tgt_preproc", tgt_pipeline, self.tgt_cols)
        ])

        self.model = Pipeline([
            ("Preprocessing", preprocessing),
            ("regressor", LinearRegression())
        ])

    def fit(self, X, Y):
        print("fit Regressor")
        X.drop(columns=['guide',  'garde', 'cuvee',
                        'prod_id',
                        'adresse', 'CP', 'commune', 'pays',
                        'INSEE_COM', 'INSEE_DEP', 'INSEE_REG'], inplace=True)
        self.model.fit(X, Y)
        

    def predict(self, X):
        X.drop(columns=['guide',  'garde', 'cuvee',
                        'prod_id',
                        'adresse', 'CP', 'commune', 'pays',
                        'INSEE_COM', 'INSEE_DEP', 'INSEE_REG'], inplace=True)
        return self.model.predict(X)

