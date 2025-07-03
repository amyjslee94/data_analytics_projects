import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew

class SkewBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_threshold=0.5):
        self.skewness_threshold = skewness_threshold
        self.imputer_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        X = pd.DataFrame(X)  # Ensure X is a DataFrame
        for col in X.columns:
            skewness = skew(X[col].dropna())
            strategy = "median" if abs(skewness) > self.skewness_threshold else "mean"
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(X[[col]])
            self.imputer_[col] = imputer
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(X, columns=self.feature_names_in_)
        for col, imputer in self.imputer_.items():
            X_transformed[[col]] = imputer.transform(X_transformed[[col]])
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            raise AttributeError("Feature names are not available. Fit the transformer first.")
        return self.feature_names_in_



if __name__=="__main__":
    df = pd.read_csv('./data/train.csv')
    testing = df[['Age','Survived']]
    print(SkewBasedImputer().fit_transform(X=testing[['Age']]))
