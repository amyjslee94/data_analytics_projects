import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew

class SkewBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_threshold = 0.5):
        self.skewness_threshold = skewness_threshold
        self.imputer_ = {}
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            skewness = skew(X[col].dropna())
            strategy = "median" if abs(skewness)>self.skewness_threshold else "mean"
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(X[[col]])
            self.imputer_[col] = imputer
        return self
    
    def transform(self, X):
        X_transformed = pd.DataFrame(X)
        for col, imputer in self.imputer_.items():
            X_transformed[[col]] = imputer.transform(X_transformed[[col]])
        return X_transformed

if __name__=="__main__":
    df = pd.read_csv('./data/train.csv')
    testing = df[['Age','Survived']]
    # testing = testing.fillna(100)
    print(SkewBasedImputer().fit_transform(X=testing[['Age']]))

    # from sklearn.base import BaseEstimator, TransformerMixin
    # from sklearn.impute import SimpleImputer
    # from sklearn.pipeline import Pipeline
    # from scipy.stats import skew
    # import pandas as pd
    # import numpy as np

    # class SkewnessBasedImputer():
    #     def __init__(self, skew_threshold=0.5):  # Adjust the skew threshold as needed
    #         self.skew_threshold = skew_threshold
    #         self.imputers_ = {}

    #     def fit(self, X, y=None):
    #         self.imputers_ = {}
    #         for col in X.columns:
    #             skewness = skew(X[col].dropna())
    #             if abs(skewness) > self.skew_threshold:
    #                 # Use median for skewed data
    #                 strategy = 'median'
    #             else:
    #                 # Use mean for approximately normal data
    #                 strategy = 'mean'
                
    #             imputer = SimpleImputer(strategy=strategy)
    #             imputer.fit(X[[col]])
    #             self.imputers_[col] = imputer
    #         return self

    #     def transform(self, X):
    #         X_transformed = X.copy()
    #         for col, imputer in self.imputers_.items():
    #             X_transformed[[col]] = imputer.transform(X[[col]])
    #         return X_transformed

    # # Example Dataset
    # data = pd.DataFrame({
    #     'feature1': [1, 2, np.nan, 4, 5],  # Low skew
    #     'feature2': [1, 2, 1000, np.nan, 5]  # High skew
    # })

    # # Create pipeline with the custom imputer
    # pipeline = Pipeline([
    #     ('skew_imputer', SkewnessBasedImputer(skew_threshold=1))
    # ])

    # # Fit and transform
    # transformed_data = pipeline.fit_transform(X=data[['feature2']], y=data[['feature1']])
    # print(pd.DataFrame(transformed_data, columns=data.columns))