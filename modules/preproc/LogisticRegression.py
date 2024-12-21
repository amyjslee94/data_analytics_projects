import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from scipy.stats import skew

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class LogOddsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=5, epsilon=1):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.transformations_ = {}
    
    def _pred_second_derivative(self, model, X, col):
        predictions = model.predict(X)
        log_odds = np.log((predictions / (1 - predictions)))
        sd = np.gradient(np.gradient(log_odds, X[col]), X[col])
        sd_val = np.nanmean(sd)
        return sd_val

    def _transform_column(self, x, transformation_method):
        if transformation_method == "sqrt":
            return x ** (1/2)
        elif transformation_method == "log":
            return np.log(x + self.epsilon)
        elif transformation_method == "neg_inv_sqrt":
            return -((x+self.epsilon) ** -(1/2)) + 1
        elif transformation_method == "neg_inv":
            return -((x+self.epsilon) ** -1) + 1
        elif transformation_method == "pwr2":
            return x** 2
        elif transformation_method == "pwr3":
            return x** 3
        elif transformation_method == "pwr4":
            return x** 4

        else:
            return x

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        X_t = X.copy()
        
        transformation_order = ["sqrt","log","pwr2", "neg_inv_sqrt","pwr3","neg_inv", "pwr4"]
        for col in X_t.columns:
            for iter in range(self.max_iter):
                feature = X_t[[col]].dropna()
                feature[f'{col}_log'] = feature[col] * np.log(feature[col] + 1)
                feature = sm.add_constant(feature, prepend = False)
                

                target = y[feature.index]
                logit = sm.Logit(target, feature).fit(disp=False)
                logodds_pval = logit.pvalues.get(f'{col}_log', None)

                if logodds_pval is None or logodds_pval > 0.05:
                    break
                
                transformation_method = transformation_order[iter]
                self.transformations_[col] = transformation_method
                X_t[col] = self._transform_column(X[col], transformation_method)

        return self
                
    def transform(self, X):
        X_transformed = pd.DataFrame(X)
        for col, transformation in self.transformations_.items():
            if transformation is not None:
                X_transformed[[col]] = self._transform_column(X_transformed[[col]], transformation)
        return X_transformed

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LogisticRegression
    from AutoImputer import SkewBasedImputer

    df = pd.read_csv('./Titanic/data/train.csv')
    categorical_columns = ['Pclass']
    continuous_columns=['Age','Fare']

    logit_continuous_transformer_pipeline = Pipeline([
        ('logodds', LogOddsTransformer(max_iter=7)),
        ('scaler', StandardScaler()),
        ('impute', SkewBasedImputer())
    ])

    logit_preprocessor = ColumnTransformer(
        transformers=[
            ('cont_pipeline', logit_continuous_transformer_pipeline, continuous_columns)
        ]
    )
    logit_full_pipeline = Pipeline([
        ('preprocessor', logit_preprocessor),
        ('model', LogisticRegression(penalty='l2', solver='liblinear'))
    ])

    X = df[categorical_columns + continuous_columns]
    y = df['Survived']

    cv = 7
    scoring = {'acc': 'accuracy',
               'precision': 'precision',
               'recall': 'recall',
               'f1': 'f1',
               'roc_auc': 'roc_auc'}
    scores = cross_validate(logit_full_pipeline, X, y, scoring=scoring, cv=cv)
    for x,y in scores.items():
        print(x)
        print(np.mean(y))
        # print(y)