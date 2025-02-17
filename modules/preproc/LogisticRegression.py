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
        self.feature_names_in_ = None
    
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
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        y = pd.Series(y)
        X_t = pd.DataFrame(X).copy()
        
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

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            raise AttributeError("Feature names are not available. Fit the transformer first.")
        return self.feature_names_in_

class InteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, interaction_pairs):
        self.interaction_pairs = interaction_pairs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        X_t=X.copy()
        for pairs in self.interaction_pairs:
            group1 = [col for col in X.columns if pairs[0] in col]
            group2 = [col for col in X.columns if pairs[1] in col]
            if len(group1)==0 or len(group2)==0:
                continue
            for x1 in group1:
                for x2 in group2:
                    X_t[f'{x1.split("__")[-1]}_{x2.split("__")[-1]}'] = X[x1] * X[x2]
        return X_t
    
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from AutoImputer import SkewBasedImputer

    df = pd.read_csv('./Titanic/data/train.csv')
    def get_family_flg(X):
        def family_onboard(X):
            if X['SibSp'] == 0 and X['Parch'] == 0:
                return 'no_fam'
            elif X['SibSp'] > 0 and X['Parch'] == 0:
                return 'SibSp'
            elif X['SibSp'] == 0 and X['Parch'] > 0:
                return 'Parch'
            elif X['SibSp'] > 0 and X['Parch'] > 0:
                return 'big_fam'
            else:
                return 'NA'
        X = X.copy()
        X['transformed'] = X.apply(lambda x: family_onboard(x), axis=1)
        X['transformed'] = pd.Categorical(X['transformed'],
                                        categories=sorted(list(set(X['transformed']))),
                                        ordered=True)
        return X[['transformed']]

    df['family_flg'] = get_family_flg(df[['SibSp','Parch']])
    categorical_columns = ['Pclass','family_flg']
    continuous_columns=['Age','Fare']
    interaction_terms = [
        # ('Age','cabin_flg'),
        ('Age','family_flg'),
        # ('Age','Pclass'),
        # ('cabin_flg','family_flg'),
        # ('cabin_flg','Fare'),
        # ('cabin_flg','Pclass_1'),
        # ('cabin_flg','Sex'),
        # ('Embarked','family_flg_no_fam'),
        # ('Embarked','Fare'),
        # ('Embarked','Pclass_3'),
        # ('Embarked','ticket_share_flg'),
        ('family_flg','Pclass')
        # ('family_flg','ticket_share_flg'),
        # ('Pclass','Sex'),
        # ('Pclass','ticket_share_flg')
    ]

    logit_continuous_pipeline = Pipeline([
        ('logodds', LogOddsTransformer(max_iter=7)),
        ('scaler', StandardScaler().set_output(transform="pandas")),
        ('impute', SkewBasedImputer())
    ])

    logit_categorical_pipeline = Pipeline([
        ('one-hot encoding', OneHotEncoder(sparse_output=False))
    ])
    logit_preprocessor = ColumnTransformer(
        transformers=[
            ('cont_pipeline', logit_continuous_pipeline, continuous_columns),
            ('cat_pipeline', logit_categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False
    )
    # logit_preprocessor.get_feature_names_out()
    logit_preprocessor.set_output(transform='pandas')

    logit_full_pipeline = Pipeline([
        ('preprocessor', logit_preprocessor),
        ('interaction', InteractionTransformer(interaction_terms)),
        ('model', LogisticRegression(penalty='l1', solver='liblinear'))
    ])
    
    df = df.loc[~df['Embarked'].isna()]
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


    # # logit_preprocessor.set_output(transform="pandas")
    # test = logit_full_pipeline.fit_transform(X,y)
    # # logit_preprocessor
    # # X_transformed = logit_preprocessor.fit_transform(X, y)
    # test = logit_full_pipeline.fit_transform(X, y)
    # print(test)
    # # logit_preprocessor.fit(X, y)
    # # print(logit_preprocessor.named_transformers_['cont_pipeline'].named_steps['scaler'].get_feature_names_out())
    # # print(logit_preprocessor.named_transformers_['cont_pipeline'].named_steps['impute'].get_feature_names_out())
    # # print(logit_preprocessor.named_transformers_['cat_pipeline'].named_steps['one-hot encoding'].get_feature_names_out(categorical_columns))
    
    # # logit_preprocessor.set_output(transform='pandas')
    
    # print(test)
    







    # #     # Get continuous feature names
    # # cont_features = continuous_columns  # 'Age', 'Fare'

    # # # Get categorical feature names after one-hot encoding
    # # cat_features = logit_preprocessor.named_transformers_['cat_pipeline'].named_steps['one-hot encoding'].get_feature_names_out(categorical_columns)

    # # # Combine feature names from continuous and categorical transformations
    # # initial_feature_names = list(cont_features) + list(cat_features)

    # # # Add interaction terms to the feature names
    # # interaction_features = []
    # # for pairs in interaction_terms:
    # #     group1 = [col for col in initial_feature_names if pairs[0] in col]
    # #     group2 = [col for col in initial_feature_names if pairs[1] in col]
    # #     for x1 in group1:
    # #         for x2 in group2:
    # #             interaction_features.append(f'{x1}_{x2}')

    # # # Combine all feature names
    # # all_feature_names = initial_feature_names + interaction_features

    # # # Inspect the transformed data
    # # X_transformed = logit_full_pipeline.fit_transform(X, y)
    # # print(X_transformed)
    # # X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)

    # # # Display the first few rows of the transformed DataFrame
    # # print(X_transformed_df.head())






    # # # Fit the preprocessing pipeline and transform the data
    # # X_transformed = logit_preprocessor.fit_transform(X, y)

    # # # Convert the transformed data into a DataFrame for inspection
    # # # Extract feature names from the preprocessing pipeline
    # # 

    # # # Get transformed feature names
    # # cont_features = continuous_columns
    # # cat_features = logit_preprocessor.named_transformers_['cat_pipeline'].named_steps['one-hot encoding'].get_feature_names_out(categorical_columns)
    

    # # # Combine all feature names
    # # feature_names = list(cont_features) + list(cat_features)

    # # # Convert to DataFrame for easier inspection
    # # X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # # # Display the first few rows of the transformed DataFrame
    # # print(X_transformed_df.head())















    
    

    
        # print(y)