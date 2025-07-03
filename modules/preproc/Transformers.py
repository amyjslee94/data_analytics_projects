import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
        
class LogOddsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to adjust features while satisfying logistic regression assumption
    """
    def __init__(self, max_iter=9, epsilon=1, pval=0.05):
        if max_iter > 9:
            raise ValueError("max_iter cannot be greater than 7")
        self.max_iter = max_iter
        self.epsilon = epsilon # avoid log(0) case
        self.transformations_pvals_ = {}
        self.transformations_ = {}
        self.feature_names_in_ = None
        self.pval = pval
        
    def __repr__(self):
        return f"LogOddsTransformer(max_iter={self.max_iter}, epsilon={self.epsilon})"

    def _transform_column(self, val, transformation_method):
        val = val.astype(float)
        transformations = {
            "sqrt": lambda x: np.sqrt(x),
            "log": lambda x: np.log(x + self.epsilon),
            "neg_inv_sqrt": lambda x: -((x + self.epsilon) ** -0.5) + 1,
            "neg_inv": lambda x: -((x + self.epsilon) ** -1) + 1,
            "pwr2": lambda x: x**2,
            "pwr3": lambda x: x**3,
            "pwr4": lambda x: x**4,
            "pwr5": lambda x: x**5,
        }
        return transformations.get(transformation_method, lambda x: x)(val)

    def _validate_input(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if y is not None:
            if not isinstance(y, (np.ndarray, pd.Series)):
                raise ValueError("y must be a numpy array or pandas DataFrame")

    def fit(self, X, y):
        """
        Use Box-Tidwell test to determine the best transformation method
        Box-Tidwell : logit(p) = b0 + b1X + b2(X*log(X))
        If b2 pval is significant, the relationship is NOT linear
        Transformation is repeated until pval becomes insignificant
        If every transformations' pval is significant, transformation method with the highest pval is chosen
        """
        self._validate_input(X, y)
        X = pd.DataFrame(X.copy())
        y = pd.Series(y)
        self.feature_names_in_ = X.columns.to_list()
        
        transformation_order = ["base","sqrt","pwr2","log","pwr3", "neg_inv_sqrt","pwr4","neg_inv", "pwr5"]

        for col in X.columns:
            logodds_pvals = {}
            found = False

            for i, transformation_method in enumerate(transformation_order[:self.max_iter]):
                feature = self._transform_column(X[[col]].dropna(), transformation_method)
                target = y[feature.index]
                
                # box-tidwell computation
                feature['log_val'] = feature[col].apply(lambda x: x * np.log(x+self.epsilon)) 
                feature = sm.add_constant(feature, prepend = False)
                logit = sm.Logit(target, feature).fit(disp=False)
                
                logodds_pval = logit.pvalues.get('log_val', None)
                logodds_pvals[transformation_method] = logodds_pval
                self.transformations_pvals_[col] = logodds_pvals

                if logodds_pval > self.pval: 
                    self.transformations_[col] = transformation_method
                    found = True
                    break

            if logodds_pvals and not found:
                best_method = max(logodds_pvals, key=logodds_pvals.get)
                self.transformations_[col] = best_method

            
        return self
                
    def transform(self, X):
        self._validate_input(X)
        X_transformed = X.copy() 
        for col, transformation in self.transformations_.items():
                X_transformed[[col]] = self._transform_column(X_transformed[[col]], transformation)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            raise AttributeError("Feature names not available.")
        return np.array(self.feature_names_in_)



class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create interaction terms defined in interaction_pairs
    """
    def __init__(self, interaction_pairs):
        if not isinstance(interaction_pairs,list):
            raise ValueError("interaction_paris must be a list of tuples")

        if not all(isinstance(e,tuple) and len(e)==2 for e in interaction_pairs):
            raise ValueError("Each pair in interaction_pairs must be a tuple of two elements exactly")

        self.interaction_pairs = interaction_pairs
        self.feature_names_out_ = None
    
    def __repr__(self):
        return f"InteractionTransformer(interaction_pairs={self.interaction_pairs})"

    def _validate_input(self, X, interaction_pairs):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        for (pair1, pair2) in interaction_pairs:
            if not any(pair1 in col for col in X.columns):
                raise ValueError(f"No column in the DataFrame contains '{pair1}' as a substring")
            if not any(pair2 in col for col in X.columns):
                raise ValueError(f"No column in the DataFrame contains '{pair2}' as a substring")

    def fit(self, X, y=None):
        self._validate_input(X, self.interaction_pairs)
        self.feature_names_in_ = X.columns.to_list()

        return self
    
    def transform(self, X):
        self._validate_input(X, self.interaction_pairs)
        X = X.copy()
        interaction_vals = []
        interaction_colnames = []
        for p1, p2 in self.interaction_pairs:
            p1_cols = [col for col in X.columns if col.startswith(p1)]
            p2_cols = [col for col in X.columns if col.startswith(p2)]
            if len(p1_cols)==0 or len(p2_cols)==0:
                print(f"{p1} or {p2} not available in DataFrame: {X.columns}\nskipping this interaction")
                break
            combinations = list(product(p1_cols, p2_cols))
            for c1, c2 in combinations:
                interaction_colnames.append(f'{c1} * {c2}')
                interaction_vals.append(X[c1] * X[c2])
        
        interaction_df = pd.concat(interaction_vals, axis=1)
        interaction_df.columns = interaction_colnames
        X = pd.concat([X, interaction_df], axis=1)
        self.feature_names_out_ = X.columns.to_list()

        return X
    
    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            raise AttributeError("Feature names are not available. Fit the transformer first.")
        return np.array(self.feature_names_out_)



class NamedFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around FunctionTransformer to provide get_feature_names_out for GridSearchCV
    """
    def __init__(self, func, feature_names_out_=None, include_original=True):
        """
        Args:
            func: function to transform the input features
            feature_name_out: transformed feature names
            include_original: returns both original and transformed features if True
        """
        self.func = func  
        self.feature_names_out_ = feature_names_out_
        self.include_original = include_original
        self.transformer = FunctionTransformer(func)

    def __repr__(self):
        return f"NamedFunctionTransformer({self.func.__name__})"
    
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        self.feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        if not isinstance(X_transformed, pd.DataFrame):
            if self.feature_names_out_ is not None:
                columns = self.feature_names_out_
            else:
                columns = [f"{self.func.__name__}_{i}" for i in range(X_transformed.shape[1])]
            X_transformed = pd.DataFrame(X_transformed, index=X.index, columns=columns)

        if self.include_original:

            return pd.concat([X, X_transformed], axis=1)
        else:
            return X_transformed

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            raise ValueError("Feature names are not specified for this transformer.")
        
        if self.include_original:
            if input_features is None:
                if self.feature_names_out_ is None:
                    raise ValueError("Original feature names are missing. Fit the transformer first.")
            return np.array(self.feature_names_in_ + self.feature_names_out_)
        
        return np.array(self.feature_names_out_)



class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='iso', contamination=0.05, n_neighbors=20, novelty=True):
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.novelty = novelty
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()

        if isinstance(X, pd.DataFrame):
            X = X.values  # Force numpy array to avoid future mismatch
        
        if self.method == 'iso':
            self.detector_ = IsolationForest(contamination=self.contamination, random_state=1)
        elif self.method == 'lof':
            self.detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors, 
                contamination=self.contamination,
                novelty=self.novelty
            )
        else:
            raise ValueError("method must be one of the following: ['iso', 'lof']")
        self.detector_.fit(X)
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X_np = X.values
        pred = self.detector_.predict(X_np)
        X[f'outlier_{self.method}'] = (pred == -1).astype(int)
        self.feature_names_out_ = X.columns.to_list()

        return X
    
    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            raise AttributeError("Feature names are not available. Fit the transformer first.")
        return np.array(self.feature_names_out_)
    



    

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from AutoImputer import SkewBasedImputer
    import re
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.model_selection import GridSearchCV

    df = pd.read_csv('./Titanic/data/train.csv')
    df = df.loc[~df['Embarked'].isna()].reset_index(drop=True)

### 1. testing logoddstransformer
    X = df[['Age','Fare']]
    y = df['Survived']

    testing1 = Pipeline([
        ('logodds',LogOddsTransformer(pval=0.3))
    ])

    t1 = testing1.fit_transform(X,y)
    print(testing1.named_steps['logodds'].transformations_pvals_)
    print(testing1.named_steps['logodds'].transformations_)

### 2. testing interactiontransformer
    # interaction_pairs=[
    #     ('Age','Survived'),
    #     ('Fare','Survived')
    # ]
    # testing2 = Pipeline([
    #     ('interaction',InteractionTransformer(interaction_pairs))
    # ])

    # t2 = testing2.fit_transform(df)
    # print(testing2.named_steps['interaction'].get_feature_names_out())


### FunctionTransformer syntax
    # transformer = FunctionTransformer(get_name_title)
    # transformed = transformer.transform(df[['Name']])
    # print(transformed)

### NamedFunctionTransformer syntax
    # transformer = NamedFunctionTransformer(get_name_title)
    # transformed = transformer.transform(df[['Name']])
    # print(transformed)

### 3. testing namedfunctiontransformer
    # def get_name_title(X):
    #     X = X.copy()
    #     X['name_title'] = [re.search(r", (.*?)\.", x).group(1) for x in X.iloc[:,-1]]
    #     return X[['name_title']].astype('category')
    
    # def get_special_title_flg(X):
    #     X = get_name_title(X)
    #     X['special_title_flg'] = ['generic' if x in ['Mr', 'Miss', 'Mrs', 'Ms'] else 'special' for x in X.iloc[:,0]]
    #     return X[['special_title_flg']].astype('category')

    # def get_family_flg(X):
    #     def family_onboard(X):
    #         if X['SibSp'] == 0 and X['Parch'] == 0:
    #             return 'no_fam'
    #         elif X['SibSp'] > 0 and X['Parch'] == 0:
    #             return 'SibSp'
    #         elif X['SibSp'] == 0 and X['Parch'] > 0:
    #             return 'Parch'
    #         elif X['SibSp'] > 0 and X['Parch'] > 0:
    #             return 'big_fam'
    #         else:
    #             return 'NA'
    #     X = X.copy()
    #     X['family_flg'] = X.apply(lambda x: family_onboard(x), axis=1)
    #     return X[['family_flg']].astype('category')
    
    # def get_cabin_count(X):
    #     X = X.copy()
    #     X['cabin_count'] = X.iloc[:, 0].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    #     return X[['cabin_count']]
    
    # def change_to_categorical(X):
    #     return X.astype('category')
    
    # feature_func = [
    #     (get_name_title, 'Name'),
    #     (get_special_title_flg, 'Name'),
    #     (get_family_flg, ['SibSp','Parch']),
    #     (get_cabin_count, 'Cabin'),
    #     (change_to_categorical, 'Pclass')
    # ]
    
    # def transform_features(funcs):
    #     out_trans = []
    #     original_feature_name = []
    #     for tup in funcs:
    #         func_name = tup[0].__name__
    #         feature_in = tup[1] if isinstance(tup[1], list) else [tup[1]]
    #         feature_out = [tup[1] if "change" in func_name else func_name.replace("get_", "")]
    #         func = tup[0]
    #         include_original = False
    #         if "get_" in func_name:
    #             if not set(feature_in).issubset(set(original_feature_name)):
    #                 include_original = True
    #                 original_feature_name = original_feature_name + feature_in

    #         transformer = NamedFunctionTransformer(func=func,
    #                                             feature_names_out_ = feature_out,
    #                                             include_original = include_original)
    #         out_trans.append((feature_out[0], transformer, feature_in))
        
    #     return out_trans
    
    # data_preprocessing = ColumnTransformer(
    #     transformers=transform_features(feature_func)
    #     # ,remainder = 'passthrough'
    #     ,verbose_feature_names_out=False
    # )
    # data_preprocessing.set_output(transform='pandas')

    # X = df.drop(['Survived','PassengerId'], axis=1)
    # y = df['Survived']

    # categorical_feature_cleaning = Pipeline([
    #     ('one-hot encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    # ])

    # continuous_feature_cleaning = Pipeline([
    #     ('scaler', StandardScaler().set_output(transform="pandas")),
    #     ('impute', SimpleImputer(strategy='mean'))
    # ])

    # feature_cleaning = ColumnTransformer(
    #     transformers=[
    #         ('cont_pipeline', continuous_feature_cleaning, ['Age','Fare'])
    #         ,('cat_pipeline', categorical_feature_cleaning, ['name_title','family_flg'])
    #     ]
    #     , verbose_feature_names_out=False
    # )

    # preprocessing_pipeline = Pipeline([
    #     ('data_preprocessing',data_preprocessing),
    #     ('feature_cleaning',feature_cleaning)
    # ])

    # X = df.drop(['Survived','PassengerId'], axis=1)
    # y = df['Survived']
    # scoring = {'acc': 'accuracy',
    #         'precision': 'precision',
    #         'recall': 'recall',
    #         'f1': 'f1',
    #         'roc_auc': 'roc_auc'}

### 3a. testing with logistic regression
    # logit_pipeline = Pipeline([
    #     ('preprocessing', preprocessing_pipeline),
    #     ('model', LogisticRegression())
    # ])

### testing with simple cross validate
    # scores = cross_validate(logit_pipeline, X, y, scoring=scoring, cv = 5)
    # print(scores)

### testing with gridsearchcv
    # param_grid = [
    #     # ElasticNet only valid with saga solver
    #     {
    #         'model__penalty': ['elasticnet'],
    #         'model__C': np.logspace(-3, 2, 10),
    #         'model__solver': ['saga'],  # saga supports elasticnet
    #         'model__max_iter': [100, 200, 500, 1000],
    #         'model__class_weight': [None, 'balanced'],
    #         'model__l1_ratio': np.linspace(0, 1, 5).tolist()  # Required for elasticnet
    #     },
    #     # l1 penalty only works with liblinear or saga
    #     {
    #         'model__penalty': ['l1'],
    #         'model__C': np.logspace(-3, 2, 10),
    #         'model__solver': ['liblinear', 'saga'],  # Only valid solvers for l1
    #         'model__max_iter': [100, 200, 500, 1000],
    #         'model__class_weight': [None, 'balanced'],
    #         'model__l1_ratio': [None]  # Not used for l1
    #     },
    #     # l2 penalty works with most solvers
    #     {
    #         'model__penalty': ['l2'],
    #         'model__C': np.logspace(-3, 2, 10),
    #         'model__solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],  # Valid solvers
    #         'model__max_iter': [100, 200, 500, 1000],
    #         'model__class_weight': [None, 'balanced'],
    #         'model__l1_ratio': [None]  # Not used for l2
    #     },
    #     # No penalty (essentially equivalent to no regularization)
    #     {
    #         'model__penalty': [None],
    #         'model__C': [1],  # Not used when penalty=None
    #         'model__solver': ['lbfgs', 'saga', 'newton-cg'],  # Valid solvers when no penalty
    #         'model__max_iter': [100, 200, 500, 1000],
    #         'model__class_weight': [None, 'balanced'],
    #         'model__l1_ratio': [None]  # Not used for no penalty
    #     }
    # ]
    # log_grid_search = GridSearchCV(
    #     estimator = logit_pipeline,
    #     param_grid = param_grid,
    #     cv = 5,
    #     scoring = scoring,
    #     refit = 'roc_auc',
    #     verbose = 0,
    #     n_jobs = -1
    # )
    # log_grid_search.fit(X, y)
    # print(log_grid_search.best_params_)
    # print(log_grid_search.best_score_)

### 3b. testing with random forest
    # rf_pipeline = Pipeline([
    #     ('preprocessing', preprocessing_pipeline),
    #     ('model', RandomForestClassifier())
    # ])

### testing with simple cross validate
    # scores = cross_validate(rf_pipeline, X, y, scoring=scoring, cv = 5)
    # print(scores)
    
### testing with gridsearchcv
    # param_grid = {
    #     'model__n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
    #     'model__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    #     'model__min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    #     'model__min_samples_leaf': [1, 2, 5],  # Minimum number of samples required to be at a leaf node
    #     'model__max_features': ['sqrt', 'log2', None],  # Number of features to consider for the best split
    #     'model__class_weight': [None, 'balanced'],  # Handling class imbalance
    #     'model__bootstrap': [True, False]  # Whether to bootstrap samples
    # }

    # rf_grid_search = GridSearchCV(
    #     estimator=rf_pipeline,
    #     param_grid = param_grid,
    #     cv = 5,
    #     scoring = scoring,
    #     refit = 'roc_auc',
    #     verbose = 0,
    #     n_jobs = -1
    # )

    # rf_grid_search.fit(X,y)
    # print(rf_grid_search.best_params_)
    # print(rf_grid_search.best_score_)
