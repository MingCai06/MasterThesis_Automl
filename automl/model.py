# coding: utf-8
import warnings
from copy import copy

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# utility for early stopping with a validation set
from sklearn.model_selection import train_test_split

# Skopt functions
from skopt.space import Real, Categorical, Integer


class Classifier():

    """
    Parameters
    ----------
    strategy : str, default = "LightGBM"
        The choice for the classifier.
        Available strategies = "LightGBM" "RandomForest"
    **params : default = None
        Parameters of the corresponding classifier.
        Examples : n_estimators, max_depth...
    """

    def __init__(self, **params):

        if ("strategy" in params):
            if params["strategy"] in ['LightGBM', 'RandomForest', 'SVC']:
                self.__strategy = params["strategy"]
            else:
                raise ValueError("Strategy invalid. Please choose between "
                                 "'LightGBM' or 'RandomForest'")
        else:
            print(
                'You donot give any Model Strategy, defalut model <LightGBM> will be used!')
            self.__strategy = "LightGBM"

        self.__classif_params = {}
        self.__classifier = None
        self.__set_classifier(self.__strategy)
        self.__col = None

        self.set_params(**params)
        self.__fitOK = False

    def get_params(self, deep=True):

        if self.__strategy == "LightGBM":
            params = {
                "objective": "binary",
                "verbosity": 1,
                "random_state": 42
            }
            params["strategy"] = self.__strategy
            params.update(self.__classif_params)

        else:
            params["strategy"] = self.__strategy
            params.update(self.__classif_params)

        return params

    def set_params(self, **params):

        self.__fitOK = False

        if 'strategy' in params.keys():
            self.__set_classifier(params['strategy'])

            for k, v in self.__classif_params.items():
                if k not in self.get_params().keys():
                    warnings.warn("Invalid parameter for classifier " +
                                  str(self.__strategy) +
                                  ". Parameter IGNORED. Check the list of "
                                  "available parameters with "
                                  "`Classifier().get_params().keys()`")
                else:
                    setattr(self.__classifier, k, v)

        for k, v in params.items():
            if(k == "strategy"):
                pass
            else:
                if k not in self.__classifier.get_params().keys():
                    warnings.warn("Invalid parameter for classifier " +
                                  str(self.__strategy) +
                                  ". Parameter IGNORED. Check the list of "
                                  "available parameters with "
                                  "`classifier.get_params().keys()`")
                else:
                    setattr(self.__classifier, k, v)
                    self.__classif_params[k] = v

    def __set_classifier(self, strategy):

        self.__strategy = strategy

        if(strategy == "LightGBM"):
            self.__classifier = LGBMClassifier(
                objective="binary",  random_state=42)

        elif (strategy == "RandomForest"):
            self.__classifier = RandomForestClassifier(random_state=0)

        elif (strategy == "SVC"):

            self.__classifier = SVC(max_iter=1000, random_state=0)
        # Here can add other classfier
        # elif(self.strategy =="")

        else:
            raise ValueError(
                "Strategy invalid. Please choose'LightGBM'")

    def fit(self, df_train, y_train):
        """Fits Classifier.
        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features.
        y_train : pandas series of shape = (n_train,)
            The numerical encoded target for classification tasks.
        Returns
        -------
        object
            self
        """

        #  checks
        if((type(df_train) != pd.SparseDataFrame) and
           (type(df_train) != pd.DataFrame)):
            raise ValueError("df_train must be a DataFrame")

        if (type(y_train) != pd.core.series.Series):
            raise ValueError("y_train must be a Series")

        df_train_train, df_val, y_train_train, y_val = train_test_split(
            df_train, y_train, test_size=0.33)

        if(strategy == "LightGBM"):
            self.__classifier.fit(df_train_train.values, y_train_train,
                                  eval_set=[(df_val.values, y_val)],
                                  early_stopping_rounds=100)
            self.__col = df_train_train.columns
            self.__fitOK = True

        elif (strategy == "RandomForest"):
            self.__classifier.fit(df_train.values, y_train)
            self.__col = df_train.columns
            self.__fitOK = True

        elif (strategy == "SVC"):
            self.__classifier.fit(df_train.values, y_train)
            self.__col = df.train.columns
            self.__fitOK = True

        return self

    def feature_importances(self):
        """Computes feature importances.
        Classifier must be fitted before.
        Returns
        -------
        dict
            Dictionnary containing a measure of feature importance (value) for
            each feature (key).
        """

        if self.__fitOK:

            if (self.get_params()["strategy"] in ["LightGBM", "RandomForest"]):

                importance = {}
                f = self.get_estimator().feature_importances_

                for i, col in enumerate(self.__col):
                    importance[col] = f[i]
            else:

                importance = {}

            return importance

        else:

            raise ValueError("You must call the fit function before !")

    def predict(self, df):
        """Predicts the target.
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.
        Returns
        -------
        array of shape = (n, )
            The encoded classes to be predicted.
        """

        try:
            if not callable(getattr(self.__classifier, "predict")):
                raise ValueError("predict attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict(df.values)

        else:
            raise ValueError("You must call the fit function before !")

    def predict_proba(self, df):
        """Predicts class probabilities for df.
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.
        Returns
        -------
        array of shape = (n, n_classes)
            The probabilities for each class
        """

        try:
            if not callable(getattr(self.__classifier, "predict_proba")):
                raise ValueError("predict_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict_proba(df.values)
        else:
            raise ValueError("You must call the fit function before !")

    def transform(self, df):
        """Transforms df.
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.
        Returns
        -------
        pandas dataframe of shape = (n, n_selected_features)
            The transformed dataset with its most important features.
        """

        try:
            if not callable(getattr(self.__classifier, "transform")):
                raise ValueError("transform attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.transform(df.values)
        else:
            raise ValueError("You must call the fit function before !")

    def score(self, df, y, sample_weight=None):
        """Returns the mean accuracy.
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.
        y : pandas series of shape = (n,)
            The numerical encoded target for classification tasks.
        Returns
        -------
        float
            Mean accuracy of self.predict(df) wrt. y.
        """

        try:
            if not callable(getattr(self.__classifier, "score")):
                raise ValueError("score attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            if(type(y) != pd.core.series.Series):
                raise ValueError("y must be a Series")

            return self.__classifier.score(df.values, y, sample_weight)
        else:
            raise ValueError("You must call the fit function before !")

    def get_estimator(self):
        return copy(self.__classifier)

    def get_search_spaces(self, need_feature_selection=False):
        model = self.get_estimator()
        if need_feature_selection is True:
            search_params = {
                "LightGBM": {
                    "model": Categorical([model]),
                    "model__class_weight": Categorical(categories=['balanced', None]),
                    "model__learning_rate": Real(0.01, 1.0),
                    "model__boosting_type": Categorical(categories=['gbdt', 'dart']),
                    "model__n_estimators": Integer(10, 500),
                    "model__min_samples_split": Integer(2, 10),
                    "model__min_samples_leaf": Integer(1, 10),
                    "model__min_child_weight": Integer(0, 50),
                    "fs__strategy": Categorical(categories=[None, 'l1', 'lgb_feature_importance', "variance"])
                },
                "RandomForest": {
                    "model": Categorical([model]),
                    "model__n_estimators": Integer(10, 200),
                    "model__min_samples_split": Real(0.01, 0.5),
                    "model__min_samples_leaf": Integer(1, 10),
                    "model__max_features": Categorical(categories=['sqrt', 'log2', None]),
                    "fs__strategy": Categorical(categories=[None, 'l1', 'lgb_feature_importance', "variance"])
                },
                "SVC": {
                    "model": Categorical([model]),
                    "model__class_weight": Categorical(categories=['balanced', None]),
                    'model__C': (1e-6, 1e+6, 'log-uniform'),
                    'model__gamma': (1e-6, 1e+1, 'log-uniform'),
                    'model__degree': (1, 8),
                    'model__kernel': ['linear', 'poly', 'rbf'],
                    "fs__strategy": Categorical(categories=[None, 'l1', 'lgb_feature_importance', "variance"])
                }
            }
        else:
            search_params = {
                "LightGBM": {
                    "model": Categorical([model]),
                    "model__is_unbalance": Categorical(categories=[True, False]),
                    "model__learning_rate": Real(0.01, 1.0),
                    "model__boosting_type": Categorical(categories=['gbdt', 'dart']),
                    "model__n_estimators": Integer(10, 500),
                    "model__max_bin": Integer(100, 10000),
                    "model__feature_fraction": Real(0.01, 1.0, 'uniform'),
                    "model__reg_alpha": Real(1e-9, 1000, 'log-uniform'),
                    "model__reg_lambda": Real(1e-9, 1000, 'log-uniform'),
                },
                "RandomForest": {
                    "model": Categorical([model]),
                    "model__n_estimators": Integer(10, 200),
                    "model__min_samples_split":  Real(0.01, 0.5),
                    "model__min_samples_leaf":  Real(0.01, 0.5),
                    "model__max_features": Categorical(categories=['sqrt', 'log2'])
                },
                "SVC": {
                    "model": Categorical([model]),
                    "model__class_weight": Categorical(categories=['balanced', None]),
                    'model__C': (1e-6, 1e+6, 'log-uniform'),
                    'model__gamma': (1e-6, 1e+1, 'log-uniform'),
                    'model__degree': (1, 8),
                    'model__kernel': ['linear', 'poly', 'rbf'],
                }
            }

        strategy = self.__strategy
        params = search_params[strategy]
        return params
