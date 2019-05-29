# coding: utf-8

import warnings
from copy import copy

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


class Classifier():

    """
    Parameters
    ----------
    strategy : str, default = "LightGBM"
        The choice for the classifier.
        Available strategies = "LightGBM"
    **params : default = None
        Parameters of the corresponding classifier.
        Examples : n_estimators, max_depth...
    """

    def __init__(self, **params):

        if ("strategy" in params):
            self.__strategy = params["strategy"]
        else:
            self.__strategy = "LightGBM"

        self.__classif_params = {}

        self.__classifier = None
        self.__set_classifier(self.__strategy)
        self.__col = None

        self.set_params(**params)
        self.__fitOK = False


    def get_params(self, deep=True):

        params = {}
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
                                  "`classifier.get_params().keys()`")
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
        self.params = { "objective":'binary',
                        }

        if(strategy == "LightGBM"):
            self.__classifier = LGBMClassifier({self.params},
                n_estimators=500, learning_rate=0.05,
                colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)


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

        # sanity checks
        if((type(df_train) != pd.SparseDataFrame) and
           (type(df_train) != pd.DataFrame)):
            raise ValueError("df_train must be a DataFrame")

        if (type(y_train) != pd.core.series.Series):
            raise ValueError("y_train must be a Series")

        self.__classifier.fit(df_train.values, y_train)
        self.__col = df_train.columns
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

            if (self.get_params()["strategy"] in ["LightGBM"]):

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

    def predict_log_proba(self, df):

        """Predicts class log-probabilities for df.
        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.
        Returns
        -------
        y : array of shape = (n, n_classes)
            The log-probabilities for each class
        """

        try:
            if not callable(getattr(self.__classifier, "predict_log_proba")):
                raise ValueError("predict_log_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict_log_proba(df.values)
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

            # sanity checks
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
