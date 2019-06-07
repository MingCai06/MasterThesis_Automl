import numpy as np
import pandas as pd
import warnings

import os

class Categorical_encoder():

    def __init__(self, strategy='label_encoding'):

        self.strategy = strategy
        self.__Lcat = []
        self.__Lnum = []
        self.__Enc = dict()
        self.__fitOK = False


    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                }


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder "
                              "Categorical_encoder. Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)


    def fit(self, df_train, y_train):


        self.__Lcat = df_train.dtypes[df_train.dtypes == 'object'].index
        self.__Lnum = df_train.dtypes[df_train.dtypes != 'object'].index

        if (len(self.__Lcat) == 0):
            self.__fitOK = True

        else:

            #################################################
            #                Label Encoding
            #################################################

            if (self.strategy == 'label_encoding'):

                for col in self.__Lcat:

                    d = dict()
                    levels = list(df_train[col].unique())
                    nan = False

                    if np.NaN in levels:
                        nan = True
                        levels.remove(np.NaN)

                    for enc, level in enumerate([np.NaN]*nan + sorted(levels)):
                        d[level] = enc  # TODO: Optimize loop?

                    self.__Enc[col] = d

                self.__fitOK = True

            #################################################
            #                One Hot Encodding
            #################################################

            elif (self.strategy == 'one_hot_encoding'):

                for col in self.__Lcat:
                    # TODO: Optimize?
                    self.__Enc[col] = list(df_train[col].dropna().unique())

                self.__fitOK = True


            else:

                raise ValueError("Strategy for categorical encoding is not valid")

        return self


    def fit_transform(self, df_train, y_train):

        self.fit(df_train, y_train)

        return self.transform(df_train)


    def transform(self, df):

        if self.__fitOK:

            if len(self.__Lcat) == 0:
                return df[self.__Lnum]

            else:

                #################################################
                #                Label Encoding
                #################################################

                if (self.strategy == 'label_encoding'):

                    for col in self.__Lcat:

                        # Handling unknown levels
                        unknown_levels = list(set(df[col].values) -
                                              set(self.__Enc[col].keys()))

                        if (len(unknown_levels) != 0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                # TODO: make sure no collisions introduced
                                d[unknown_level] = new_enc
                                self.__Enc[col] = d

                    if (len(self.__Lnum) == 0):
                        return pd.concat(
                            [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).values,
                                columns=[col], index=df.index
                                         ) for col in self.__Lcat],
                            axis=1)[df.columns]
                    else:
                        return pd.concat(
                            [df[self.__Lnum]] +
                            [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).values,
                                columns=[col],
                                index=df.index
                                ) for col in self.__Lcat],
                            axis=1)[df.columns]

                #################################################
                #                 One Hot Encodding
                #################################################

                elif (self.strategy == 'one_hot_encoding'):

                    sub_var = []
                    missing_var = []

                    for col in self.__Lcat:

                        # Handling unknown and missing levels
                        unique_levels = set(df[col].values)
                        sub_levels = unique_levels & set(self.__Enc[col])
                        missing_levels = [col + "_" + str(s)
                                          for s in list(set(self.__Enc[col]) - sub_levels)]
                        sub_levels = [col + "_" + str(s)
                                      for s in list(sub_levels)]

                        sub_var = sub_var + sub_levels
                        missing_var = missing_var + missing_levels

                    if (len(missing_var) != 0):

                        return pd.SparseDataFrame(
                            pd.concat(
                                [pd.get_dummies(df,
                                                sparse=True)[list(self.__Lnum) +
                                                             sub_var]] +
                                [pd.DataFrame(np.zeros((df.shape[0],
                                                        len(missing_var))),
                                              columns=missing_var,
                                              index=df.index)],
                                axis=1
                            )[list(self.__Lnum)+sorted(missing_var+sub_var)])

                    else:

                        return pd.get_dummies(df, sparse=True)[list(self.__Lnum) + sorted(sub_var)]
        else:

            raise ValueError("Call fit or fit_transform function before")