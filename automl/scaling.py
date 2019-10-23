import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler


class Scaler():

    """Scaling for both numerical features.
    """

    def __init__(self):

        self.__Lcat = []
        self.__Lnum = []
        self.__scal = None
        self.__fitOK = False

    def fit(self, df_train, y_train=None):

        self.__Lcat = df_train.dtypes[df_train.dtypes != np.float].index
        self.__Lnum = df_train.dtypes[df_train.dtypes == np.float].index

        # Dealing with numerical features
        self.__scal = StandardScaler(copy=True, with_mean=True, with_std=True)
        if (len(self.__Lnum) != 0):
            self.__scal.fit(df_train[self.__Lnum])
        else:
            pass

        self.__fitOK = True

        return self

    def fit_transform(self, df_train, y_train=None):

        self.fit(df_train, y_train)

        return self.transform(df_train)

    def transform(self, df):

        if(self.__fitOK):

            if(len(self.__Lnum) != 0):
                if (len(self.__Lcat) != 0):
                    return pd.concat((pd.DataFrame(self.__scal.transform(df[self.__Lnum]),
                                                   columns=self.__Lnum, index=df.index),
                                      df[self.__Lcat]),
                                     axis=1)[df.columns]
                else:
                    return self.__scal.transform(df[self.__Lnum])

        else:
            raise ValueError("Call fit or fit_transform function before")
