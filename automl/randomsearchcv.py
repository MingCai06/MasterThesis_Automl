# Importing core libraries
import time
import pprint
import joblib
import pandas as pd
import numpy as np
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Model selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args  # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper  # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback  # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper  # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta

# import local function
from util import log, timeit, mprint, dump_result, load_result
from encoder import Categorical_encoder
from scaling import Scaler
from model import Classifier
from feature_selection import feature_selector


class RandomOptimiser():

    def __init__(self, models, params,
                 scoring=None,
                 n_folds=5,
                 random_state=42,
                 verbose=True,
                 to_path="save",
                 perform_scaling=True,
                 parallel_strategy=True):

        self.scoring = scoring
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.to_path = to_path
        self.perform_scaling = perform_scaling
        self.parallel_strategy = parallel_strategy
        self.time_cost_CPU = None

        if self.to_path is True:
            warnings.warn("Optimiser will save all your fitted models result ")

        # Default scoring for classification
        if (self.scoring is None):
            self.scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

        elif (self.scoring == "log_loss"):
            self.scoring = 'log_loss'

        else:
            if (type(self.scoring) == str):
                if (self.scoring in ["accuracy", "roc_auc", "f1",
                                     "log_loss", "precision", "recall"]):
                    pass
                else:
                    warnings.warn("Invalid scoring metric. "
                                  "auc is used instead.")
                    self.scoring is None
            else:
                pass
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.random_searches = {}

    def fit(self, df_train, df_target, n_jobs=-1, verbose=0, n_iter=10):
        tuning_result = {}
        ce = Categorical_encoder()
        X = ce.fit_transform(df_train, df_target)

        if self.perform_scaling is True:
            scal = Scaler()
            X = scal.fit_transform(X, df_target)

        for key in self.keys:
            print("Running RandomSearchCV for %s." % key)
            start = time.time()
            start_cpu = time.process_time()
            model = self.models[key]
            params = self.params[key]
            rs = RandomizedSearchCV(model, params, cv=self.n_folds, n_jobs=n_jobs, n_iter=n_iter,
                                    verbose=verbose, scoring=self.scoring, refit=True,
                                    return_train_score=True)
            rs.fit(X, df_target)
            self.random_searches[key] = rs
            self.time_cost_CPU = time.process_time() - start_cpu
            time_cost = time.time() - start
            result = {}
            result['best_score'] = rs.best_score_
            result['best_score_std'] = rs.cv_results_['std_test_score'][rs.best_index_]
            result['best_parmas'] = rs.best_params_
            result['params'] = rs.cv_results_['params']
            result['CPU_Time'] = round(self.time_cost_CPU, 0)
            result['Time_cost'] = round(time_cost, 0)
            result['all_cv_results'] = rs.cv_results_['mean_test_score'][:]
            result['CV'] = rs.cv_results_
            #        print('>' + title + ':')
            time_cost_CPU = round(result['CPU_Time'], 0)
            time_cost = round(result['Time_cost'], 0)
            cand = len(result['all_cv_results'])
            best_cv = round(result['best_score'], 8)
            best_cv_sd = round(result['best_score_std'], 4)
            print(f'took CPU Time: {time_cost_CPU}s, candidates checked:{cand} ,best CV score: {best_cv} \u00B1 {best_cv_sd}')
            print("")
            tuning_result[key] = result
        return tuning_result

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.random_searches:
            print(k)
            params = self.random_searches[k].cv_results_['params']
            scores = []
            for i in range(self.random_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.random_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

    def get_params(self, deep=True):

        return {'scoring': self.scoring,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'verbose': self.verbose,
                'save_result': self.save_result,
                'perform_scaling': self.perform_scaling,
                'parallel_strategy': self.parallel_strategy}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for optimiser Optimiser. "
                              "Parameter IGNORED. Check the list of available "
                              "parameters with `optimiser.get_params().keys()`")
            else:
                setattr(self, k, v)
