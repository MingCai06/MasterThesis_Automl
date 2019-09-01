import numpy as np
import pandas as pd
import pprint

from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold

from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeadlineStopper  # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback  # Callback to control the verbosity
from skopt.callbacks import DeltaYStopper
from skopt import dummy_minimize

import encoder
from scaling import Scaler
from util import timeit, mprint
from model import Classifier
import time


class randomsearch():
    def __init__(self, models, params, random_state=42, verbose=False):
        self.params = params
        self.keys = models.keys()
        self.verbose = verbose
        self.models = models
        self.random_state = random_state

        self.score_std = []

    @timeit
    def fit(self, X, y_bin, n_iter=10, need_callback=True):
        enc = encoder.Categorical_encoder()
        X = enc.fit_transform(X, y_bin)
        if len(X.dtypes[X.dtypes == 'float'].index) != 0:
            scal = Scaler()
            X = scal.fit_transform(X, y_bin)
        else:
            pass

        # Converting average precision score into a scorer suitable for model selection
        avg_prec = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=False)
        rs = {}
        for key in self.keys:
            mprint("Running RandomSearchCV for %s." % key)
            start = time.time()
            start_cpu = time.process_time()
            model = self.models[key]
            params = self.params[key]

            param_keys, param_vecs = zip(*params.items())
            param_keys = list(param_keys)
            param_vecs = list(param_vecs)

            def objective(param_vec):
                params = dict(zip(param_keys, param_vec))
                model.set_params(**params)
                score = cross_val_score(model, X, y_bin, cv=5, n_jobs=-1, scoring=avg_prec)
                self.score_std.append(np.std(score))
                return -np.mean(score)

                start = time.time()
                start_cpu = time.process_time()

            def on_step(gp_round):
                    scores = np.sort(gp_round.func_vals)
                    score = scores[0]
                   # print("best score: %s" % score)
                    if score == -1:
                        print('Interrupting!')
                        return True

            if need_callback:
                print('Running with Callback function....')

                gp_round = dummy_minimize(func=objective,
                                          dimensions=list(param_vecs),
                                          n_calls=n_iter,
                                          callback=[on_step, DeadlineStopper(60 * 10)],#,on_stepDeltaYStopper(0.000001)
                                          random_state=self.random_state,
                                          verbose=self.verbose)

            else:
                gp_round = dummy_minimize(func=objective,
                                          dimensions=list(param_vecs),
                                          n_calls=n_iter,
                                          random_state=self.random_state,
                                          verbose=self.verbose)

            clock_time = time.time() - start
            cpu_time = time.process_time() - start_cpu

            rm_result = {}
            results = []
            for err, param_vec in zip(gp_round.func_vals, gp_round.x_iters):
                params = dict(zip(param_keys, param_vec))
                results.append({'score': -err, 'params': params})

            best_index = np.argmin(gp_round.func_vals)
            rm_result['best_score'] = -gp_round.fun
            rm_result['best_params'] = dict(zip(param_keys, gp_round.x))
            rm_result['params'] = results
            rm_result['CPU_Time'] = round(cpu_time, 2)
            rm_result['Time_cost'] = round(clock_time, 2)
            rm_result['all_cv_results'] = -gp_round.func_vals
            rm_result['test_score_std'] = self.score_std
            rm_result['best_score_std'] = round(self.score_std[best_index], 4)

            cpu = round(cpu_time)
            clock = round(clock_time)
            best_cv_sd = rm_result['best_score_std']
            cand = len(rm_result['all_cv_results'])
            best_score = round(-gp_round.fun, 4)

            print(f'Finished, took CPU Time: {cpu}s,clock time: {clock}s, candidates checked:{cand} ,best CV score: {best_score} \u00B1 {best_cv_sd}')
            print("")
            rs[key] = rm_result
        bests = pd.DataFrame()
        for key in rs.keys():
            if rs[key]['best_score'] == max(d['best_score'] for d in rs.values()):
                bests = bests.append({'best_score': rs[key]['best_score'],
                                      'best_model': key,
                                      'time': rs[key]['Time_cost']}, ignore_index=True)
                bests = bests.sort_values(by=['time'], ascending=True).reset_index()

                best_model = bests['best_model'][0]
                best_param = rs[best_model]['best_params']

        print("")
        print('######## Congratulations! Here is the Best Parameters: #######')
        print('Best Score is:', rs[best_model]['best_score'])
        print('Best Model is:', best_model)
        pprint.pprint(best_param)
        return rs

    def result_summary(self, random_result):
        r = pd.DataFrame()
        all_row = []
        for k in random_result:
            params = random_result[k]['params']
            for s, t in zip(random_result[k]['all_cv_results'], random_result[k]['test_score_std']):
                row = np.array([k, s, t])
                all_row.append(row)
        r = r.append(all_row)
        r.columns = ['model', 'mean_score', 'std_score']
        res = r.sort_values(by=['mean_score'], ascending=True).reset_index(drop=True)
        res['mean_score'] = res['mean_score'].astype(float)
        res['std_score'] = res['std_score'].astype(float)
        return res
