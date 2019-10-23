import numpy as np
import pandas as pd
import pprint

from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold

from skopt.space import Real, Categorical, Integer
# Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import DeadlineStopper
from skopt.callbacks import VerboseCallback  # Callback to control the verbosity
from skopt.callbacks import DeltaYStopper
from skopt.callbacks import check_callback
from skopt import dummy_minimize
from sk_utils import eval_callbacks

#from skopt import dummy_minimize
from skopt import dump, load
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
        self.final_best_score = 0
        self.final_best_std = 0
        self.cand = 0
        self.clock = 0

    @timeit
    def fit(self, X, y_bin, n_iter=10, need_callback=True):

        enc = encoder.Categorical_encoder()
        X = enc.fit_transform(X, y_bin)
        if len(X.dtypes[X.dtypes == 'float'].index) != 0:
            scal = Scaler()
            X = scal.fit_transform(X, y_bin)
        else:
            pass

        np.random.seed(55)
        s_lgb, f_lgb = 0, 0
        s_svc, f_svc = 0, 0

        p_lgb = np.random.beta(s_lgb + 1.0, f_lgb + 1.0)
        p_svc = np.random.beta(s_svc + 1.0, f_svc + 1.0)

        # Converting average precision score into a scorer suitable for model selection
        avg_prec = make_scorer(
            roc_auc_score, greater_is_better=True, needs_proba=False)
        rs = {}
        adj_iter = int(n_iter / 10)
        mprint("Running RandomSearchCV...")

        while (adj_iter > 0):
            start = time.time()
            start_cpu = time.process_time()

            if (p_lgb >= p_svc):
                key = "LGBMClassifier"
            else:
                key = "SVC"
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
                score = cross_val_score(
                    model, X, y_bin, cv=5, n_jobs=-1, scoring=avg_prec)
                self.score_std.append(np.std(score))
                return -np.mean(score)

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
                                          n_calls=10,
                                          callback=[on_step  # , DeadlineStopper(60 * 10)
                                                    ],  # ,on_stepDeltaYStopper(0.000001)
                                          random_state=self.random_state,
                                          verbose=self.verbose)

            else:
                gp_round = dummy_minimize(func=objective,
                                          dimensions=list(param_vecs),
                                          n_calls=10,
                                          random_state=self.random_state,
                                          verbose=self.verbose)

            rm_result = {}
            results = []
            score = []
            p = []
            for err, param_vec in zip(gp_round.func_vals, gp_round.x_iters):
                params = dict(zip(param_keys, param_vec))
                mparams = dict({"model": model.__class__.__name__}, **params)
                score.append(-err)
                p.append(mparams)
            bes = np.argmax(score)
            best_index = np.argmin(gp_round.func_vals)
            rm_result['best_score'] = -gp_round.fun
            rm_result['best_params'] = p[bes]
            rm_result['params'] = params
            rm_result['all_cv_results'] = -gp_round.func_vals

            clock_time = time.time() - start
            cpu_time = time.process_time() - start_cpu
            rm_result['test_score_std'] = self.score_std
            rm_result['best_score_std'] = round(self.score_std[best_index], 4)
            rm_result['CPU_Time'] = round(cpu_time, 2)
            rm_result['Time_cost'] = round(clock_time, 2)
            cand = len(rm_result['all_cv_results'])
            cpu = round(cpu_time)
            clock = round(clock_time)
            best_cv_sd = rm_result['best_score_std']
            best_score = round(-gp_round.fun, 4)
            adj_iter -= 1
            rs[adj_iter] = rm_result

            if rs[adj_iter]['best_score'] > np.mean([d['all_cv_results'] for d in rs.values()]):
                if gp_round.x.__class__.__name__ == "SVC":
                    f_svc += 1
                elif gp_round.x.__class__.__name__ == "LGBMClassifier":
                    f_lgb += 1
            else:
                if gp_round.x.__class__.__name__ == "SVC":
                    s_svc += 1
                elif gp_round.x.__class__.__name__ == "LGBMClassifier":
                    s_lgb += 1

            p_lgb = np.random.beta(s_lgb + 1.0, f_lgb + 1.0)
            p_svc = np.random.beta(s_svc + 1.0, f_svc + 1.0)

            self.cand += int(cand)
            self.clock += clock

            if self.final_best_score < best_score:
                self.final_best_score = best_score
                self.final_best_std = best_cv_sd
            else:
                self.final_best_score = self.final_best_score
                self.final_best_std = self.final_best_std

        print(
            f'Finished, took CPU Time: {cpu}s,clock time: {self.clock}s, candidates checked:{self.cand} ,best CV score: {self.final_best_score} \u00B1 {self.final_best_std}')
        print("")

        bests = pd.DataFrame()
        for key in rs.keys():
            if rs[key]['best_score'] == max(d["best_score"] for d in rs.values()):
                bests = bests.append({'best_score': rs[key]['best_score'],
                                      'best_param': rs[key]['best_params'],
                                      "ind": key,
                                      'time': rs[key]['Time_cost']}, ignore_index=True)
                bests = bests.sort_values(
                    by=['time'], ascending=True).reset_index(drop=True)
                best_param = rs[key]['best_params']
                score = rs[key]['best_score']
        print("")
        print('######## Congratulations! Here is the Best Parameters: #######')
        print('Best Score is:', score)
        print('Best Model is:')
        pprint.pprint(best_param)
       # rs["Time_cost"] = self.clock
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
        res = r
        #res = r.sort_values(by=['mean_score'], ascending=True).reset_index(drop=True)
        res['mean_score'] = res['mean_score'].astype(float)
        res['std_score'] = res['std_score'].astype(float)
        return res
