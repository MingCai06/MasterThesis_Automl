from joblib import Parallel, delayed
import multiprocessing
import time
from sklearn.pipeline import Pipeline
# Model selection
from sklearn.model_selection import StratifiedKFold
# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV

import encoder
from scaling import Scaler
from util import timeit, mprint
from model import Classifier

def model_bo_parallel():

    y = data['target'] #y_test = data['y_test']

    random_state = 42
    verbose = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    enc = encoder.Categorical_encoder()
    X = enc.fit_transform(data['train'],y)

    if len(X.dtypes[X.dtypes == 'float'].index) != 0:
        scal = Scaler()
        X = scal.fit_transform(X, y)
        perform_scaling is True
    else:
        pass
 #   scal = Scaler()
 #   X = scal.fit_transform(X, y)

    ee = ['GP', 'RF']


    def optm(baseEstimator):
      # Reporting util for different optimizers
        def report_perf(optimizer, X, y, title, callbacks=None):

            start = time.time()
            start_cpu = time.process_time()

            if callbacks:
                print("")
                mprint(f'start tuning {title}...')
                optimizer.fit(X, y, callback=callbacks)
            else:
                print("")
                mprint(f'start tuning {title}...')
                optimizer.fit(X, y)

            time_cost_CPU = time.process_time() - start_cpu
            time_cost = time.time() - start
            result = {}
            result['best_score'] = optimizer.best_score_
            result['best_score_std'] = optimizer.cv_results_['std_test_score'][optimizer.best_index_]
            result['best_parmas'] = optimizer.best_params_
            result['params'] = optimizer.cv_results_['params']
            result['CPU_Time'] = round(time_cost_CPU, 0)
            result['Time_cost'] = round(time_cost, 0)
            result['all_cv_results'] = optimizer.cv_results_['mean_test_score'][:]
            result['CV'] = optimizer.cv_results_
            print("")

            time_cost_CPU = round(result['CPU_Time'], 0)
            time_cost = round(result['Time_cost'], 0)
            cand = len(result['all_cv_results'])
            best_cv = round(result['best_score'], 8)
            best_cv_sd = round(result['best_score_std'], 4)
            print(f'took CPU Time: {time_cost_CPU}s,clock time: {time_cost}s, candidates checked:{cand} ,best CV score: {best_cv} \u00B1 {best_cv_sd}')
            print("")
            return result

        # Initialize a pipeline with a model
        lgb = Classifier(strategy="LightGBM").get_estimator()
        pipe = Pipeline([('model', lgb)])
        search_space_LGB = Classifier(strategy = "LightGBM").get_search_spaces(need_feature_selection=False)

        mid_res = {}
        #  for baseEstimator in ['GP']:
        opt = BayesSearchCV(pipe,
                            search_spaces=[(search_space_LGB,10)],
                            scoring=roc_auc,
                            cv=skf,
                            n_points=6,
                            n_jobs=-1,
                            return_train_score=False,
                            optimizer_kwargs={'base_estimator': baseEstimator,
                                              "acq_func": "EI"},
                            random_state=random_state,
                            verbose=verbose)

        result = report_perf(opt, X, y,'BayesSearchCV_'+baseEstimator,
                                                   #      callbacks=[DeltaXStopper(0.0001)]
                                                 )
        mid_res[baseEstimator] = result
        return mid_res

    num_cores = multiprocessing.cpu_count()

    @timeit
    def test_model_parallel ():
        start = time.time()
        start_cpu = time.process_time()
        results = Parallel(n_jobs=-1,verbose=0)(delayed(optm)(baseEstimator) for baseEstimator in ee)
        time_cost = time.time() - start
        time_cost_CPU = time.process_time() - start_cpu
        return results,time_cost, time_cost_CPU

    model_parallel,model_time_cost,model_cpu_time = test_model_parallel()
    print('Total time:',round(model_time_cost,1))
    print('Total CPU time:',round(model_cpu_time,1))
    print('GP:' ,model_parallel[0]['GP']['best_score'])
    print('RF:' ,model_parallel[1]['RF']['best_score'])

    model_parallel_results = {}
    model_parallel_results['GP'] = model_parallel[0]['GP']
    model_parallel_results['RF'] = model_parallel[1]['RF']
    model_parallel_results['Total_time_cost'] = round(model_time_cost, 1)
    model_parallel_results['Total_CPU_time'] = round(model_cpu_time, 1)

    return model_parallel_results
#model_parallel_results
