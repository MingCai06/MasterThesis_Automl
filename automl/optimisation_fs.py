# Importing core libraries
from time import time
import pprint
import joblib

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Model selection
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
from util import log, timeit, mprint, write_result
from encoder import Categorical_encoder
from scaling import Scaler
from model import Classifier
from feature_selection import feature_selector


class Optimiser():

    """Optimises hyper-parameters
    - Estimator (classifier )
    Parameters
    ----------
    scoring : str, callable or None. default: None
        A string or a scorer callable object.
        If None, "auc" is used for classification
        Available scorings for classification : {"accuracy","roc_auc", "f1",
        "log_loss", "precision", "recall"}
    n_folds : int, default = 3
        The number of folds for cross validation (stratified for classification)
    random_state : int, default = 42
        Pseudo-random number generator state used for shuffling
    save_result : bool, default = False
        weather models be saved
    verbose : bool, default = True
        Verbose mode
    Return
    ----------
    final_result : dict, result for all surrogated models
        - final_result['GP'] 
        - final_result['RF'] 
    ----------
    Example
    ----------
    from skopt.space import Real, Categorical, Integer
    from optimisation import Optimiser

    optimiser = Optimiser()

    df_train = data['train'] #df_test = data['test']
    y = data['target'] #y_test = data['y_test']
    search_space_LGB = Classifier(strategy = "LightGBM").get_search_spaces()
    search_space_RF  = Classifier(strategy = "RandomForest").get_search_spaces()

    best_param,results = optimiser.optimise_step(space = [(search_space_LGB_fs,5),(search_space_RF,5)],
                                df_train=X_train,
                                df_target=y_train,
                                max_evals = 5,
                                set_callbacks=True)
    """

    def __init__(self, scoring=None,
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

        if self.to_path is True:
            warnings.warn("Optimiser will save all your fitted models result ")

        self.cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

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

    @timeit
    def optimise_step(self, space, df_train, df_target, max_evals=20, set_callbacks=True):
        """Evaluates the data.
        Build the pipeline. If no parameters are set, default configuration for
        each step is used
        Parameters
        ----------
        space : dict, default = None.
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features.
        y_train : pandas series of shape = (n_train,)
            The numerical encoded target for classification tasks.
        max_evals : int, default = 20, max evaluation times
        set_callbacks (opt): bool,default: True
             If callable then callback(res) is called after each call to func. If list of callables, then each callable in the list is called.
        ----------
        Returns
        ---------
        result : dict
            - result['best_score'] : Best Score after Tuning
            - result['best_score_std'] : Standar Divation of best score
            - result['best_parmas'] : Best parameters
            - result['params'] : all paramsters (# = checked candicated)
            - result['time_cost(s)'] : total time of finding out the best parameters
            - result['all_cv_results'] : all cv results
            - result['mean_score_time'] : time for each cv result
        """
        # checke parallel strategy

        ce = Categorical_encoder()
        X = ce.fit_transform(df_train, df_target)

        if self.perform_scaling is True:
            scal = Scaler()
            X = scal.fit_transform(X, df_target)

        mid_result = {}
        tuning_result = {}

#        lgb = Classifier(strategy="LightGBM").get_estimator()
#        rf = Classifier(strategy="RandomForest").get_estimator()

        # Creating a correct space for skopt
        if (space is None):
            warnings.warn(
                "Space is empty. Please define a search space. "
                "Otherwise, call the method 'evaluate' for custom settings")
            return dict()

        else:

            if (len(space) == 0):
                warnings.warn(
                    "Space is empty. Please define a search space. "
                    "Otherwise, call the method 'evaluate' for custom settings")
                return dict()

            else:
                search_spaces = space

                # Initialize a pipeline
                fs = None
                for i in range(len(search_spaces)):
                    for p in search_spaces[i][0].keys():
                        if (p.startswith("fs__")):
                            fs = feature_selector()
                        else:
                            pass

                # Do we need to cache transformers?
                cache = False

                if (fs is not None):
                    if ("fs__strategy" in search_spaces):
                        if(search_spaces["fs__strategy"] != "variance"):
                            cache = True
                        else:
                            pass
                else:
                    pass
                mprint(f'Start turning Hyperparameters .... ')
                print("")
                print(">>> Categorical Features have encoded with :" + str({'strategy': ce.strategy}))
                print("")
                if self.perform_scaling is True:
                    print(">>> Numerical Features have encoded with :" + scal.__class__.__name__)
                    print("")

                for baseEstimator in ['GP', 'RF']:
                    # Pipeline creation

                    lgb = Classifier(strategy="LightGBM").get_estimator()
                  #  rf = Classifier(strategy="RandomForest").get_estimator()
                  #  svc = Classifier(strategy="SVC").get_estimator()

                    if (fs is not None):
                        if cache:
                            pipe = Pipeline([('fs', fs), ('model', lgb)], memory=self.to_path)
                        else:
                            pipe = Pipeline([('fs', fs), ('model', lgb)])
                    else:
                        if cache:
                            pipe = Pipeline([('model', lgb)], memory=self.to_path)
                        else:
                            pipe = Pipeline([('model', lgb)])

                    if (self.parallel_strategy is True):
                        opt = BayesSearchCV(pipe,
                                            search_spaces=search_spaces,
                                            scoring=self.scoring,
                                            cv=self.cv,
                                            n_points=10,
                                            n_jobs=-1,
                                            #          n_iter=max_evals,
                                            return_train_score=False,
                                            optimizer_kwargs={'base_estimator': baseEstimator,
                                                              "acq_func": "EI"},
                                            random_state=self.random_state,
                                            verbose=self.verbose)
                    else:
                        opt = BayesSearchCV(pipe,
                                            search_spaces=search_spaces,
                                            scoring=self.scoring,
                                            cv=self.cv,
                                            n_points=10,
                                            n_jobs=1,
                                            #         n_iter=max_evals,
                                            return_train_score=False,
                                            optimizer_kwargs={'base_estimator': baseEstimator,
                                                              "acq_func": "EI"},
                                            random_state=self.random_state,
                                            verbose=self.verbose)

                    if set_callbacks is True:

                        mid_result = self.report_perf(opt, X, df_target, ' with baseEstimator:' + baseEstimator,
                                                      callbacks=[DeltaXStopper(0.0001)
                                                                 #, DeadlineStopper(60 * 5)
                                                                 ])
                    else:
                        mid_result = self.report_perf(opt, X, df_target, ' with baseEstimator: ' + baseEstimator,
                                                      )
                    tuning_result[baseEstimator] = mid_result

        for key in tuning_result.keys():
            if tuning_result[key]['best_score'] == max(d['best_score'] for d in tuning_result.values()):
                best_base_estimator = key
                best_param = tuning_result[best_base_estimator]['best_parmas']
        print("")
        print('######## Congratulations! Here is the Best Parameters: #######')
        print('Best Score is:', tuning_result[best_base_estimator]['best_score'])
        print(pipe.get_params()['model'].__class__.__name__ + ' with baseEstimator ' + best_base_estimator)
        pprint.pprint(best_param)
        return best_param, tuning_result

    # Define Call back function
    # def on_step(self, optim_result):
    #     score = opt.best_score_
    #     score_std = opt.cv_results_['std_test_score'][opt.best_index_]
    #     if score >= 0.99:
    #         print('Best Score >0.99,Interrupting!')
    #         return True

    # Reporting util for different optimizers
    def report_perf(self, optimizer, X, y, title, callbacks=None):
        """
        optimizer = a sklearn or a skopt optimizer
        X = the training set 
        y = our target
        title = a string label for the experiment
        """
        start = time()
        if callbacks:
            mprint(f'start tuning {title}...')

            optimizer.fit(X, y, callback=callbacks)
        else:
            mprint(f'start tuning {title}...')

            optimizer.fit(X, y)

        time_cost = time() - start
        result = {}
        result['best_score'] = optimizer.best_score_
        result['best_score_std'] = optimizer.cv_results_['std_test_score'][optimizer.best_index_]
        result['best_parmas'] = optimizer.best_params_
        result['params'] = optimizer.cv_results_['params']
        result['time_cost(s)'] = round(time_cost, 0)
        result['all_cv_results'] = optimizer.cv_results_['mean_test_score'][:]
        result['mean_score_time'] = optimizer.cv_results_['mean_score_time'][:]
        result['mean_score_time'] = optimizer.cv_results_['mean_score_time'][:]
        result['CV'] = optimizer.cv_results_
        print("")
        print('>' + title + ':')
        time_cost = round(result['time_cost(s)'], 0)
        cand = len(result['all_cv_results'])
        best_cv = round(result['best_score'], 4)
        best_cv_sd = round(result['best_score_std'], 4)
        print(f'took{time_cost}s, candidates checked:{cand},best CV score: {best_cv} \u00B1 {best_cv_sd}')
        print("")

        return result
