import matplotlib.pyplot as plt
import numpy as np
from util import log, timeit, mprint
from optimisation import Optimiser


def plot_all_cv_result(final_result, kpi='all_cv_results', max_iters=20):
    '''
    plot the result of fisrt optimizer 
    Example
    ----------
    plot_final_result_step1(step_1_result)

    '''

    markers = '.*xv'
    for i, marker in zip(final_result.keys(), markers):
        l = int(len(final_result[i]['all_cv_results']) / 2)
        iterations = range(1, l + 1)
        lgb_mins = sorted(final_result[i][kpi][:l])
        svc_mins = sorted(final_result[i][kpi][l:])

        plt.plot(iterations, lgb_mins, marker=marker, label=i + '_lgb', markersize=8)
        plt.plot(iterations, svc_mins, marker=marker, label=i + '_scv', markersize=8)

        print(i + ' best Score:', round(final_result[i]['best_score'], 4), 'with', l, 'iterations')

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.xlabel('Iteration')
    plt.ylabel('AUC')

   # plt.ylim((0.5, 1))
    #plt.xlim((0, 20))
    plt.title('Value of the best sampled CV score')
    plt.grid()
    plt.legend()
