import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from util import log, timeit, mprint
from optimisation import Optimiser
import numpy as np


def plot_all_cv_result(final_result, kpi='all_cv_results', max_iters=20):
    '''
    plot the result of fisrt optimizer 
    Example
    ----------
    plot_final_result_step1(step_1_result)

    '''
    plt.rcParams["figure.figsize"] = (16, 6)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    markers = '.*xv'
    for i, marker in zip(final_result.keys(), markers):
        l = int(len(final_result[i]['all_cv_results']) / 2)
        iterations = range(1, l + 1)
        lgb_mins = sorted(final_result[i][kpi][:l])
        svc_mins = sorted(final_result[i][kpi][l:])
        ax0 = plt.subplot(gs[0])
        ax0.plot(iterations, lgb_mins, marker=marker, label=i + '_lgb', markersize=8)
        ax0.plot(iterations, svc_mins, marker=marker, label=i + '_scv', markersize=8)
        print(i + ' best Score:', round(final_result[i]['best_score'], 4), 'with', l, 'iterations')

        ax0.set_xlabel('Iterations')
        ax0.set_ylabel('AUC')
        ax0.set_title('Convergence')
        ax0.grid()
        ax0.legend()

        ax1 = plt.subplot(gs[1])
        time = final_result[i]['CPU_Time']
        ax1.bar(i, time, label=i, width=0.5, alpha=0.8, ec='grey', ls="--")
        ax1.axes.get_yaxis().set_visible(False)

        ax1.set_title('CPU Time in seconds')
        ax1.text(i, time + 0.05, '%.0f' % time, ha='center', va='bottom', fontsize=11)
        ax1.legend()
    plt.tight_layout()
