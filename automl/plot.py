import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from util import log, timeit, mprint
import pandas as pd


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


def plot_bar(ax1, result, bar_name="BO"):
    for i, c in zip(result.keys(), ["orange", 'g']):
        time = result[i]['CPU_Time']
        ax1.bar(bar_name, time, width=0.5, alpha=0.8, ec='grey', color=c, ls="--")
        print(bar_name + "         " + i + ' best Score:', round(result[i]['best_score'], 4), 'with', len(result[i]['all_cv_results']), 'iterations')
        ax1.text(bar_name, time + 0.05, '%.0f' % time, ha='center', va='bottom', fontsize=11)


def transfer_GP_to_table(result):
    df = pd.DataFrame()
    df['mean_test_score'] = result["GP"]['CV']['mean_test_score']
    df['std_test_score'] = result["GP"]['CV']['std_test_score']
    df = df.sort_values(by="mean_test_score", ascending=True)
    return df


def transfer_RF_to_table(result):
    df = pd.DataFrame()
    df['mean_test_score'] = result["RF"]['CV']['mean_test_score']
    df['std_test_score'] = result["RF"]['CV']['std_test_score']
    df = df.sort_values(by="mean_test_score", ascending=True)
    return df


def plot_line(ax0, result, t_gp, t_rf, color=["orange", "g"], labelname="BO"):
    ax0.plot(range(1, len(result["GP"]['all_cv_results']) + 1), t_gp['mean_test_score'], label=labelname + "_GP", color=color[0], marker="v", markersize=6)
    r_gp1 = list(map(lambda x: x[0] - x[1], zip(t_gp["mean_test_score"], t_gp["std_test_score"])))
    r_gp2 = list(map(lambda x: x[0] + x[1], zip(t_gp["mean_test_score"], t_gp["std_test_score"])))
    ax0.fill_between(range(1, len(result["GP"]['all_cv_results']) + 1), r_gp1, r_gp2, color=color[0], alpha=0.1)

    ax0.plot(range(1, len(result["RF"]['all_cv_results']) + 1), t_rf['mean_test_score'], label=labelname + "_RF", color=color[1], marker="x", markersize=6)
    r_rf1 = list(map(lambda x: x[0] - x[1], zip(t_rf["mean_test_score"], t_rf["std_test_score"])))
    r_rf2 = list(map(lambda x: x[0] + x[1], zip(t_rf["mean_test_score"], t_rf["std_test_score"])))
    ax0.fill_between(range(1, len(result["RF"]['all_cv_results']) + 1), r_rf1, r_rf2, color=color[1], alpha=0.1)
