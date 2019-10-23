import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from util import log, timeit, mprint
import pandas as pd


def plot_convergence(results, name, color='b', ax=None, ls='-', title=None):
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(title)
    else:
        if 'GP' in str.upper(name):
            ax.set_title("Convergence plot for BO_GP")
        elif 'RF' in str.upper(name):
            ax.set_title("Convergence plot for BO_RF")
        else:
            ax.set_title("Convergence plot for RandomSearch")

    ax.set_xlabel("Number of samples ")
    ax.set_ylabel("loss")
    ax.grid(color='grey', linestyle='--', alpha=0.6)

    #colors = cm.viridis(np.linspace(0.25, 1.0, 10))
    #color = colors[i]
    if name:
        label_name = name
    else:
        label_name = None

    if isinstance(results, pd.DataFrame):
        curr_max = None
        best_ = []
        for index, results in enumerate(results.values):
            # print(results)
            if curr_max is None:
                curr_max_index = index
                curr_max = float(results[1])
                loss = float(curr_max) - 1
                best_.append([-loss, float(results[2])])
                # best_std_.append(results[2])
            elif float(results[1]) > float(curr_max):
                curr_max_index = index
                curr_max = float(results[1])
                #best_.append([float(curr_max), float(results[2])])
                loss = float(curr_max) - 1
                best_.append([-loss, float(results[2])])
            else:
                best_.append(best_[-1])

        ax.plot(range(1, len(best_) + 1), np.mat(best_)
                [:, 0], c=color, lw=2, label=label_name, ls=ls)  # marker=".", markersize=12,
        r1 = []
        r2 = []
        for item in best_:
            r1.append(item[0] + item[1])
            r2.append(item[0] - item[1])
        ax.fill_between(range(1, len(best_) + 1), r1,
                        r2, color=color, alpha=0.05)
    if name:
        ax.legend(loc='best')
    plt.tight_layout()
    return ax


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


def plot_bar(ax1, result, bar_name="BO", kpi="CPU_Time", text=True):
    ax1.bar(bar_name, result['GP'][kpi], width=0.5, alpha=0.8, ec='grey', color="orange", ls="--")
    ax1.text(bar_name, result['GP'][kpi], '%.0f' % result['GP'][kpi], ha='center', va='bottom', fontsize=11)

    ax1.bar(bar_name, result['RF'][kpi], bottom=result['GP'][kpi], width=0.5, alpha=0.8, ec='grey', color='g', ls="--")
    ax1.text(bar_name, result['RF'][kpi] + result['GP'][kpi], '%.0f' % result['RF'][kpi], ha='center', va='bottom', fontsize=11)

    for i, c in zip(result.keys(), ["orange", 'g']):
        if text is True:
            print(bar_name + " " + i + ' best Score:', round(result[i]['best_score'], 4), 'with', len(result[i]['all_cv_results']), 'iterations')
        else:
            pass


def transfer_GP_to_table(result):
    df = pd.DataFrame()
    models = []
    for i in range(len(result['GP']['CV']['params'])):
        models.append(result['GP']['CV']['params'][i]['model'].__class__.__name__)
    df['model'] = models
    df['mean_score'] = result["GP"]['CV']['mean_test_score']
    df['std_score'] = result["GP"]['CV']['std_test_score']
  #  df = df.sort_values(by="model", ascending=True)
    return df


def transfer_RF_to_table(result):
    df = pd.DataFrame()
    models = []
    for i in range(len(result['RF']['CV']['params'])):
        models.append(result['RF']['CV']['params'][i]['model'].__class__.__name__)
    df['model'] = models
    df['mean_test_score'] = result["RF"]['CV']['mean_test_score']
    df['std_test_score'] = result["RF"]['CV']['std_test_score']
   # df = df.sort_values(by="model", ascending=True)
    return df


def plot_line(ax0, result, t_gp, t_rf, color=["orange", "g"], labelname="BO"):
    ax0.plot(range(1, len(result["GP"]['all_cv_results']) + 1), t_gp['mean_test_score'], label=labelname + "_GP", color=color[0], linewidth=2.5)
    r_gp1 = list(map(lambda x: x[0] - x[1], zip(t_gp["mean_test_score"], t_gp["std_test_score"])))
    r_gp2 = list(map(lambda x: x[0] + x[1], zip(t_gp["mean_test_score"], t_gp["std_test_score"])))
    ax0.fill_between(range(1, len(result["GP"]['all_cv_results']) + 1), r_gp1, r_gp2, color=color[0], alpha=0.1)

    ax0.plot(range(1, len(result["RF"]['all_cv_results']) + 1), t_rf['mean_test_score'], label=labelname + "_RF", color=color[1], ls="--", linewidth=2.5)
    r_rf1 = list(map(lambda x: x[0] - x[1], zip(t_rf["mean_test_score"], t_rf["std_test_score"])))
    r_rf2 = list(map(lambda x: x[0] + x[1], zip(t_rf["mean_test_score"], t_rf["std_test_score"])))
    ax0.fill_between(range(1, len(result["RF"]['all_cv_results']) + 1), r_rf1, r_rf2, color=color[1], alpha=0.1)
