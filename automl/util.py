import time
from typing import Any
import os
from os.path import join, isfile
import sys
from sklearn.externals.joblib import dump, load
import pandas as pd
import numpy as np

nesting_level = 0
is_start = None


class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)


def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

# log function


def mprint(msg):
    """info"""
    from datetime import datetime
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")


def dump_result(data, datanames, custom_name=None, save_with_time=False):
    if len(sys.argv) == 3:
        # default local
        ROOT_DIR = os.getcwd()
        DIRS = {
            'input': join(ROOT_DIR, 'data'),
            'output': join(ROOT_DIR, 'result_output'),
        }
    path = DIRS['output']
    #datanames = listdirInMac(DIRS['input'])[0].split(".")[0]
    datanames = datanames
    timestr = time.strftime("%Y%m%d%H%M%S")
    if save_with_time is True:
        filename = datanames + '_' + custom_name + '_' + timestr + '.json'
    else:
        filename = datanames + '_' + custom_name + '.json'
    dump(data, (path + '/' + filename))
    print("Dump successful! File Name:", filename)


def load_result(filename):
    if len(sys.argv) == 3:
        # default local
        ROOT_DIR = os.getcwd()
        DIRS = {
            'input': join(ROOT_DIR, 'data'),
            'output': join(ROOT_DIR, 'result_output'),
        }
    path = DIRS['output']
    return load(path + '/' + filename)


def listdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list


def result_summary(random_result):
    r = pd.DataFrame()
    all_row = []
    for k in random_result:
        try:
            for s, t in zip(random_result[k]['all_cv_results'], random_result[k]['test_score_std']):
                row = np.array([k, s, t])
                all_row.append(row)
        except:
            for s, t in zip(random_result[k]['all_cv_results'], random_result[k]['CV']['std_score_time']):
                row = np.array([k, s, t])
                all_row.append(row)
    r = r.append(all_row)
    r.columns = ['model', 'mean_score', 'std_score']
    r["mean_score"] = r["mean_score"].astype(float)
    r["std_score"] = r["std_score"].astype(float)
    return r
