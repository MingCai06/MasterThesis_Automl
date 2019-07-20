import time
from typing import Any
import os
from os.path import join, isfile
import sys
from sklearn.externals.joblib import dump, load


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


def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")


# log function
def mprint(msg):
    """info"""
    from datetime import datetime
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")


# # wirte result
# def init_dirs():

#     if len(sys.argv) == 1:
#         # default local
#         root_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
#         dirs = {
#             'data': join(root_dir, 'data'),
#             'output': join(root_dir, 'result_output'),
#             'prediction': join(root_dir, 'predictions')
#         }

#     elif len(sys.argv) == 3:
#         # default local
#         root_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
#         dirs = {
#             'data': join(root_dir, 'data'),
#             'output': join(root_dir, 'result_output'),
#             'prediction': join(root_dir, 'predictions')
#         }

#     elif len(sys.argv) == 3:
#         # codalab
#         dirs = {
#             'data': join(sys.argv[1], 'data'),
#             'output': sys.argv[2],
#             'prediction': join(sys.argv[1], 'res')
#         }

#     elif len(sys.argv) == 5 and sys.argv[1] == 'local':
#         # full call in local
#         dirs = {
#             'prediction': join(sys.argv[2]),
#             'ref': join(sys.argv[3]),
#             'output': sys.argv[4]
#         }
#     else:
#         raise ValueError("Wrong number of arguments")

#     os.makedirs(dirs['output'], exist_ok=True)
#     return dirs


def dump_result(data, custom_name=None, save_with_time=False):
    if len(sys.argv) == 3:
        # default local
        ROOT_DIR = os.getcwd()
        DIRS = {
            'input': join(ROOT_DIR, 'data'),
            'output': join(ROOT_DIR, 'result_output'),
        }
    path = DIRS['output']
    datanames = listdirInMac(DIRS['input'])[0].split(".")[0]
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
