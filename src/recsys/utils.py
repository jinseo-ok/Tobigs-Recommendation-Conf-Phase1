import itertools as it
import time
from contextlib import contextmanager

import git
import numpy as np

import sys
sys.path.append('/home/janice/el/recsys2019/src')

from recsys.log_utils import get_logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


def group_lengths(group_ids):
    return np.array([sum(1 for _ in i) for k, i in it.groupby(group_ids)])


def jaccard(a, b):
    return len(a & b) / (len(a | b) + 1)


def reduce_mem_usage(df, verbose=False, aggressive=False):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8) if aggressive else df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16) if aggressive else df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16) if aggressive else df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def str_split(text: str):
    return text.split()


def group_time(t):
    if t <= 12:
        return t
    else:
        return int(t / 4) * 4


def get_git_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def get_sort_index(row):
    if (row["is_val"] == False) and (row["is_test"] == False):
        find = int(row[i, "clickout_id"]) % 25
        return f"01_train_{find:04d}"
    elif (row["is_val"] == True) and (row["is_test"] == False):
        find = int(row["clickout_id"]) % 2
        return f"02_val_{find:04d}"
    elif row["is_test"] == True:
        find = int(row["clickout_id"]) % 4
        return f"03_test_{find:04d}"
