import numpy as np
import pandas as pd
import torch.cuda

from metrics.calc_corr import calc_corr
import globalvar as gl
import config

# file_name = gl.get_value('file_name')
path = gl.get_value('path')
path_raw = gl.get_value('path_raw')


def rank(data, method, rank_path):
    corr_dict = calc_corr(data, method)
    corr_dict_tuplelist = list(zip(corr_dict.values(),
                                   corr_dict.keys()))
    corr_dict_tuplelist_sorted = sorted(corr_dict_tuplelist,
                                        reverse=True)
    names = ['rate', 'line']
    test = pd.DataFrame(columns=names, data=corr_dict_tuplelist_sorted)
    test.replace(np.inf, np.nan, inplace=True)
    test.sort_values(by='rate', axis=0, ascending=False, na_position="last", inplace=True)
    test = test.reset_index(drop=True)
    test.index = test.index + 1
    test.insert(loc=0, column='rank', value=test.index)
    test.to_csv(rank_path, index=0)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    rank()
