import os
import numpy as np
import pandas as pd
from pathlib import Path
import globalvar as gl
import config

path = gl.get_value('path')
path_raw = gl.get_value('path_raw')
method = gl.get_value('method')

save_r_path = gl.get_value('save_r_path')
Slice_rank_path = gl.get_value('Slice_rank_path')
def find_rank(method1, rank_final_method):
    csvpath_list = np.array(os.listdir(path_raw))
    list1 = []
    for csv_file in csvpath_list:
        if os.path.getsize(path + str(csv_file) + "/" + rank_final_method) > 0:
            df = pd.read_csv(path_raw + str(csv_file) + '/' + rank_final_method, header=0)
            df.insert(loc=0, column='version', value=csv_file)
            array1 = np.array(df)
            list2 = array1.tolist()
            list1.append(list2)
    list3 = []
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            list3.append(list1[i][j])
    names = ['version', 'rank', 'suspicious', 'bugline']
    test = pd.DataFrame(columns=names, data=list3)
    Path(save_r_path + method1).mkdir(parents=True, exist_ok=True)
    test.to_csv(save_r_path + method1 + "/r_all.csv", index=0)


if __name__ == '__main__':
    find_rank()
