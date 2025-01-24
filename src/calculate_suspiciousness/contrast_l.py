import os
import numpy as np
import pandas as pd
import globalvar as gl
import config
method = gl.get_value('method')


def contrast_line(
        rank_temporary_path, rank_final_method_path, bugline_path
):
    if os.path.getsize(rank_temporary_path) > 0:
        df1 = pd.read_csv(bugline_path, names=['line'])
        df2 = pd.read_csv(rank_temporary_path, index_col=None, header=0)
        list1 = df1['line'].tolist()
        array1 = np.array(df2)
        list2 = array1.tolist()
        list3 = []
        for j in range(len(list2)):
            for i in range(len(list1)):
                if list1[i]==list2[j][2]:
                    list3.append(list2[j])
        names = ['rank', 'rate', 'line']
        test = pd.DataFrame(columns=names,data=list3)
        test.to_csv(rank_final_method_path, index=0)
        df = pd.read_csv(rank_final_method_path, index_col=None)
        cols = ['rate']
        df = pd.merge(
            df.groupby('line', as_index=False)[cols].max(),
            df,
            how='left'
        ).drop_duplicates(subset=['line','rate'], keep='first')
        df=df[['rank','rate','line']]
        df.to_csv(rank_final_method_path, index=0)
if __name__ == "__main__":
    contrast_line()
