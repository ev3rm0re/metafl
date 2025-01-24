import random

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData


class UndersamplingData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        if len(fail_feature) < len(pass_feature):
            fail_is_minor = True
            major_feature = pass_feature
            minor_feature = fail_feature
        elif len(fail_feature) > len(pass_feature):
            fail_is_minor = False
            major_feature = fail_feature
            minor_feature = pass_feature
        else:
            return self.data_df

        select_num = len(minor_feature)

        major_i = []
        while len(major_i) <= select_num:
            random_i = random.randint(0, len(major_feature) - 1)
            if random_i not in major_i:
                major_i.append(random_i)

        temp_array = np.zeros([select_num, len(self.feature_df.values[0])])
        for i in range(select_num):
            temp_array[i] = major_feature[major_i[i]]

        compose_feature = np.vstack((minor_feature, temp_array))

        if fail_is_minor:
            label_np = np.ones(select_num).reshape((-1, 1))
            gen_label = np.zeros(select_num).reshape((-1, 1))
        else:
            label_np = np.zeros(select_num).reshape((-1, 1))
            gen_label = np.ones(select_num).reshape((-1, 1))
        compose_label = np.vstack((label_np, gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)
        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
        return self.data_df
        # self.file_dir = self.raw_data.file_dir
        # self.data_df.to_csv(file.dir)
