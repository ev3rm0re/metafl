from imblearn.over_sampling import SMOTE, ADASYN
import csv
import sys
import pandas as pd
import numpy as np
from data_process.pre_data import csv2txt
from paths import Paths
from sklearn.neighbors import NearestNeighbors
def SMOTE(X, y, N, k=5,label=0):
    """
    合成少数类过采样技术（SMOTE）
    参数：
        X (numpy数组): 包含数据点的特征矩阵。
        y (numpy数组): 对应的标签数组（多数类为0，少数类为1）。
        N (int): 生成的合成样本数量。
        k (int, 可选): 考虑的最近邻居数量，默认为5。
    返回：
        X_synthetic (numpy数组): 包含生成样本的合成特征矩阵。
        y_synthetic (numpy数组): 合成样本对应的标签数组。
    """
    # 分离多数类和少数类样本
    inlabel=not label
    X_majority=[]
    X_minority=[]
    for i in range(len(X[:,0])):
        if y[i]==label:
            X_minority.append(X[i])
        else:
            X_majority.append(X[i])
    # 计算每个少数类样本需要生成的合成样本数量
    # 如果k大于少数样本数量，则将其减少到可能的最大值
    k = min(k, len(X_minority) - 1)
    # 初始化列表以存储合成样本和相应的标签
    synthetic_samples = []
    synthetic_labels = []
    # 在少数类样本上拟合k近邻
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_minority)
    genernum=-1
    for num in range(N):
        for minority_sample in X_minority:
        # 查找当前少数类样本的k个最近邻居
            genernum=genernum+1
            if genernum>=N:
                break
            _, indices = knn.kneighbors(minority_sample.reshape(1, -1), n_neighbors=k)
        # 随机选择k个邻居并创建合成样本
        #for _ in range(N_per_sample):
            neighbor_index = np.random.choice(indices[0])
            neighbor = X_minority[neighbor_index]
            # 计算当前少数类样本和邻居之间的差异
            difference = neighbor - minority_sample
            # 生成一个0到1之间的随机数
            alpha = np.random.random()
            # 创建一个合成样本作为少数类样本和邻居的线性组合
            synthetic_sample = minority_sample + alpha * difference
            # 将合成样本及其标签追加到列表中
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(1)
        if genernum >= N:
            break
    # 将列表转换为numpy数组
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)
    # 将原始多数类样本与合成样本合并
    #X_balanced = np.concatenate((X_majority, X_synthetic), axis=0)
    y_balanced = np.concatenate((np.zeros(len(X_majority)), y_synthetic), axis=0)
    return X_synthetic, y_balanced


def save_matrix_as_csv(data, version_dir, path_w, label):
    df = pd.read_csv(version_dir / "matrix.csv")
    header_list = list(df.columns)[0:-1]
    header_list.append('error')
    error_list = []
    #for i in len(data[0]):
    #data=np.concatenate([data,label.reshape(-1,1)],axis=1)
    #print(data.head(5))
    with open(path_w, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        data = np.insert(data, len(data[0]), values=label, axis=1)
        #print(data.shape,8)
        p=0
        for d in data:
            p=p+1
            writer.writerow(d)
        print(p)
###################################################################################################
def main(program, version):
    version_dir = Paths.get_d4j_version_dir(program, version)
    print("load coverage information matrix")
    if not (version_dir / "matrix.txt").exists():
        csv2txt(version_dir)
    f1 = open(version_dir / "matrix.txt", 'r', encoding='utf-8')
    f2 = open(version_dir / "error.txt", 'r', encoding='utf-8')
    first_ele = True
    for data in f1.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_x = np.c_[matrix_x, nums]
    f1.close()
    print(matrix_x.shape)

    first_ele = True
    for data in f2.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_y = np.c_[matrix_y, nums]
    f2.close()
    print("matirx_y:")
    print(matrix_y.shape)
    matrix_x = matrix_x.transpose()
    matrix_y = matrix_y.transpose()
    inputs_pre = []
    fail_num = np.sum(matrix_y == 1)
    pass_num = np.sum(matrix_y == 0)
    assert fail_num != pass_num
    if fail_num < pass_num:
        minority_label = 1
        minority_num = fail_num
        gnum =pass_num-fail_num
    else:
        minority_label = 0
        minority_num = pass_num
        gnum =fail_num -  pass_num
    for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == minority_label:
            inputs_pre.append(matrix_x[testcase_num])
            #inputs_pre.append(matrix_y[testcase_num])
    #print(np.array(inputs_pre).shape)
    #inputs = torch.FloatTensor(inputs_pre)
    if inputs_pre == 0:
        print("skip！", nums[version])
        sys.exit()
    print(gnum)
    X_balanced, y_balanced = SMOTE(matrix_x, matrix_y,N=gnum,label= minority_label)
    X_balanced= (X_balanced >= 0.5).astype(int)
    path_w = version_dir / "matrix_smote.csv"
    print(path_w)
    save_matrix_as_csv(X_balanced, version_dir, path_w, minority_label)
    print("generate complete")
if __name__ == "__main__":
    main('clac', '1-f')



