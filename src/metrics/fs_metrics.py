import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from .metrics import get_N_para


def pearson(feature, label):
    cov = feature.cov(label)
    std_x = feature.std()
    std_y = label.std()
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        pearson_corr = cov / (std_x * std_y)
        return pearson_corr


def spearman(feature, label):
    feature_rank = feature.rank()
    label_rank = label.rank()

    cov = feature_rank.cov(label_rank)
    std_x = feature_rank.std()
    std_y = label_rank.std()
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        spearman_corr = cov / (std_x * std_y)
        return spearman_corr


def kendall(feature, label):
    x = np.array(feature)
    y = np.array(label)

    size = x.size
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = calc_dis(y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()
    xtie = count_tie(x)
    ytie = count_tie(y)

    tot = (size * (size - 1)) // 2

    # tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #     = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

    # tau = min(1., max(-1., tau))
    return tau


def count_tie(vector):
    cnt = np.bincount(vector).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return (cnt * (cnt - 1) // 2).sum()


def mergeSortInversion(data, aux, low, high):
    if low >= high:
        return 0

    mid = low + (high - low) // 2

    leftCount = mergeSortInversion(data, aux, low, mid)
    rightCount = mergeSortInversion(data, aux, mid + 1, high)

    for index in range(low, high + 1):
        aux[index] = data[index]
    count = 0
    i = low
    j = mid + 1
    k = i
    while k <= high:
        if i > mid and j <= high:
            data[k] = aux[j]
            j += 1
        elif j > high and i <= mid:
            data[k] = aux[i]
            i += 1
        elif aux[i] <= aux[j]:
            data[k] = aux[i]
            i += 1
        elif aux[i] > aux[j]:
            data[k] = aux[j]
            j += 1
            count += mid - i + 1
        k += 1

    return leftCount + rightCount + count


def calc_dis(y):
    aux = [y[i] for i in range(len(y))]
    nSwap = mergeSortInversion(y, aux, 0, len(y) - 1)
    return nSwap


def chisq(feature, label):
    obs = list(get_N_para(feature, label))

    fail_ratio = np.sum(label) / len(label)
    success_ratio = 1 - fail_ratio

    cover = np.sum(feature)
    uncover = len(feature) - cover

    Ncf = cover * fail_ratio
    Nuf = uncover * fail_ratio
    Ncs = cover * success_ratio
    Nus = uncover * success_ratio
    exp = [Ncf, Nuf, Ncs, Nus]

    return 1 - stats.chisquare(obs, exp)[1]


def NMI(feature, label):
    return metrics.normalized_mutual_info_score(feature, label)


def binary_fisher_score(sample, label):
    if len(sample) != len(label):
        print('Sample does not match label')
        exit()
    df1 = pd.DataFrame(sample)
    df2 = pd.DataFrame(label, columns=['label'])
    data = pd.concat([df1, df2], axis=1)
    data0 = data[data.label == 0]
    data1 = data[data.label == 1]
    n = len(label)
    n1 = sum(label)
    n0 = n - n1
    lst = []
    features_list = list(data.columns)[:-1]
    for feature in features_list:

        m0_feature_mean = data0[feature].mean()
        m0_SW = sum((data0[feature] - m0_feature_mean) ** 2)

        m1_feature_mean = data1[feature].mean()

        m1_SW = sum((data1[feature] - m1_feature_mean) ** 2)

        m_all_feature_mean = data[feature].mean()

        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2

        m_SB = m1_SB + m0_SB

        m_SW = (m0_SW + m1_SW) / n
        if m_SW == 0:

            m_fisher_score = np.nan
        else:

            m_fisher_score = m_SB / m_SW

        lst.append(m_fisher_score)

    return lst
