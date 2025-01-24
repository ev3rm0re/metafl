from .metrics import *
from .dl_metrics import *
from .fs_metrics import *


def calc_corr(data, method):
    features_list = list(data.columns)[:-1]
    label = list(data.columns)[-1]
    corr_dict = {}

    if method == 'dstar':
        for feature in features_list:
            corr_dict[feature] = dstar(data[feature], data[label])
    elif method == 'ochiai':
        for feature in features_list:
            corr_dict[feature] = ochiai(data[feature], data[label])
    elif method == 'barinel':
        for feature in features_list:
            corr_dict[feature] = barinel(data[feature], data[label])
    elif method == "ER1":
        for feature in features_list:
            corr_dict[feature] = ER1(data[feature], data[label])
    elif method == "ER5":
        for feature in features_list:
            corr_dict[feature] = ER5(data[feature], data[label])
    elif method == "GP02":
        for feature in features_list:
            corr_dict[feature] = GP02(data[feature], data[label])
    elif method == "GP03":
        for feature in features_list:
            corr_dict[feature] = GP03(data[feature], data[label])
    elif method == "GP19":
        for feature in features_list:
            corr_dict[feature] = GP19(data[feature], data[label])
    elif method == "Jaccard":
        for feature in features_list:
            corr_dict[feature] = Jaccard(data[feature], data[label])
    elif method == "Op2":
        for feature in features_list:
            corr_dict[feature] = Op2(data[feature], data[label])

    elif method == "MLP":
        corr_dict = MLP(data[features_list], data[label])
    elif method == "CNN":
        corr_dict = CNN(data[features_list], data[label])
    elif method == "RNN":
        corr_dict = RNN(data[features_list], data[label])

    elif method == "pearson":
        for feature in features_list:
            corr_dict[feature] = pearson(data[feature], data[label])
    elif method == "spearman":
        for feature in features_list:
            corr_dict[feature] = spearman(data[feature], data[label])
    elif method == "kendall":
        for feature in features_list:
            kendall_corr = stats.kendalltau(data[feature].tolist(), data[label].tolist())
            corr_dict[feature] = round(kendall_corr[0], 6)
    elif method == "chisquare":
        for feature in features_list:
            corr_dict[feature] = chisq(data[feature], data[label])
    elif method == "mutual_information":
        for feature in features_list:
            corr_dict[feature] = NMI(data[feature], data[label])
    elif method == "fisher_score":
        sample = data.iloc[:, :-1].values.tolist()
        label = data.iloc[:, -1].values.tolist()

        fisher_score_list = binary_fisher_score(sample, label)
        corr_dict = dict(zip(features_list, fisher_score_list))
    else:
        raise Exception(f"Argument value error: No method '{method}'")
    return corr_dict
