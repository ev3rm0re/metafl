import numpy as np
np.seterr(divide='ignore',invalid='ignore')

# Ncf, Nuf, Ncs, and Nus
def get_N_para(feature, label):
    # binary covMatrix and error vector
    feature = np.array(feature)
    label = np.array(label)
    Ncf = np.sum(feature[label == 1])
    Nuf = np.sum(label == 1) - Ncf

    Ncs = np.sum(feature[label == 0])
    Nus = np.sum(label == 0) - Ncs


    return Ncf, Nuf, Ncs, Nus


def dstar(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf ** 2 / (Ncs + Nuf)


# Ochiai
def ochiai(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    div = (Ncf + Nuf) * (Ncf + Ncs)
    div = 0 if div < 0 else div
    return Ncf / np.sqrt(div)


# Barinel
def barinel(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return 1 - Ncs / (Ncs + Ncf)


# ER1
def ER1(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf * (1 + Ncs / (2 * Ncs + Ncf))


# ER5
def ER5(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf * (Ncf + Nuf + Ncs + Nus)


# GP02
def GP02(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    Nus = 0 if Nus < 0 else Nus
    Ncs = 0 if Ncs < 0 else Ncs
    return 2 * (Ncf + np.sqrt(Nus)) + np.sqrt(Ncs)


# GP03
def GP03(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    Ncs = 0 if Ncs < 0 else Ncs
    return np.sqrt(np.abs(Ncf * Ncf - np.sqrt(Ncs)))


# GP19
def GP19(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf * np.sqrt(np.abs(Ncs - Ncf + Nuf - Nus))


# Jaccard
def Jaccard(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf / (Nuf + Ncs + Nus)


def Op2(feature, label):
    Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
    return Ncf - Ncs / (Ncs + Nus + 1)
