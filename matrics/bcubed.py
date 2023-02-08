from sklearn.metrics.cluster import contingency_matrix
import numpy as np


def bcubed_f1(true, pred):
    classes = np.unique(true)
    contingency_mtrx = contingency_matrix(true, pred)
    pr = {k: [] for k in classes}
    pr_denom = {k: [] for k in classes}
    re = {k: [] for k in classes}
    for i in range(contingency_mtrx.shape[0]):
        for ii in range(contingency_mtrx.shape[1]):
            pr[classes[i]].append(contingency_mtrx[i][ii]*contingency_mtrx[i][ii] / contingency_mtrx[:, ii].sum())
            pr_denom[classes[i]].append(contingency_mtrx[i][ii])
            re[classes[i]].append(contingency_mtrx[i][ii]*contingency_mtrx[i][ii] / contingency_mtrx[i, :].sum())
    for k, v in pr_denom.items():
        pr[k] = sum(pr[k]) / sum(v)
        re[k] = sum(re[k]) / sum(v)
    aver_pr = sum(pr.values()) / len(pr)
    aver_re = sum(re.values()) / len(re)
    # f = 2*aver_pr*aver_re/(aver_pr+aver_re)
    f = 1 / (0.5*1/aver_pr + (1-0.5)*1/aver_re)
    return f


c1labels = [1, 2, 3, 4]
c2labels = [5, 5, 5, 5]
c3labels = [6]

destr1labels = [1]*len(c1labels) + [2]*len(c2labels) + [2]*len(c3labels)
destr2labels = [1]*len(c1labels) + [1]*len(c3labels) + [2]*len(c2labels)

destrlabels_true = c1labels + c3labels + c2labels

print('Reference: ')
print(destrlabels_true)
print('Destribution 1 - prediction 1: ')
print(destr1labels)
print('Destribution 2 - prediction 2: ')
print(destr2labels)


print('Мешаем хороший кластер грязным сэмплом:')
print('bcuded f1:', bcubed_f1(destrlabels_true, destr1labels))
print('Убираем грязный сэмпл в сборную солянку:')
print('bcuded f1:', bcubed_f1(destrlabels_true, destr2labels))


# Reference:
# [1, 2, 3, 4, 6, 5, 5, 5, 5]
# Destribution 1 - prediction 1:
# [1, 1, 1, 1, 2, 2, 2, 2, 2]
# Destribution 2 - prediction 2:
# [1, 1, 1, 1, 1, 2, 2, 2, 2]
# Мешаем хороший кластер грязным сэмплом:
# bcuded f1: 0.5
# Убираем грязный сэмпл в сборную солянку:
# bcuded f1: 0.5
# precision for destr 1 {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.8, 6: 0.2}
# recal for destr 1 {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
# precision for destr 2 {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 1.0, 6: 0.2}
# recal for destr 2 {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
