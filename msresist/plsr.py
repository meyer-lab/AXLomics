import scipy as sp
import numpy as np
from numpy import sign, log10, abs
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans


###------------ Scaling Matrices ------------------###
'''
Note that the sklearn PLSRegression function already handles scaling
'''


def zscore_columns(matrix):
    matrix_z = np.zeros((matrix.shape[0], matrix.shape[1]))
    for a in range(matrix.shape[1]):
        column = []
        column = matrix[:, a]
        column_mean = np.mean(column)
        column_std = np.std(column)
        matrix_z[:, a] = np.asmatrix([(column - column_mean) / column_std])
    return matrix_z


###------------ Q2Y/R2Y ------------------###
'''
Description
Arguments
Returns
'''


def R2Y_across_components(X, Y, max_comps):
    R2Ys = []
    for b in range(1, max_comps):
        plsr = PLSRegression(n_components=b)
        plsr.fit(X, Y)
        R2Y = plsr.score(X, Y)
        R2Ys.append(R2Y)
    return R2Ys


def Q2Y_across_components(X, Y, max_comps):
    Q2Ys = []
    for b in range(1, max_comps):
        plsr_model = PLSRegression(n_components=b)
        y_pred = cross_val_predict(plsr_model, X, Y, cv=Y.size)
        Q2Ys.append(explained_variance_score(Y, y_pred))
    return Q2Ys


def Q2Y_across_comp_manual(X_z, Y_z, max_comps, sublabel):
    PRESS = 0
    SS = 0
    Q2Ys = []
    for b in range(1, max_comps):
        plsr_model = PLSRegression(n_components=b)
        for train_index, test_index in loo.split(X_z, Y_z):
            X_train, X_test = X_z[train_index], X_z[test_index]
            Y_train, Y_test = Y_z[train_index], Y_z[test_index]
            X_train = zscore_columns(X_train)
            Y_train = sp.stats.zscore(Y_train)
            plsr_model.fit_transform(X_train, Y_train)
            Y_predict_cv = plsr_model.predict(X_test)
            PRESS_i = (Y_predict_cv - Y_test) ** 2
            SS_i = (Y_test) ** 2
            PRESS = PRESS + PRESS_i
            SS = SS + SS_i
        Q2Y = 1 - (PRESS / SS)
        Q2Ys.append(Q2Y)
    return Q2Ys

###------------ Fitting PLSR and CV ------------------###


def PLSR(X, Y, nComponents):
    plsr = PLSRegression(n_components=nComponents)
    X_scores, Y_scores = plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload


def MeasuredVsPredicted_LOOCVplot(X, Y, plsr_model, fig, ax, axs):
    Y_predictions = np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size))
    coeff, pval = sp.stats.pearsonr(list(Y_predictions), list(Y))
    print("Pearson's R: ", coeff, "\n", "p-value: ", pval)
    if ax == "none":
        plt.scatter(Y, np.squeeze(Y_predictions))
        plt.title("Correlation Measured vs Predicted")
        plt.xlabel("Measured Cell Viability")
        plt.ylabel("Predicted Cell Viability")
    else:
        axs[ax].scatter(Y, np.squeeze(Y_predictions))
        axs[ax].set(title="Correlation Measured vs Predicted", xlabel='Actual Y', ylabel='Predicted Y')

###------------ Phosphopeptide Filter ------------------###


def MeanCenterAndFilter(X, header):
    Xf, Xf_protnames, Xf_seqs = [], [], []
    for _, row in X.iterrows():
        m = np.mean(row[2:])
        if any(value <= m / 2 or value >= m * 2 for value in row[2:]):
            centered = np.array(list(map(lambda x: x - m, row[2:])))
            Xf.append(np.array(list(map(lambda x: sign(x) * (np.log10(abs(x) + 1)), centered))))
            Xf_seqs.append(row[0])
            Xf_protnames.append(row[1].split("OS")[0])
    frames = [pd.DataFrame(Xf_seqs), pd.DataFrame(Xf_protnames), pd.DataFrame(Xf)]
    Xf = pd.concat(frames, axis=1)
    Xf.columns = header
    return Xf


def FoldChangeFilter(X, header):
    Xf, Xf_protnames, Xf_seqs = [], [], []
    for idx, row in X.iterrows():
        if any(value <= 0.5 or value >= 2 for value in row[2:]):
            Xf.append(np.array(list(map(lambda x: np.log(x), row[2:]))))
            Xf_seqs.append(row[0])
            Xf_protnames.append(row[1].split("OS")[0])

    frames = [pd.DataFrame(Xf_seqs), pd.DataFrame(Xf_protnames), pd.DataFrame(Xf)]
    Xf = pd.concat(frames, axis=1)
    Xf.columns = header
    return Xf


###------------ Computing Cluster Averages ------------------###

def ClusterAverages(X_, cluster_assignments, nClusters, nObs, ProtNames, peptide_phosphosite):  # XXX: Shouldn't nClusters, nObs be able to come from the other arguments?
    X_FCl = np.insert(X_, 0, cluster_assignments, axis=0)  # 11:96   11 = 10cond + clust_assgms
    X_FCl = np.transpose(X_FCl)  # 96:11
    ClusterAvgs = []
    ClusterAvgs_arr = np.zeros((nClusters, nObs))  # 5:10
    DictClusterToMembers = {}
    for i in range(nClusters):
        CurrentCluster = []
#         DictMemberToSeq = {}
        ClusterMembers = []
        ClusterSeqs = []
        for idx, arr in enumerate(X_FCl):
            if i == arr[0]:  # arr[0] is the location of the cluster assignment of the specific peptide
                CurrentCluster.append(arr)  # array with 96:11, so every arr contains a single peptide's values
                ClusterMembers.append(ProtNames[idx])
                ClusterSeqs.append(peptide_phosphosite[idx])
#                 DictMemberToSeq[ProtNames[idx]] = peptide_phosphosite[idx]
        CurrentCluster_T = np.transpose(CurrentCluster)  # 11:96, so every arr contains a single condition's values (eg. all peptides values within cluster X in Erl)
        CurrentAvgs = []
        for x, arr in enumerate(CurrentCluster_T):
            if x == 0:  # cluster assignments
                continue
            else:
                avg = np.mean(arr)
                CurrentAvgs.append(avg)
        CurrentKey = []
        DictClusterToMembers[i + 1] = (ClusterMembers)
        DictClusterToMembers["Seqs_Cluster_" + str(i + 1)] = ClusterSeqs
        ClusterAvgs_arr[i, :] = CurrentAvgs
        AvgsArr = np.transpose(ClusterAvgs_arr)
    return AvgsArr, DictClusterToMembers


###------------ GridSearch ------------------###
'''
Exhaustive search over specified parameter values for an estimator
'''


def GridSearch_CV(model, X, Y, parameters, cv, scoring=None):
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max
