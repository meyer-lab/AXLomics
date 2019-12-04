""" Clustering functions. """

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from msresist.sequence_analysis import EM_clustering


class MyOwnKMEANS(BaseEstimator):
    """ Runs k-means providing the centers and cluster members and sequences. """

    def __init__(self, n_clusters):
        """ Define variables. """
        self.n_clusters = n_clusters

    def fit(self, X, _):
        """ fit data into k-means. """
        self.kmeans_ = KMeans(n_clusters=self.n_clusters).fit(X.T)
        return self

    def transform(self, X):
        """ calculate cluster averages. """
        centers, _ = ClusterAverages(X, self.kmeans_.labels_)
        return centers

    def clustermembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster. """
        _, clustermembers = ClusterAverages(X, self.kmeans_.labels_)
        return clustermembers


class MyOwnGMM(BaseEstimator):
    """ Runs GMM providing the centers and cluster members and sequences. """

    def __init__(self, n_components):
        """ Define variables """
        self.n_components = n_components

    def fit(self, X, _):
        """ fit data into GMM. """
        self.gmm_ = GaussianMixture(n_components=self.n_components, covariance_type="full").fit(X.T)
        self.labels_ = self.gmm_.predict(X.T)
        return self

    def transform(self, X):
        """ calculate cluster averages. """
        centers, _ = ClusterAverages(X, self.labels_)
        return centers

    def clustermembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster. """
        _, clustermembers = ClusterAverages(X, self.labels_)
        return clustermembers

    def probs(self, X):
        """ probabilities of cluster assignment. """
        return self.gmm_.predict_proba(X.T)

    def weights(self):
        """ weights of each cluster. """
        return self.gmm_.weights_


class MassSpecClustering(BaseEstimator):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. GMMweight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """

    def __init__(self, seqs, names, ncl, GMMweight=1, pYTS="Y", distance_method="PAM250", covariance_type="diag", max_n_iter=20):
        self.seqs = seqs
        self.names = names
        self.ncl = ncl
        self.GMMweight = GMMweight
        self.pYTS = pYTS
        self.distance_method = distance_method
        self.covariance_type = covariance_type
        self.max_n_iter = max_n_iter

    def fit(self, X, _):
        """ Compute EM clustering. """
        self.Cl_seqs_, self.labels_, self.scores_, self.IC_, self.n_iter_ = EM_clustering(X, self.seqs,
                                                                                          self.names, self.ncl, self.GMMweight, self.pYTS, self.distance_method, self.covariance_type, self.max_n_iter)
        return self

    def transform(self, X):
        """ calculate cluster averages. """
        check_is_fitted(self, ["Cl_seqs_", "labels_", "scores_", "IC_", "n_iter_"])
        centers, _ = ClusterAverages(X, self.labels_)
        return centers

    def clustermembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster. """
        check_is_fitted(self, ["Cl_seqs_", "labels_", "scores_", "IC_", "n_iter_"])
        _, clustermembers = ClusterAverages(X, self.labels_)
        return clustermembers

    def predict(self, X, _Y=None):
        """ Predict the cluster each sequence in ABC belongs to."""
        check_is_fitted(self, ["Cl_seqs_", "labels_", "scores_", "IC_", "n_iter_"])
        _, labels, _, _, _ = EM_clustering(X, self.ncl, self.seqs, self.names, self.GMMweight,
                                           self.pYTS, self.distance_method, self.covariance_type, self.max_n_iter)
        return labels

    def score(self, X, _Y=None):
        """ Scoring method, mean of combined p-value of all peptides"""
        check_is_fitted(self, ["Cl_seqs_", "labels_", "scores_", "IC_", "n_iter_"])
        _, _, scores, _, _ = EM_clustering(X, self.ncl, self.seqs, self.names,
                                           self.GMMweight, self.pYTS, self.distance_method, self.covariance_type, self.max_n_iter)
        return np.mean(scores)

    def get_params(self, deep=True):
        """ Returns a dict of the estimator parameters with their values. """
        return {"seqs": self.seqs, "names": self.names, "ncl": self.ncl,
                "GMMweight": self.GMMweight, "pYTS": self.pYTS, "distance_method": self.distance_method,
                "covariance_type": self.covariance_type, "max_n_iter": self.max_n_iter}

    def set_params(self, **parameters):
        """ Necessary to make this estimator scikit learn-compatible."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def ClusterAverages(X, labels):
    """ calculate cluster averages and dictionary with cluster members and sequences. """
    X = X.T.assign(cluster=labels)
    centers = []
    dict_clustermembers = {}
    for i in range(0, max(labels) + 1):
        centers.append(list(X[X["cluster"] == i].iloc[:, :-1].mean()))
        dict_clustermembers["Cluster_" + str(i + 1)] = list(X[X["cluster"] == i].iloc[:, 1])
        dict_clustermembers["seqs_Cluster_" + str(i + 1)] = list(X[X["cluster"] == i].iloc[:, 0])
        dict_clustermembers["pos_Cluster_" + str(i + 1)] = list(X[X["cluster"] == i].iloc[:, -2])

    return pd.DataFrame(centers).T, pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_clustermembers.items()]))
