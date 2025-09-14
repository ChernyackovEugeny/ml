import numpy as np
import pandas as pd


# Gaussian discriminant analysis
class GDA():
    def __init__(self, k):
        self.k = k  # кол-во классов от 0 до k-1

    def fit(self, X, y):
        X = np.asarray(X)
        self.N, self.D = X.shape

        y = np.asarray(y).reshape(-1)

        X_k = []

        self.y_prob = []
        self.average = []
        self.cormat = []
        for i in range(self.k):
            indexes = [j for j in range(self.N) if y[j] == i]
            X_k.append(X[indexes])
            self.y_prob.append(len(indexes) / self.N)

        for i in range(self.k):
            # X_k[i], y_k[j]
            average_k = np.sum(X_k[i], axis=0) / X_k[i].shape[0]
            cormat_k = np.cov(X_k[i].T, bias=True)
            self.average.append(average_k)
            self.cormat.append(cormat_k)

        self.y_prob = np.array(self.y_prob)
        self.average = np.array(self.average)
        self.cormat = np.array(self.cormat)

    def predict(self, X):
        X = np.asarray(X)
        self.N, self.D = X.shape

        self.pred_proba = []
        answear = []

        for x in X:
            probs = []
            for i in range(self.k):
                prob_x_rel_y = self.prob_x_rel_y(x, self.average[i], self.cormat[i])
                probs.append(self.y_prob[i] * prob_x_rel_y)

            probs = np.array(probs)
            probs /= np.sum(probs)

            self.pred_proba.append(probs)
            answear.append(np.argmax(probs))

        self.pred_proba = np.array(self.pred_proba)
        answear = np.array(answear)
        return answear

    def prob_x_rel_y(self, x, average, cormat):
        D = len(x)
        proba = 1 / ((2*np.pi)**(D/2) * np.linalg.det(cormat)**(1/2)) * \
                np.exp(-1/2 * (x - average).T @ np.linalg.inv(cormat) @ (x - average))
        return proba
