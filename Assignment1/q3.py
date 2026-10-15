# Data (download before running):
#   Wine Quality:  https://archive.ics.uci.edu/dataset/186/wine+quality
#                  -> use winequality-white.csv
#   HAR:           https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
#                  -> unzip "UCI HAR Dataset.zip" so the "UCI HAR Dataset" folder is next to this file
#
# In Colab I downloaded them with:
#   !wget -q -O wine.zip "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
#   !unzip -o -q wine.zip
#   !wget -q -O har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
#   !unzip -o -q har.zip
#   !unzip -o -q "UCI HAR Dataset.zip"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

alpha = 0.05 


def load_wine():
    data = np.genfromtxt("winequality-white.csv", delimiter=";", skip_header=1)
    return data[:, :-1], data[:, -1].astype(int)


def load_har():
    Xtr = np.loadtxt("UCI HAR Dataset/train/X_train.txt")
    ytr = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)
    Xte = np.loadtxt("UCI HAR Dataset/test/X_test.txt")
    yte = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int)

    return np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - mu) / sd


def train(X, y):
    classes = np.unique(y)
    means, covs, priors = [], [], []
    for c in classes:
        Xc = X[y == c]
        mu = np.mean(Xc, axis=0)
        C = np.cov(Xc.T)
        lam = alpha * np.trace(C) / np.linalg.matrix_rank(C)
        C = C + lam * np.eye(X.shape[1]) 
        means.append(mu)
        covs.append(C)
        priors.append(len(Xc) / len(X))
        print("class", c, "n =", len(Xc), "prior = %.4f" % priors[-1], "lambda = %.4g" % lam)
    return classes, means, covs, priors


def classify(X, classes, means, covs, priors):

    scores = np.zeros((len(X), len(classes)))
    for k in range(len(classes)):
        scores[:, k] = multivariate_normal.logpdf(X, means[k], covs[k]) + np.log(priors[k])
    return classes[np.argmax(scores, axis=1)]


def confusion(y, dec, classes):
    CM = np.zeros((len(classes), len(classes)))
    for j in range(len(classes)):
        for i in range(len(classes)):
            CM[i, j] = np.mean(dec[y == classes[j]] == classes[i])
    return CM


def pca_plot(X, y, name):

    Xc = X - X.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(Xc.T))
    order = np.argsort(evals)[::-1]
    Z = Xc @ evecs[:, order[:3]]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    for c in np.unique(y):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=3, alpha=0.5, label=str(c))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=4)
    ax.set_title(name + ": first 2 PCs")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    for c in np.unique(y):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], Z[y == c, 2], s=2, alpha=0.4, label=str(c))
    ax.set_title(name + ": first 3 PCs")
    plt.savefig("q3_" + name + "_pca.png", dpi=150)
    plt.show()


def run(name, X, y):
    print("\n=====", name, "=====")
    X = standardize(X)
    classes, means, covs, priors = train(X, y)
    dec = classify(X, classes, means, covs, priors)
    print("P(error) =", np.mean(dec != y), "(", np.sum(dec != y), "errors )")
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print("Confusion matrix P(D=i|L=j), rows = decision, cols = true label:")
    print(confusion(y, dec, classes))
    pca_plot(X, y, name)



Xw, yw = load_wine()
run("Wine", Xw, yw)

Xh, yh = load_har()
run("HAR", Xh, yh)


print("\n===== alpha check =====")
for a in [0.001, 0.01, 0.05, 0.1]:
    alpha = a
    for name, X, y in [("Wine", standardize(Xw), yw), ("HAR", standardize(Xh), yh)]:
        cl, mu, cv, pr = train(X, y)
        print(name, "alpha =", a, "P(error) =", round(np.mean(classify(X, cl, mu, cv, pr) != y), 4))
alpha = 0.05
