import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import eigh

# parameters from the assignment
N = 10000
p0 = 0.35
p1 = 0.65

m0 = np.array([-0.5, -0.5, -0.5, -0.5])
C0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
                      [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0],
                      [0, 0, 0, 2]])

m1 = np.array([1, 1, 1, 1])
C1 = np.array([[1, 0.3, -0.2, 0],
               [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0],
               [0, 0, 0, 3]])

rng = np.random.default_rng(2026)

# pick labels: 1 with prob 0.65, 0 with prob 0.35
labels = (rng.random(N) >= p0).astype(int)
N0 = np.sum(labels == 0)
N1 = np.sum(labels == 1)

# generate samples for each class
X = np.zeros((N, 4))
X[labels == 0] = rng.multivariate_normal(m0, C0, N0)
X[labels == 1] = rng.multivariate_normal(m1, C1, N1)

np.savez("q1_data.npz", X=X, labels=labels)
print(N0, N1)


# ROC: decide 1 if score > t, try every sample score as threshold t
def get_roc(score, labels):
    thresholds = np.concatenate(([-np.inf], np.sort(score), [np.inf]))
    tpr = []
    fpr = []
    for t in thresholds:
        dec = score > t
        tpr.append(np.mean(dec[labels == 1]))
        fpr.append(np.mean(dec[labels == 0]))
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    perr = fpr * p0 + (1 - tpr) * p1
    return thresholds, tpr, fpr, perr


def plot_roc(fpr, tpr, k, fp_t, tp_t, title, fname):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot(fpr[k], tpr[k], "ro", markersize=8, label="min P(error) (empirical)")
    if fp_t is not None:
        plt.plot(fp_t, tp_t, "g^", markersize=8, label="theoretical threshold")
    plt.xlabel("False positive  P(D=1|L=0)")
    plt.ylabel("True positive  P(D=1|L=1)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(fname, dpi=150)
    plt.show()


gamma = p0 / p1   # 0-1 loss
print("theoretical gamma =", gamma)

# ---------- Part A: true pdfs ----------
# using log of the likelihood ratio so the numbers don't underflow
llr = multivariate_normal.logpdf(X, m1, C1) - multivariate_normal.logpdf(X, m0, C0)
thrA, tprA, fprA, perrA = get_roc(llr, labels)
kA = np.argmin(perrA)

# error at the theoretical gamma
decA = llr > np.log(gamma)
tpA = np.mean(decA[labels == 1])
fpA = np.mean(decA[labels == 0])
peA = fpA * p0 + (1 - tpA) * p1

print("Part A: min P(error) =", perrA[kA], "gamma =", np.exp(thrA[kA]),
      "TPR =", tprA[kA], "FPR =", fprA[kA])
print("Part A: theoretical gamma P(error) =", peA, "TPR =", tpA, "FPR =", fpA)
plot_roc(fprA, tprA, kA, fpA, tpA, "Part A: ROC with true pdf", "q1_partA.png")

# ---------- Part B: naive Bayes (diagonal covariances) ----------
C0_nb = np.diag(np.diag(C0))
C1_nb = np.diag(np.diag(C1))
llr_nb = multivariate_normal.logpdf(X, m1, C1_nb) - multivariate_normal.logpdf(X, m0, C0_nb)
thrB, tprB, fprB, perrB = get_roc(llr_nb, labels)
kB = np.argmin(perrB)

decB = llr_nb > np.log(gamma)
tpB = np.mean(decB[labels == 1])
fpB = np.mean(decB[labels == 0])
peB = fpB * p0 + (1 - tpB) * p1

print("Part B: min P(error) =", perrB[kB], "gamma =", np.exp(thrB[kB]),
      "TPR =", tprB[kB], "FPR =", fprB[kB])
print("Part B: theoretical gamma P(error) =", peB, "TPR =", tpB, "FPR =", fpB)
plot_roc(fprB, tprB, kB, fpB, tpB, "Part B: ROC with naive Bayes", "q1_partB.png")

# ---------- Part C: Fisher LDA ----------
# sample estimates of mean and covariance
mu0 = np.mean(X[labels == 0], axis=0)
mu1 = np.mean(X[labels == 1], axis=0)
S0 = np.cov(X[labels == 0].T)
S1 = np.cov(X[labels == 1].T)

# scatter matrices (equal weights)
Sb = np.outer(mu1 - mu0, mu1 - mu0)
Sw = S0 + S1

# generalized eigenvalue problem Sb w = lambda Sw w
evals, evecs = eigh(Sb, Sw)
w = evecs[:, np.argmax(evals)]
if w @ (mu1 - mu0) < 0:   # make class 1 project to larger values
    w = -w
w = w / np.linalg.norm(w)

print("mu0 =", mu0)
print("mu1 =", mu1)
print("S0 =\n", S0)
print("S1 =\n", S1)
print("w_LDA =", w)

y = X @ w
thrC, tprC, fprC, perrC = get_roc(y, labels)
kC = np.argmin(perrC)
print("Part C: min P(error) =", perrC[kC], "tau =", thrC[kC],
      "TPR =", tprC[kC], "FPR =", fprC[kC])
plot_roc(fprC, tprC, kC, None, None, "Part C: ROC with Fisher LDA", "q1_partC.png")
