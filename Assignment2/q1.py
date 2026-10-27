import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

rng = np.random.default_rng(2026)

p0 = 0.6
p1 = 0.4
m01 = np.array([-0.9, -1.1])
m02 = np.array([0.8, 0.75])
m11 = np.array([-1.1, 0.9])
m12 = np.array([0.9, -0.75])
C = np.array([[0.75, 0], [0, 1.25]])


def generate(N):
    labels = (rng.random(N) >= p0).astype(int)
    comp = rng.integers(0, 2, N)
    X = np.zeros((N, 2))
    for lab, comps in [(0, [m01, m02]), (1, [m11, m12])]:
        for k in range(2):
            idx = (labels == lab) & (comp == k)
            X[idx] = rng.multivariate_normal(comps[k], C, np.sum(idx))
    return X, labels


X20, y20 = generate(20)
X200, y200 = generate(200)
X2000, y2000 = generate(2000)
Xval, yval = generate(10000)


def px0(X):
    return 0.5 * multivariate_normal.pdf(X, m01, C) + 0.5 * multivariate_normal.pdf(X, m02, C)


def px1(X):
    return 0.5 * multivariate_normal.pdf(X, m11, C) + 0.5 * multivariate_normal.pdf(X, m12, C)


score = np.log(px1(Xval)) - np.log(px0(Xval))
gamma = p0 / p1
dec = score > np.log(gamma)
perr_opt = np.mean(dec != yval)
tp = np.mean(dec[yval == 1])
fp = np.mean(dec[yval == 0])
print("Part 1: min P(error) =", perr_opt)

thresholds = np.concatenate(([-np.inf], np.sort(score), [np.inf]))
tpr = [np.mean(score[yval == 1] > t) for t in thresholds]
fpr = [np.mean(score[yval == 0] > t) for t in thresholds]

plt.figure()
plt.plot(fpr, tpr, label="ROC curve")
plt.plot(fp, tp, "ro", markersize=8, label="min P(error)")
plt.xlabel("P(D=1|L=0)")
plt.ylabel("P(D=1|L=1)")
plt.title("ROC curve")
plt.legend()
plt.grid(True)
plt.savefig("q1_roc.png", dpi=150)
plt.show()

def z_lin(X):
    return np.column_stack([np.ones(len(X)), X])


def z_quad(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1, x2, x1 ** 2, x1 * x2, x2 ** 2])


def h(Z, w):
    return 1 / (1 + np.exp(-Z @ w))


def train(Z, y, lr=0.1, iters=20000):
    w = np.zeros(Z.shape[1])
    for i in range(iters):
        grad = Z.T @ (h(Z, w) - y) / len(y)
        w = w - lr * grad
    return w


for name, zf in [("linear", z_lin), ("quadratic", z_quad)]:
    for n, X, y in [(20, X20, y20), (200, X200, y200), (2000, X2000, y2000)]:
        w = train(zf(X), y)
        dec = h(zf(Xval), w) > 0.5
        print(name, "N =", n, "P(error) =", np.mean(dec != yval))
