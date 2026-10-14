import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

N = 10000
priors = np.array([0.3, 0.3, 0.4])

a = 2.5 / (2 * np.sqrt(2))
means = a * np.array([[1, 1, 1],
                      [1, -1, -1],
                      [-1, 1, -1],
                      [-1, -1, 1]])
covs = [np.array([[1.0, 0.3, 0.0], [0.3, 0.8, 0.1], [0.0, 0.1, 1.2]]),
        np.array([[0.9, -0.2, 0.1], [-0.2, 1.1, 0.0], [0.1, 0.0, 1.0]]),
        np.array([[1.2, 0.0, 0.2], [0.0, 0.9, -0.3], [0.2, -0.3, 1.0]]),
        np.array([[0.8, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 0.9]])]

rng = np.random.default_rng(2026)

labels = rng.choice([1, 2, 3], size=N, p=priors)

comp = np.zeros(N, dtype=int)
comp[labels == 1] = 0
comp[labels == 2] = 1
n3 = np.sum(labels == 3)
comp[labels == 3] = 2 + rng.integers(0, 2, n3)

X = np.zeros((N, 3))
for c in range(4):
    idx = comp == c
    X[idx] = rng.multivariate_normal(means[c], covs[c], np.sum(idx))

np.savez("q2_data.npz", X=X, labels=labels)
print("class counts:", np.sum(labels == 1), np.sum(labels == 2), np.sum(labels == 3))

px = np.zeros((N, 3))
px[:, 0] = multivariate_normal.pdf(X, means[0], covs[0])
px[:, 1] = multivariate_normal.pdf(X, means[1], covs[1])
px[:, 2] = 0.5 * multivariate_normal.pdf(X, means[2], covs[2]) + \
           0.5 * multivariate_normal.pdf(X, means[3], covs[3])

post = px * priors
post = post / post.sum(axis=1, keepdims=True)


def classify(Lam):
    risk = np.zeros((N, 3))
    for i in range(3):
        for j in range(3):
            risk[:, i] += Lam[i, j] * post[:, j]
    return np.argmin(risk, axis=1) + 1


def confusion(dec):
    CM = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            CM[i, j] = np.mean(dec[labels == j + 1] == i + 1)
    return CM


def plot3d(dec, title, fname):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    markers = {1: "o", 2: "^", 3: "s"}
    for j in [1, 2, 3]:
        right = (labels == j) & (dec == labels)
        wrong = (labels == j) & (dec != labels)
        ax.scatter(X[right, 0], X[right, 1], X[right, 2], c="green", marker=markers[j], s=5,
                   alpha=0.4, label="L=%d correct" % j)
        ax.scatter(X[wrong, 0], X[wrong, 1], X[wrong, 2], c="red", marker=markers[j], s=5,
                   alpha=0.6, label="L=%d incorrect" % j)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2)
    plt.savefig(fname, dpi=150)
    plt.show()

loss_01 = np.ones((3, 3)) - np.eye(3)
loss_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
loss_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])

for name, Lam, fname in [("0-1 loss (MAP)", loss_01, "q2_map.png"),
                         ("Lambda10", loss_10, "q2_lambda10.png"),
                         ("Lambda100", loss_100, "q2_lambda100.png")]:
    dec = classify(Lam)
    CM = confusion(dec)
    risk = np.mean(Lam[dec - 1, labels - 1])
    print("\n", name)
    print(np.round(CM, 4))
    print("P(error) =", np.mean(dec != labels), " expected risk =", risk)
    plot3d(dec, name, fname)
