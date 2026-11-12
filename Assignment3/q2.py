import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(2065)

weights = np.array([0.35, 0.15, 0.30, 0.20])
means = np.array([[0, 0], [1.6, 1.0], [6, 0], [2, 6]], float)
covs = [np.array([[1.0, 0.3], [0.3, 0.8]]),
        np.array([[0.8, -0.2], [-0.2, 1.2]]),
        np.array([[0.6, 0.0], [0.0, 1.4]]),
        np.array([[1.2, 0.4], [0.4, 0.7]])]

d = np.linalg.norm(means[0] - means[1])
s = np.mean(np.linalg.eigvalsh(covs[0])) + np.mean(np.linalg.eigvalsh(covs[1]))
print("distance between means 1 and 2 =", round(d, 3), " sum of average eigenvalues =", round(s, 3))


def generate(N):
    comp = rng.choice(4, size=N, p=weights)
    X = np.zeros((N, 2))
    for k in range(4):
        X[comp == k] = rng.multivariate_normal(means[k], covs[k], np.sum(comp == k))
    return X


def select_order(X):
    N = len(X)
    folds = np.array_split(rng.permutation(N), 10)
    scores = np.zeros(10)
    for m in range(1, 11):
        total = 0.0
        for k in range(10):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(10) if j != k])
            if m > len(tr):
                total = -np.inf
                break
            gmm = GaussianMixture(n_components=m, covariance_type="full", n_init=1,
                                  max_iter=200, random_state=int(rng.integers(1e9)))
            gmm.fit(X[tr])
            total += gmm.score_samples(X[val]).sum()
        scores[m - 1] = total / N
    return int(np.argmax(scores)) + 1


sizes = [10, 100, 1000]
R = 200
counts = {n: np.zeros(10, int) for n in sizes}
for r in range(R):
    for n in sizes:
        counts[n][select_order(generate(n)) - 1] += 1
    if (r + 1) % 20 == 0:
        print("finished", r + 1, "repetitions")

print("\nSelection rate (%) for each order (1-10):")
for n in sizes:
    print("N =", n, np.round(100 * counts[n] / R, 1))

X_show = generate(1000)
plt.figure(figsize=(6, 5))
plt.scatter(X_show[:, 0], X_show[:, 1], s=4, alpha=0.5)
plt.title("Samples from the true GMM (N = 1000)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.savefig("q2_data.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
orders = np.arange(1, 11)
for i, n in enumerate(sizes):
    plt.bar(orders + (i - 1) * 0.27, 100 * counts[n] / R, width=0.27, label="N = %d" % n)
plt.xticks(orders)
plt.xlabel("selected number of components")
plt.ylabel("selection rate (%)")
plt.title("GMM order selected by 10-fold CV (%d repetitions)" % R)
plt.legend()
plt.grid(True, axis="y")
plt.savefig("q2_selection.png", dpi=150)
plt.show()
