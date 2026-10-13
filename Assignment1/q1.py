import numpy as np

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

rng = np.random.default_rng(1)

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
