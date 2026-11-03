import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, check_grad

rng = np.random.default_rng(2065)

C = 4
means = 1.05 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
covs = [np.array([[1.0, 0.3, 0.1], [0.3, 0.8, 0.0], [0.1, 0.0, 1.2]]),
        np.array([[0.9, -0.2, 0.0], [-0.2, 1.1, 0.2], [0.0, 0.2, 0.8]]),
        np.array([[1.2, 0.0, -0.3], [0.0, 0.9, 0.1], [-0.3, 0.1, 1.0]]),
        np.array([[0.8, 0.1, 0.2], [0.1, 1.0, -0.2], [0.2, -0.2, 1.1]])]


def generate(N):
    y = rng.integers(0, C, N)
    X = np.zeros((N, 3))
    for c in range(C):
        X[y == c] = rng.multivariate_normal(means[c], covs[c], np.sum(y == c))
    return X, y


train_sizes = [100, 500, 1000, 5000, 10000]
train = {n: generate(n) for n in train_sizes}
Xtest, ytest = generate(100000)

logp = np.column_stack([multivariate_normal.logpdf(Xtest, means[c], covs[c]) for c in range(C)])
perr_opt = np.mean(np.argmax(logp, axis=1) != ytest)
print("Optimal classifier test P(error) =", perr_opt)


def unpack(w, P):
    i = 0
    W1 = w[i:i + 3 * P].reshape(3, P); i += 3 * P
    b1 = w[i:i + P]; i += P
    W2 = w[i:i + P * C].reshape(P, C); i += P * C
    b2 = w[i:i + C]
    return W1, b1, W2, b2


def elu(a):
    return np.where(a > 0, a, np.exp(np.minimum(a, 0)) - 1)


def forward(w, X, P):
    W1, b1, W2, b2 = unpack(w, P)
    a = X @ W1 + b1
    h = elu(a)
    s = h @ W2 + b2
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=1, keepdims=True)
    return a, h, p


def loss_grad(w, X, Y, P):
    W1, b1, W2, b2 = unpack(w, P)
    a, h, p = forward(w, X, P)
    N = len(X)
    loss = -np.sum(Y * np.log(p + 1e-12)) / N
    ds = (p - Y) / N
    gW2 = h.T @ ds
    gb2 = ds.sum(axis=0)
    dh = ds @ W2.T
    da = dh * np.where(a > 0, 1, np.exp(np.minimum(a, 0)))
    gW1 = X.T @ da
    gb1 = da.sum(axis=0)
    return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])


def train_mlp(X, y, P, restarts):
    Y = np.eye(C)[y]
    nw = 3 * P + P + P * C + C
    best_w, best_loss = None, np.inf
    for r in range(restarts):
        w0 = rng.normal(0, 0.5, nw)
        res = minimize(loss_grad, w0, args=(X, Y, P), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        if res.fun < best_loss:
            best_loss, best_w = res.fun, res.x
    return best_w, best_loss


def predict(w, X, P):
    return np.argmax(forward(w, X, P)[2], axis=1)


Xc, yc = train[100]
Yc = np.eye(C)[yc]
w_test = rng.normal(0, 0.5, 3 * 4 + 4 + 4 * C + C)
print("gradient check error =", check_grad(lambda w: loss_grad(w, Xc, Yc, 4)[0],
                                           lambda w: loss_grad(w, Xc, Yc, 4)[1], w_test))

P_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
best_P = {}
for n in train_sizes:
    X, y = train[n]
    folds = np.array_split(rng.permutation(n), 10)
    cv_err = []
    for P in P_list:
        errs = []
        for k in range(10):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(10) if j != k])
            w, _ = train_mlp(X[tr], y[tr], P, restarts=2)
            errs.append(np.mean(predict(w, X[val], P) != y[val]))
        cv_err.append(np.mean(errs))
    best_P[n] = P_list[int(np.argmin(cv_err))]
    print("N =", n, "CV P(error) per P:", np.round(cv_err, 4), "-> best P =", best_P[n])

test_err = []
for n in train_sizes:
    X, y = train[n]
    w, loss = train_mlp(X, y, best_P[n], restarts=10)
    e = np.mean(predict(w, Xtest, best_P[n]) != ytest)
    test_err.append(e)
    print("N =", n, "P =", best_P[n], "train loss =", round(loss, 4), "test P(error) =", e)

plt.figure()
plt.semilogx(train_sizes, test_err, "o-", label="trained MLP")
plt.axhline(perr_opt, color="r", linestyle="--", label="theoretically optimal")
plt.xlabel("number of training samples")
plt.ylabel("test P(error)")
plt.title("Test P(error) vs training set size")
plt.legend()
plt.grid(True)
plt.savefig("q1_perror.png", dpi=150)
plt.show()
