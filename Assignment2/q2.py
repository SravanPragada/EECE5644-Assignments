import numpy as np
import matplotlib.pyplot as plt


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x


def generateDataFromGMM(N, gmmParameters):
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]
    C = len(priors)
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[0, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))
    return x, labels


np.random.seed(2026)
data = generateData(100)
xTrain, yTrain = data[0:2, :], data[2, :]
data = generateData(1000)
xVal, yVal = data[0:2, :], data[2, :]

def z(x):
    x1, x2 = x[0], x[1]
    return np.column_stack([np.ones(len(x1)), x1, x2, x1**2, x1*x2, x2**2,
                            x1**3, x1**2*x2, x1*x2**2, x2**3])


Ztr = z(xTrain)
Zval = z(xVal)

w_ml = np.linalg.solve(Ztr.T @ Ztr, Ztr.T @ yTrain)
mse_ml = np.mean((yVal - Zval @ w_ml) ** 2)
sigma2 = np.mean((yTrain - Ztr @ w_ml) ** 2)
print("ML: validation MSE =", mse_ml, " sigma^2 estimate =", sigma2)

gammas = np.logspace(-10, 6, 81)
mse_map = []
for g in gammas:
    w_map = np.linalg.solve(Ztr.T @ Ztr + (sigma2 / g) * np.eye(10), Ztr.T @ yTrain)
    mse_map.append(np.mean((yVal - Zval @ w_map) ** 2))
mse_map = np.array(mse_map)
k = np.argmin(mse_map)
print("MAP: best gamma =", gammas[k], " validation MSE =", mse_map[k])
for g in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]:
    i = np.argmin(np.abs(gammas - g))
    print("gamma = %g  MSE = %.4f" % (gammas[i], mse_map[i]))

plt.figure()
plt.semilogx(gammas, mse_map, label="MAP")
plt.axhline(mse_ml, color="r", linestyle="--", label="ML")
plt.xlabel("gamma")
plt.ylabel("average squared error (validation)")
plt.title("Validation error vs gamma")
plt.legend()
plt.grid(True)
plt.savefig("q2_mse_vs_gamma.png", dpi=150)
plt.show()
