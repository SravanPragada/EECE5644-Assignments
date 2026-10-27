import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)

sigma = 0.3
sx = 0.25
sy = 0.25

r = np.sqrt(rng.random())
theta = 2 * np.pi * rng.random()
xT, yT = r * np.cos(theta), r * np.sin(theta)
print("true position =", np.round([xT, yT], 4))

gx, gy = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401))
levels = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for K, ax in zip([1, 2, 3, 4], axes.ravel()):
    ang = 2 * np.pi * np.arange(K) / K
    lx, ly = np.cos(ang), np.sin(ang)

    rmeas = np.zeros(K)
    for i in range(K):
        d = np.sqrt((xT - lx[i]) ** 2 + (yT - ly[i]) ** 2)
        rmeas[i] = d + sigma * rng.standard_normal()
        while rmeas[i] < 0:
            rmeas[i] = d + sigma * rng.standard_normal()

    J = gx ** 2 / sx ** 2 + gy ** 2 / sy ** 2
    for i in range(K):
        di = np.sqrt((gx - lx[i]) ** 2 + (gy - ly[i]) ** 2)
        J = J + (rmeas[i] - di) ** 2 / sigma ** 2

    k = np.argmin(J)
    print("K =", K, "ranges =", np.round(rmeas, 3),
          "MAP (grid) =", np.round([gx.ravel()[k], gy.ravel()[k]], 3),
          "distance to true =", round(np.hypot(gx.ravel()[k] - xT, gy.ravel()[k] - yT), 3))

    cs = ax.contour(gx, gy, J, levels=levels, cmap="viridis")
    ax.clabel(cs, fontsize=7)
    ax.plot(xT, yT, "r+", markersize=14, mew=2, label="true position")
    ax.plot(lx, ly, "ko", markersize=8, mfc="none", mew=2, label="landmarks")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("K = %d" % K)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("q3_contours.png", dpi=150)
plt.show()
